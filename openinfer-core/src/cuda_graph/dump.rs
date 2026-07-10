use std::collections::HashMap;
use std::ffi::CStr;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::{self, CUgraphNode};

use super::{CudaGraphState, check};

#[derive(Clone, Debug)]
pub struct CudaGraphDumpSummary {
    pub nodes: usize,
    pub edges: usize,
    pub kernels: usize,
    pub dot_path: PathBuf,
    pub png_path: PathBuf,
}

struct GraphDescription {
    nodes: Vec<GraphNode>,
    edges: Vec<(usize, usize)>,
}

struct GraphNode {
    kind: GraphNodeKind,
}

#[derive(Clone, Copy)]
struct RepeatedRun {
    start: usize,
    width: usize,
    repetitions: usize,
}

enum GraphNodeKind {
    Kernel {
        raw_symbol: String,
        demangled: String,
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_shared_mem_bytes: u32,
    },
    Other {
        node_type: String,
    },
}

pub fn validate_graph_dump_request(png_path: &Path) -> Result<()> {
    ensure!(
        png_path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png")),
        "--dump-graph-png expects a .png output path, got {}",
        png_path.display()
    );
    let parent = png_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).with_context(|| {
        format!(
            "create CUDA Graph dump output directory {}",
            parent.display()
        )
    })?;
    require_graph_dump_driver()?;
    require_tool("dot", &["-Tpng:cairo"], "Graphviz Cairo PNG renderer")?;
    require_tool("c++filt", &["--version"], "C++ demangler")?;
    Ok(())
}

fn require_graph_dump_driver() -> Result<()> {
    const MIN_DRIVER_API_VERSION: i32 = 12_030;

    let mut version = 0i32;
    check(
        unsafe { sys::cuDriverGetVersion(&raw mut version) },
        "cuDriverGetVersion",
    )?;
    ensure!(
        version >= MIN_DRIVER_API_VERSION,
        "--dump-graph-png requires CUDA driver API 12.3 or newer for kernel names; found {}",
        format_driver_api_version(version)
    );
    Ok(())
}

fn format_driver_api_version(version: i32) -> String {
    format!("{}.{}", version / 1000, version % 1000 / 10)
}

fn require_tool(program: &str, args: &[&str], description: &str) -> Result<()> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("{description} `{program}` is required for CUDA Graph export"))?;
    ensure!(
        output.status.success(),
        "{description} `{program}` check failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

impl CudaGraphState {
    pub fn dump_png(&self, png_path: &Path, title: &str) -> Result<CudaGraphDumpSummary> {
        ensure!(
            self.is_captured(),
            "CUDA graph dump requested before capture"
        );
        let graph = self.inspect()?;
        let dot_path = png_path.with_extension("dot");
        std::fs::write(&dot_path, graph.detailed_dot())
            .with_context(|| format!("write detailed CUDA Graph DOT to {}", dot_path.display()))?;
        render_png(&graph.human_dot(title), png_path)?;
        Ok(CudaGraphDumpSummary {
            nodes: graph.nodes.len(),
            edges: graph.edges.len(),
            kernels: graph
                .nodes
                .iter()
                .filter(|node| matches!(&node.kind, GraphNodeKind::Kernel { .. }))
                .count(),
            dot_path,
            png_path: png_path.to_path_buf(),
        })
    }

    fn inspect(&self) -> Result<GraphDescription> {
        let handles = graph_nodes(self.graph)?;
        let handle_to_index: HashMap<usize, usize> = handles
            .iter()
            .enumerate()
            .map(|(index, &handle)| (handle as usize, index))
            .collect();
        let raw_kinds = handles
            .iter()
            .map(|&handle| inspect_node(handle))
            .collect::<Result<Vec<_>>>()?;
        let symbols = raw_kinds
            .iter()
            .filter_map(|kind| match kind {
                RawNodeKind::Kernel { raw_symbol, .. } => Some(raw_symbol.as_str()),
                RawNodeKind::Other { .. } => None,
            })
            .collect::<Vec<_>>();
        let demangled = demangle(&symbols)?;
        let mut demangled = demangled.into_iter();
        let nodes = raw_kinds
            .into_iter()
            .map(|kind| GraphNode {
                kind: match kind {
                    RawNodeKind::Kernel {
                        raw_symbol,
                        grid,
                        block,
                        dynamic_shared_mem_bytes,
                    } => GraphNodeKind::Kernel {
                        raw_symbol,
                        demangled: demangled
                            .next()
                            .expect("one demangled name per kernel node"),
                        grid,
                        block,
                        dynamic_shared_mem_bytes,
                    },
                    RawNodeKind::Other { node_type } => GraphNodeKind::Other { node_type },
                },
            })
            .collect();
        let edges = graph_edges(self.graph)?
            .into_iter()
            .map(|(from, to)| {
                let from = handle_to_index
                    .get(&(from as usize))
                    .copied()
                    .context("CUDA graph edge source is absent from the node list")?;
                let to = handle_to_index
                    .get(&(to as usize))
                    .copied()
                    .context("CUDA graph edge destination is absent from the node list")?;
                Ok((from, to))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(GraphDescription { nodes, edges })
    }
}

enum RawNodeKind {
    Kernel {
        raw_symbol: String,
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_shared_mem_bytes: u32,
    },
    Other {
        node_type: String,
    },
}

fn graph_nodes(graph: sys::CUgraph) -> Result<Vec<CUgraphNode>> {
    let mut count = 0usize;
    check(
        unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &raw mut count) },
        "cuGraphGetNodes(count)",
    )?;
    let mut nodes = vec![std::ptr::null_mut(); count];
    check(
        unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &raw mut count) },
        "cuGraphGetNodes(nodes)",
    )?;
    nodes.truncate(count);
    Ok(nodes)
}

fn graph_edges(graph: sys::CUgraph) -> Result<Vec<(CUgraphNode, CUgraphNode)>> {
    let mut count = 0usize;
    check(
        unsafe {
            sys::cuGraphGetEdges(
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &raw mut count,
            )
        },
        "cuGraphGetEdges(count)",
    )?;
    let mut from = vec![std::ptr::null_mut(); count];
    let mut to = vec![std::ptr::null_mut(); count];
    check(
        unsafe { sys::cuGraphGetEdges(graph, from.as_mut_ptr(), to.as_mut_ptr(), &raw mut count) },
        "cuGraphGetEdges(edges)",
    )?;
    Ok(from.into_iter().zip(to).take(count).collect())
}

fn inspect_node(node: CUgraphNode) -> Result<RawNodeKind> {
    let mut node_type = std::mem::MaybeUninit::uninit();
    check(
        unsafe { sys::cuGraphNodeGetType(node, node_type.as_mut_ptr()) },
        "cuGraphNodeGetType",
    )?;
    let node_type = unsafe { node_type.assume_init() };
    if node_type != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
        return Ok(RawNodeKind::Other {
            node_type: format!("{node_type:?}")
                .trim_start_matches("CU_GRAPH_NODE_TYPE_")
                .to_ascii_lowercase(),
        });
    }

    let mut params = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();
    check(
        unsafe { sys::cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()) },
        "cuGraphKernelNodeGetParams",
    )?;
    let params = unsafe { params.assume_init() };
    ensure!(
        !params.func.is_null(),
        "CUDA graph kernel node has no CUfunction"
    );
    let mut name = std::ptr::null();
    check(
        unsafe { sys::cuFuncGetName(&raw mut name, params.func) },
        "cuFuncGetName",
    )?;
    ensure!(!name.is_null(), "cuFuncGetName returned a null name");
    let raw_symbol = unsafe { CStr::from_ptr(name) }
        .to_str()
        .context("CUDA kernel name is not UTF-8")?
        .to_owned();
    Ok(RawNodeKind::Kernel {
        raw_symbol,
        grid: [params.gridDimX, params.gridDimY, params.gridDimZ],
        block: [params.blockDimX, params.blockDimY, params.blockDimZ],
        dynamic_shared_mem_bytes: params.sharedMemBytes,
    })
}

fn demangle(symbols: &[&str]) -> Result<Vec<String>> {
    if symbols.is_empty() {
        return Ok(Vec::new());
    }
    let mut child = Command::new("c++filt")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn C++ demangler `c++filt`")?;
    {
        let stdin = child.stdin.as_mut().context("open c++filt stdin")?;
        for symbol in symbols {
            writeln!(stdin, "{symbol}").context("write symbol to c++filt")?;
        }
    }
    let output = child.wait_with_output().context("wait for c++filt")?;
    ensure!(
        output.status.success(),
        "c++filt failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let names = String::from_utf8(output.stdout)
        .context("c++filt output is not UTF-8")?
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(
        names.len() == symbols.len(),
        "c++filt returned {} names for {} symbols",
        names.len(),
        symbols.len()
    );
    Ok(names)
}

fn render_png(dot: &str, png_path: &Path) -> Result<()> {
    let mut child = Command::new("dot")
        .args(["-Tpng:cairo", "-Gdpi=192", "-o"])
        .arg(png_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn Graphviz `dot`")?;
    child
        .stdin
        .as_mut()
        .context("open Graphviz stdin")?
        .write_all(dot.as_bytes())
        .context("write clean CUDA Graph DOT to Graphviz")?;
    let output = child.wait_with_output().context("wait for Graphviz")?;
    ensure!(
        output.status.success(),
        "Graphviz failed to render {}: {}",
        png_path.display(),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

impl GraphDescription {
    fn detailed_dot(&self) -> String {
        let mut dot = String::from("digraph cuda_graph_detailed {\n");
        dot.push_str("  graph [rankdir=TB];\n  node [shape=box];\n");
        for (index, node) in self.nodes.iter().enumerate() {
            let label = match &node.kind {
                GraphNodeKind::Kernel {
                    raw_symbol,
                    demangled,
                    grid,
                    block,
                    dynamic_shared_mem_bytes,
                } => format!(
                    "id={index}\\ntype=kernel\\nname={}\\nraw_symbol={}\\ngrid={}\\nblock={}\\ndynamic_shared_mem_bytes={dynamic_shared_mem_bytes}",
                    dot_escape(demangled),
                    dot_escape(raw_symbol),
                    dims(*grid),
                    dims(*block),
                ),
                GraphNodeKind::Other { node_type } => {
                    format!("id={index}\\ntype={}", dot_escape(node_type))
                }
            };
            let _ = writeln!(dot, "  n{index} [label=\"{label}\"];");
        }
        for &(from, to) in &self.edges {
            let _ = writeln!(dot, "  n{from} -> n{to};");
        }
        dot.push_str("}\n");
        dot
    }

    fn human_dot(&self, title: &str) -> String {
        let mut indegree = vec![0usize; self.nodes.len()];
        let mut outdegree = vec![0usize; self.nodes.len()];
        for &(from, to) in &self.edges {
            outdegree[from] += 1;
            indegree[to] += 1;
        }

        let mut dot = String::from("digraph cuda_graph {\n");
        let _ = writeln!(
            dot,
            "  graph [rankdir=TB, bgcolor=\"white\", pad=0.2, nodesep=0.25, ranksep=0.35, fontname=\"Helvetica\", label=\"{}\", labelloc=t, fontsize=18];",
            dot_escape(title)
        );
        dot.push_str(
            "  node [shape=box, style=\"rounded,filled\", color=\"#374151\", \
             fontname=\"Helvetica\", fontsize=10, margin=\"0.12,0.08\"];\n",
        );
        dot.push_str("  edge [color=\"#6B7280\", arrowsize=0.6];\n");
        if let Some((order, repeated)) = self.repeated_linear_run() {
            let first_end = repeated.start + repeated.width;
            let repeated_end = repeated.start + repeated.width * repeated.repetitions;
            for &index in &order[..repeated.start] {
                self.write_human_node(&mut dot, index, indegree[index], outdegree[index]);
            }
            let _ = writeln!(
                dot,
                "  subgraph cluster_repeated {{\n    label=\"Repeated physical block ×{} · {} kernels/instance\";\n    color=\"#9CA3AF\";\n    style=\"rounded,dashed\";",
                repeated.repetitions, repeated.width
            );
            for &index in &order[repeated.start..first_end] {
                self.write_human_node(&mut dot, index, indegree[index], outdegree[index]);
            }
            dot.push_str("  }\n");
            for &index in &order[repeated_end..] {
                self.write_human_node(&mut dot, index, indegree[index], outdegree[index]);
            }

            let visible_order = order[..first_end]
                .iter()
                .chain(order[repeated_end..].iter())
                .copied()
                .collect::<Vec<_>>();
            for pair in visible_order.windows(2) {
                let label = if pair[0] == order[first_end - 1] {
                    format!(" [label=\"after ×{}\", fontsize=9]", repeated.repetitions)
                } else {
                    String::new()
                };
                let _ = writeln!(dot, "  n{} -> n{}{};", pair[0], pair[1], label);
            }
        } else {
            for index in 0..self.nodes.len() {
                self.write_human_node(&mut dot, index, indegree[index], outdegree[index]);
            }
            for &(from, to) in &self.edges {
                let _ = writeln!(dot, "  n{from} -> n{to};");
            }
        }
        dot.push_str("}\n");
        dot
    }

    fn write_human_node(&self, dot: &mut String, index: usize, indegree: usize, outdegree: usize) {
        let fill = node_color(indegree, outdegree);
        let label = match &self.nodes[index].kind {
            GraphNodeKind::Kernel {
                demangled,
                grid,
                block,
                dynamic_shared_mem_bytes,
                ..
            } => format!(
                "{}\\ngrid={} block={} dynamic_smem={}",
                dot_escape(&compact_kernel_name(demangled)),
                dims(*grid),
                dims(*block),
                dynamic_shared_mem_bytes,
            ),
            GraphNodeKind::Other { node_type } => dot_escape(&node_type.to_ascii_uppercase()),
        };
        let _ = writeln!(dot, "  n{index} [fillcolor=\"{fill}\", label=\"{label}\"];");
    }

    fn repeated_linear_run(&self) -> Option<(Vec<usize>, RepeatedRun)> {
        let order = self.linear_order()?;
        let signatures = order
            .iter()
            .map(|&index| self.nodes[index].signature())
            .collect::<Vec<_>>();
        let mut best = None::<RepeatedRun>;
        for start in 0..signatures.len() {
            let max_width = (signatures.len() - start) / 3;
            for width in 2..=max_width {
                let pattern = &signatures[start..start + width];
                let mut repetitions = 1usize;
                while start + (repetitions + 1) * width <= signatures.len()
                    && signatures[start + repetitions * width..start + (repetitions + 1) * width]
                        == *pattern
                {
                    repetitions += 1;
                }
                if repetitions < 3 {
                    continue;
                }
                let candidate = RepeatedRun {
                    start,
                    width,
                    repetitions,
                };
                let better = best.is_none_or(|current| {
                    let covered = candidate.width * candidate.repetitions;
                    let current_covered = current.width * current.repetitions;
                    covered > current_covered
                        || (covered == current_covered
                            && candidate.repetitions > current.repetitions)
                });
                if better {
                    best = Some(candidate);
                }
            }
        }
        let repeated = best?;
        (repeated.width * repeated.repetitions >= self.nodes.len() / 2).then_some((order, repeated))
    }

    fn linear_order(&self) -> Option<Vec<usize>> {
        if self.nodes.is_empty() || self.edges.len() + 1 != self.nodes.len() {
            return None;
        }
        let mut indegree = vec![0usize; self.nodes.len()];
        let mut next = vec![None; self.nodes.len()];
        for &(from, to) in &self.edges {
            indegree[to] += 1;
            if indegree[to] > 1 || next[from].replace(to).is_some() {
                return None;
            }
        }
        let roots = indegree
            .iter()
            .enumerate()
            .filter_map(|(index, &degree)| (degree == 0).then_some(index))
            .collect::<Vec<_>>();
        if roots.len() != 1 {
            return None;
        }
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut cursor = Some(roots[0]);
        while let Some(index) = cursor {
            order.push(index);
            cursor = next[index];
        }
        (order.len() == self.nodes.len()).then_some(order)
    }
}

impl GraphNode {
    fn signature(&self) -> String {
        match &self.kind {
            GraphNodeKind::Kernel {
                raw_symbol,
                grid,
                block,
                dynamic_shared_mem_bytes,
                ..
            } => format!(
                "kernel|{raw_symbol}|{}|{}|{dynamic_shared_mem_bytes}",
                dims(*grid),
                dims(*block)
            ),
            GraphNodeKind::Other { node_type } => format!("other|{node_type}"),
        }
    }
}

fn dims(dims: [u32; 3]) -> String {
    format!("({},{},{})", dims[0], dims[1], dims[2])
}

fn node_color(indegree: usize, outdegree: usize) -> &'static str {
    if indegree == 0 {
        "#DCEEFF"
    } else if outdegree == 0 {
        "#E8DEFF"
    } else if indegree > 1 {
        "#FFE8C2"
    } else if outdegree > 1 {
        "#E3F6E8"
    } else {
        "#F5F5F5"
    }
}

fn compact_kernel_name(name: &str) -> String {
    const MAX_CHARS: usize = 72;

    if name.contains("internal::gemvx::kernel") {
        return "cuBLAS GEMV".to_owned();
    }
    let signature = name.rsplit_once('(').map_or(name, |(name, _)| name);
    let mut compact = String::with_capacity(signature.len());
    let mut template_depth = 0usize;
    for ch in signature.chars() {
        match ch {
            '<' => {
                if template_depth == 0 {
                    compact.push_str("<…>");
                }
                template_depth += 1;
            }
            '>' if template_depth > 0 => template_depth -= 1,
            _ if template_depth == 0 => compact.push(ch),
            _ => {}
        }
    }
    let leaf = compact.rsplit("::").next().unwrap_or(&compact);
    if leaf.chars().count() <= MAX_CHARS {
        return leaf.to_owned();
    }
    let prefix = leaf.chars().take(MAX_CHARS - 1).collect::<String>();
    format!("{prefix}…")
}

fn dot_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "")
}

#[cfg(test)]
mod tests {
    use super::{
        GraphDescription, GraphNode, GraphNodeKind, compact_kernel_name, dot_escape,
        format_driver_api_version,
    };

    #[test]
    fn compact_name_drops_namespace_signature_and_template_body() {
        assert_eq!(
            compact_kernel_name("flashinfer::BatchDecodeKernel<128, foo::Bar>(float*, int)"),
            "BatchDecodeKernel<…>"
        );
        assert_eq!(
            compact_kernel_name(
                "std::enable_if<true, void>::type internal::gemvx::kernel<int>(int)"
            ),
            "cuBLAS GEMV"
        );
    }

    #[test]
    fn dot_label_escaping_preserves_graph_syntax() {
        assert_eq!(dot_escape("a\\b\n\"c\""), "a\\\\b\\n\\\"c\\\"");
    }

    #[test]
    fn formats_cuda_driver_api_version() {
        assert_eq!(format_driver_api_version(12_020), "12.2");
        assert_eq!(format_driver_api_version(12_030), "12.3");
        assert_eq!(format_driver_api_version(13_000), "13.0");
    }

    #[test]
    fn human_dot_folds_a_repeated_linear_block() {
        let kernel = |name: &str| GraphNode {
            kind: GraphNodeKind::Kernel {
                raw_symbol: name.to_string(),
                demangled: format!("{name}()"),
                grid: [1, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_mem_bytes: 0,
            },
        };
        let names = ["head", "a", "b", "a", "b", "a", "b", "tail"];
        let graph = GraphDescription {
            nodes: names.into_iter().map(kernel).collect(),
            edges: (0..7).map(|index| (index, index + 1)).collect(),
        };

        let dot = graph.human_dot("test graph");

        assert!(dot.contains("label=\"test graph\""));
        assert!(dot.contains("Repeated physical block ×3 · 2 kernels/instance"));
        assert!(dot.contains("n0 -> n1"));
        assert!(dot.contains("n2 -> n7 [label=\"after ×3\""));
        assert!(!dot.contains("n3 [fillcolor"));
    }
}
