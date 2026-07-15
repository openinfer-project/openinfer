#!/usr/bin/env python3
"""Render a CUDA-graph .dot dump to a folded, human-browsable PNG.

One renderer for both dot dialects we produce, so every engineer gets the
same image from the same input:

  - openinfer detailed dot (from `openinfer --dump-graph-png`, the .dot sidecar)
  - CUDA `cudaGraphDebugDotPrint` verbose dot (e.g. dumped from vLLM)

Repeated per-layer kernel blocks are detected by label signature and folded
into one representative block plus an explicit fold marker, because an
unfolded 36-layer chain exceeds Graphviz's ~32767px raster ceiling
(docs/models/qwen3/cuda-graph-png.md). Rendering is pinned to the Cairo PNG
backend, with DPI auto-scaled down for graphs that stay tall after folding;
a missing renderer is a hard error, not a degraded image.

Usage: uv run python tools/cuda_graph_png.py INPUT.dot [-o OUT.png] [--title TITLE] [--no-fold]

Needs `dot` (graphviz with cairo) and `c++filt` on PATH.
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

def pick_dpi(kept_nodes):
    """Graphviz PNG rasters cap at 32767px; a folded chain runs ~105px/node
    at 192 DPI, so tall graphs must drop DPI to stay renderable."""
    if kept_nodes <= 300:
        return 192
    if kept_nodes <= 600:
        return 96
    return 72

# category -> fill color; category is derived from the kernel's provenance
COLORS = {
    "cublas": "#bfd9f2",
    "flash-attn": "#f9d9b0",
    "flashinfer": "#f9d9b0",
    "triton": "#c8e6c9",
    "vllm": "#e1cff0",
    "openinfer": "#e1cff0",
    "memop": "#e0e0e0",
    "other": "#f5f5f5",
}


@dataclass
class Node:
    nid: str
    order: int  # execution order (topological)
    label: str  # compact two-line display label
    signature: str  # full label (name + launch dims); coarser keys alias
    # across distinct per-layer kernels and produce false periods
    category: str


def demangle_all(symbols):
    raw = [s for s in symbols if s.startswith("_Z")]
    if not raw:
        return {}
    out = subprocess.run(
        ["c++filt"], input="\n".join(raw), capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return dict(zip(raw, out))


def compact_name(demangled):
    """Map a demangled kernel name to a short label + provenance category."""
    d = demangled
    if "internal::gemvx::kernel" in d or "cublas" in d:
        return "cuBLAS GEMV", "cublas"
    m = re.match(r"(?:void )?flash::(\w+)", d)
    if m:
        return f"flash {m.group(1).removesuffix('_kernel')}", "flash-attn"
    m = re.match(r"(?:void )?flashinfer::(?:\w+::)*(\w+)", d)
    if m:
        return f"flashinfer {m.group(1).removesuffix('Kernel')}", "flashinfer"
    m = re.match(r"(?:void )?vllm::(?:\w+::)*(\w+)", d)
    if m:
        return f"vllm {m.group(1).removesuffix('_kernel')}", "vllm"
    m = re.match(r"(?:void )?openinfer::(?:\w+::)*(\w+)", d)
    if m:
        return f"openinfer {m.group(1).removesuffix('Kernel')}", "openinfer"
    if d.startswith("triton"):
        return d.rstrip("_0123456789") or d, "triton"
    base = re.sub(r"<.*", "", d.split("(")[0]).strip().split()[-1]
    return base[:56], "other"


def parse_dim3(text):
    """'{1,16,8}' or '1536' -> '(1,16,8)' / '(1536,1,1)'."""
    text = text.strip()
    if text.startswith("{"):
        return "(" + text.strip("{}") + ")"
    return f"({text},1,1)"


def split_launch(launch):
    """Split 'grid,block,smem' where grid/block may be '{x,y,z}'."""
    parts, depth, cur = [], 0, ""
    for ch in launch:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            depth += ch == "{"
            depth -= ch == "}"
            cur += ch
    parts.append(cur)
    return parts


def parse_cuda_debug_dot(text):
    """cudaGraphDebugDotPrint verbose format: record nodes with (topoId: N)."""
    node_re = re.compile(r'"(graph_\d+_node_\d+)"\[.*?label="\{(.*?)\}"\];', re.S)
    kernel_re = re.compile(r"\(topoId: \d+\) \| (.+?)\\<\\<\\<(.+?)\\>\\>\\>")
    topo_re = re.compile(r"\(topoId: (\d+)\)")

    raw_nodes = []
    for nid, body in node_re.findall(text):
        kind = body.strip().lstrip("{").strip().split("\n")[0].split("|")[0].strip()
        topo = int(topo_re.search(body).group(1))
        km = kernel_re.search(body)
        raw_nodes.append((nid, -topo, kind, km.group(1) if km else "", km.group(2) if km else ""))

    names = demangle_all(sym for _, _, _, sym, _ in raw_nodes)
    nodes = {}
    for nid, order, kind, sym, launch in raw_nodes:
        if sym:
            short, cat = compact_name(names.get(sym, sym))
            grid, block, smem = split_launch(launch)
            detail = f"grid={parse_dim3(grid)} block={parse_dim3(block)} smem={smem}"
        else:
            short, cat, detail = kind.lower(), "memop", ""
        label = f"{short}\\n{detail}".rstrip("\\n")
        nodes[nid] = Node(nid, order, label, label, cat)

    edges = re.findall(r'"(graph_\d+_node_\d+)"\s*->\s*"(graph_\d+_node_\d+)"', text)
    return nodes, edges


def parse_openinfer_dot(text):
    """openinfer detailed sidecar: nN [label="id=..\\ntype=..\\nname=..\\n.."]."""
    # edges also carry [label="from_port=..."]; anchor on the id= body so an
    # edge's target node is never re-parsed as a node definition
    node_re = re.compile(r'^\s*(n\d+) \[label="(id=.*?)"\];?', re.S | re.M)
    nodes, order = {}, 0
    for nid, body in node_re.findall(text):
        fields = dict(
            f.split("=", 1) for f in body.split("\\n") if "=" in f
        )
        kind = fields.get("type", "other")
        if kind == "kernel":
            short, cat = compact_name(fields["name"])
            detail = (
                f"grid={fields.get('grid', '?')} block={fields.get('block', '?')} "
                f"smem={fields.get('dynamic_shared_mem_bytes', '?')}"
            )
            label = f"{short}\\n{detail}"
        else:
            short, cat, label = kind, "memop", kind
        nodes[nid] = Node(nid, order, label, label, cat)
        order += 1
    edges = re.findall(r"(n\d+)\s*->\s*(n\d+)", text)
    return nodes, edges


def detect_folds(signatures, max_period=512, min_cover=12):
    """Scan for every maximal periodic run and return [(start, period, repeats)].

    Real decode graphs are multi-phase (a few dense layers, then a long MoE
    run, then an MTP tail), so a single dominant period misses most of the
    graph; each phase gets its own fold."""
    interned = {}
    ids = [interned.setdefault(s, len(interned)) for s in signatures]
    n = len(ids)
    folds, i = [], 0
    while i < n:
        best = None
        for period in range(4, max_period + 1):
            if i + 2 * period > n:
                break
            if ids[i : i + period] != ids[i + period : i + 2 * period]:
                continue
            reps = 2
            while (
                i + (reps + 1) * period <= n
                and ids[i + reps * period : i + (reps + 1) * period] == ids[i : i + period]
            ):
                reps += 1
            if best is None or period * reps > best[1] * best[2]:
                best = (i, period, reps)
        if best and best[1] * best[2] >= min_cover:
            folds.append(best)
            i = best[0] + best[1] * best[2]
        else:
            i += 1
    return folds


def emit_folded_dot(nodes, edges, title, folds):
    order = sorted(nodes.values(), key=lambda x: x.order)
    removed = set()
    for start, period, reps in folds:
        removed.update(range(start + period, start + period * reps))

    kept = [n for i, n in enumerate(order) if i not in removed]
    kept_ids = {n.nid for n in kept}
    note = "; ".join(f"x{reps} blocks of {period} folded" for _, period, reps in folds)
    lines = [
        "digraph folded {",
        f'  graph [label="{title}\\n{len(nodes)} nodes / {len(edges)} edges'
        f'{"; " + note if note else ""}", labelloc=t, fontsize=20, fontname="Helvetica"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
    ]
    for n in kept:
        lines.append(f'  "{n.nid}" [label="{n.label}", fillcolor="{COLORS[n.category]}"];')
    for a, b in edges:
        if a in kept_ids and b in kept_ids:
            lines.append(f'  "{a}" -> "{b}";')
    for k, (start, period, reps) in enumerate(folds):
        prev = order[start + period - 1]
        lines.append(
            f'  "fold{k}" [label="... x{reps - 1} more identical blocks ...",'
            ' shape=box, style="dashed,rounded", fontsize=14];'
        )
        lines.append(f'  "{prev.nid}" -> "fold{k}";')
        after = start + period * reps
        if after < len(order):
            lines.append(f'  "fold{k}" -> "{order[after].nid}";')
    lines.append("}")
    return "\n".join(lines), len(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help=".dot from --dump-graph-png or cudaGraphDebugDotPrint")
    ap.add_argument("-o", "--output", type=Path, help="output PNG (default: INPUT with .png)")
    ap.add_argument("--title", help="graph title (default: input file name)")
    ap.add_argument("--no-fold", action="store_true", help="render all nodes unfolded")
    args = ap.parse_args()

    text = args.input.read_text()
    if "topoId" in text:
        nodes, edges = parse_cuda_debug_dot(text)
    elif "raw_symbol=" in text:
        nodes, edges = parse_openinfer_dot(text)
    else:
        sys.exit("error: unrecognized dot dialect (expected cudaGraphDebugDotPrint or openinfer detailed dot)")
    if not nodes:
        sys.exit("error: no graph nodes parsed")

    folds = []
    if not args.no_fold:
        ordered = sorted(nodes.values(), key=lambda x: x.order)
        folds = detect_folds([n.signature for n in ordered])
        if folds:
            for start, period, reps in folds:
                print(f"fold: {period} nodes/block x{reps} (start offset {start})")
        else:
            print("fold: no repeated block detected, rendering unfolded")

    out = args.output or args.input.with_suffix(".png")
    title = args.title or args.input.name
    folded, kept_nodes = emit_folded_dot(nodes, edges, title, folds)

    dpi = pick_dpi(kept_nodes)
    print(f"{kept_nodes} nodes after folding, rendering at {dpi} dpi")
    r = subprocess.run(
        ["dot", "-Tpng:cairo", f"-Gdpi={dpi}", "-o", str(out)],
        input=folded, capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"error: graphviz cairo render failed: {r.stderr.strip()}")
    print(f"wrote {out} ({len(nodes)} nodes, {len(edges)} edges)")


if __name__ == "__main__":
    main()
