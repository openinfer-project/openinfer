//! GLM5.2 single-layer MLA decode microbench (bs=1, synthetic weights).
//!
//! Usage:
//!   cargo run --release -p openinfer-glm52 --features glm52 \
//!     --bin glm52_kernel_bench -- [--contexts 512,2048] [--iters 64]

use anyhow::{Result, bail};
use openinfer_glm52::kernel_bench::Glm52MlaDecodeBench;

struct Args {
    contexts: Vec<usize>,
    iters: u64,
    sm_parts: Option<usize>,
}

fn parse_args(mut argv: impl Iterator<Item = String>) -> Result<Args> {
    let mut args = Args {
        contexts: vec![512, 2048],
        iters: 64,
        sm_parts: None,
    };
    while let Some(flag) = argv.next() {
        let mut value = || {
            argv.next()
                .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--contexts" => {
                args.contexts = value()?
                    .split(',')
                    .map(|v| v.trim().parse::<usize>())
                    .collect::<Result<_, _>>()?;
            }
            "--iters" => args.iters = value()?.parse()?,
            "--sm-parts" => args.sm_parts = Some(value()?.parse()?),
            other => {
                bail!("unknown flag `{other}` (supported: --contexts, --iters, --sm-parts)")
            }
        }
    }
    if args.contexts.is_empty() || args.iters == 0 {
        bail!("--contexts must be non-empty and --iters positive");
    }
    Ok(args)
}

fn main() -> Result<()> {
    let args = parse_args(std::env::args().skip(1))?;
    for &context in &args.contexts {
        let mut bench = Glm52MlaDecodeBench::new(context)?;
        if let Some(parts) = args.sm_parts {
            bench.set_num_sm_parts(parts)?;
            println!("(num_sm_parts overridden to {parts})");
        }
        bench.verify_scratch_parity()?;
        let (gpu, wall) = bench.measure_forward(args.iters)?;
        let per = |d: std::time::Duration| d.as_secs_f64() * 1.0e6 / args.iters as f64;
        println!("== context {context} (iters {}) ==", args.iters);
        println!(
            "layer forward      gpu {:>9.1}us  wall {:>9.1}us  host-side gap {:>9.1}us",
            per(gpu),
            per(wall),
            per(wall) - per(gpu)
        );
        let (gpu_s, wall_s) = bench.measure_forward_scratch(args.iters)?;
        println!(
            "layer fwd scratch  gpu {:>9.1}us  wall {:>9.1}us  host-side gap {:>9.1}us",
            per(gpu_s),
            per(wall_s),
            per(wall_s) - per(gpu_s)
        );
        println!(
            "-> alloc bill (as-is wall - scratch wall): {:>9.1}us/layer",
            per(wall) - per(wall_s)
        );
        let (gpu_g, wall_g) = bench.measure_forward_graph(args.iters)?;
        println!(
            "layer fwd graph    gpu {:>9.1}us  wall {:>9.1}us  host-side gap {:>9.1}us",
            per(gpu_g),
            per(wall_g),
            per(wall_g) - per(gpu_g)
        );
        println!(
            "-> total vs as-is (as-is wall - graph wall): {:>9.1}us/layer",
            per(wall) - per(wall_g)
        );
        for proj in ["q_a", "q_b", "kv_a", "o_proj"] {
            let d = bench.measure_projection(proj, args.iters)?;
            println!(
                "fp8_linear {proj:<8} wall {:>9.1}us (alloc chain included)",
                per(d)
            );
        }
        let d = bench.measure_assembly_family(args.iters)?;
        println!(
            "assembly family    wall {:>9.1}us (assemble+quant+pack, buffers reused)",
            per(d)
        );
        let d = bench.measure_flashmla(args.iters)?;
        println!(
            "flashmla sparse    wall {:>9.1}us (metadata+decode, buffers reused)",
            per(d)
        );
        // Sweep the split count: the device default over-splits a bs=1 sparse
        // decode, so a smaller num_sm_parts can shrink the combine round-trip
        // faster than it costs partial parallelism.
        let default_parts = bench.default_num_sm_parts();
        print!("flashmla sweep     ");
        for parts in [1usize, 8, 16, 32, 64, 96]
            .iter()
            .copied()
            .filter(|&p| p != default_parts)
            .chain(std::iter::once(default_parts))
        {
            if let Some(t) = bench.measure_flashmla_at(parts, args.iters)? {
                let tag = if parts == default_parts { "*" } else { "" };
                print!("p{parts}{tag}={:.1}us ", per(t));
            }
        }
        println!("(* = device default)");
        let diff16 = bench.flashmla_parts_max_diff(16)?;
        println!(
            "flashmla p16 vs default: max abs latent diff {diff16:.3e} (parallelization-only knob → safe if ~fp noise)"
        );
        let projected_token = per(wall) * 75.0 / 1000.0;
        println!(
            "-> projected 75 MoE-layer attention share: {projected_token:.2} ms/token (as-is)\n"
        );
    }
    Ok(())
}
