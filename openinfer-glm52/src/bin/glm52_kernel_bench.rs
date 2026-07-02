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
}

fn parse_args(mut argv: impl Iterator<Item = String>) -> Result<Args> {
    let mut args = Args {
        contexts: vec![512, 2048],
        iters: 64,
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
            other => bail!("unknown flag `{other}` (supported: --contexts, --iters)"),
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
        let (gpu, wall) = bench.measure_forward(args.iters)?;
        let per = |d: std::time::Duration| d.as_secs_f64() * 1.0e6 / args.iters as f64;
        println!("== context {context} (iters {}) ==", args.iters);
        println!(
            "layer forward      gpu {:>9.1}us  wall {:>9.1}us  host-side gap {:>9.1}us",
            per(gpu),
            per(wall),
            per(wall) - per(gpu)
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
        let projected_token = per(wall) * 75.0 / 1000.0;
        println!(
            "-> projected 75 MoE-layer attention share: {projected_token:.2} ms/token (as-is)\n"
        );
    }
    Ok(())
}
