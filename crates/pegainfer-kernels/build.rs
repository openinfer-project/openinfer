use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

struct TritonKernelSpec {
    artifact_dir: &'static str,
    kernel_path: &'static str,
    kernel_name: &'static str,
    signature: &'static str,
    grid: &'static str,
    out_name: &'static str,
    num_warps: u32,
    num_stages: u32,
}

fn workspace_root() -> PathBuf {
    crate_root().join("../..")
}

fn crate_root() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"))
}

fn parse_sm_token(raw: &str) -> Option<String> {
    let token = raw.trim().trim_matches('"');
    if token.is_empty() {
        return None;
    }

    let token = token
        .strip_prefix("sm_")
        .or_else(|| token.strip_prefix("compute_"))
        .unwrap_or(token);

    if let Some((major, minor)) = token.split_once('.') {
        if major.chars().all(|c| c.is_ascii_digit()) && minor.chars().all(|c| c.is_ascii_digit()) {
            return Some(format!("{}{}", major, minor));
        }
        return None;
    }

    if token.chars().all(|c| c.is_ascii_digit()) {
        if token.len() == 1 {
            return Some(format!("{}0", token));
        }
        return Some(token.to_string());
    }

    None
}

fn sm_targets_from_nvidia_smi() -> Option<Vec<String>> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut sms = BTreeSet::new();
    for line in stdout.lines() {
        let cap = line.split(',').next().unwrap_or(line).trim();
        if let Some(sm) = parse_sm_token(cap) {
            sms.insert(sm);
        }
    }

    if sms.is_empty() {
        None
    } else {
        Some(sms.into_iter().collect())
    }
}

fn detect_sm_targets() -> Vec<String> {
    if let Ok(env) = std::env::var("PEGAINFER_CUDA_SM").or_else(|_| std::env::var("CUDA_SM")) {
        let mut sms = Vec::new();
        for token in env.split(',') {
            if let Some(sm) = parse_sm_token(token) {
                sms.push(sm);
            } else {
                print!(
                    "cargo:warning=Invalid SM token '{}' in CUDA_SM environment variable, skipping.",
                    token
                );
            }
        }
        if !sms.is_empty() {
            return sms;
        }
        print!(
            "cargo:warning=No valid SM tokens found in CUDA_SM environment variable '{}', falling back to auto-detection.",
            env
        );
    }

    if let Some(sms) = sm_targets_from_nvidia_smi() {
        return sms;
    }

    print!(
        "cargo:warning=Failed to detect GPU SMs via nvidia-smi. Set PEGAINFER_CUDA_SM/CUDA_SM environment variable to override."
    );
    panic!("GPU detection failed");
}

fn nvcc_arch_args(sm_targets: &[String]) -> Vec<String> {
    let mut args = Vec::new();
    for sm in sm_targets {
        args.push("-gencode".to_string());
        args.push(format!("arch=compute_{sm},code=sm_{sm}"));
    }

    if let Some(max_sm) = sm_targets
        .iter()
        .filter_map(|sm| sm.parse::<u32>().ok())
        .max()
    {
        args.push("-gencode".to_string());
        args.push(format!("arch=compute_{max_sm},code=compute_{max_sm}"));
    }

    args
}

fn probe_triton_python(candidate: &str) -> Result<String, String> {
    let output = Command::new(candidate)
        .args(["-c", "import triton"])
        .output()
        .map_err(|err| format!("{candidate}: {err}"))?;

    if output.status.success() {
        Ok(candidate.to_string())
    } else {
        Err(format!(
            "{candidate}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn find_triton_python() -> Result<String, String> {
    if let Ok(candidate) = std::env::var("PEGAINFER_TRITON_PYTHON") {
        let candidate = candidate.trim();
        if candidate.is_empty() {
            return Err(
                "PEGAINFER_TRITON_PYTHON is set but empty. See crates/pegainfer-kernels/tools/triton/README.md.".to_string(),
            );
        }
        return probe_triton_python(candidate).map_err(|message| {
            format!(
                "PEGAINFER_TRITON_PYTHON=`{candidate}` could not import Triton. {message}. See crates/pegainfer-kernels/tools/triton/README.md."
            )
        });
    }

    let local_venv = workspace_root().join(".venv/bin/python");
    let mut diagnostics = Vec::new();
    let mut candidates = Vec::new();
    if local_venv.exists() {
        candidates.push(local_venv.to_string_lossy().to_string());
    }
    candidates.extend(["python3".to_string(), "python".to_string()]);

    for candidate in candidates {
        match probe_triton_python(&candidate) {
            Ok(path) => return Ok(path),
            Err(message) => diagnostics.push(message),
        }
    }

    Err(format!(
        "Could not find a Python interpreter with Triton installed. Set PEGAINFER_TRITON_PYTHON, bootstrap .venv, or ensure `python3 -c 'import triton'` works. Probe results: {}.",
        diagnostics.join(" | ")
    ))
}

fn triton_target(sm_targets: &[String]) -> String {
    let max_sm = sm_targets
        .iter()
        .filter_map(|sm| sm.parse::<u32>().ok())
        .max()
        .expect("expected at least one CUDA SM target for Triton AOT");

    if sm_targets.len() > 1 {
        println!(
            "cargo:warning=Triton AOT currently emits one cubin per kernel spec; using highest detected target sm_{max_sm}. Set PEGAINFER_CUDA_SM to pin one target explicitly."
        );
    }

    format!("cuda:{max_sm}:32")
}

fn generate_triton_artifacts(
    python: &str,
    out_dir: &Path,
    triton_target: &str,
    spec: &TritonKernelSpec,
) -> (String, PathBuf) {
    let root = crate_root();
    let generator_path = root.join("tools/triton/gen_triton_aot.py");
    let artifact_dir = out_dir.join("triton_aot").join(spec.artifact_dir);

    let output = Command::new(python)
        .arg(&generator_path)
        .arg("--kernel-path")
        .arg(root.join(spec.kernel_path))
        .arg("--kernel-name")
        .arg(spec.kernel_name)
        .arg("--signature")
        .arg(spec.signature)
        .arg("--grid")
        .arg(spec.grid)
        .arg("--out-name")
        .arg(spec.out_name)
        .arg("--out-dir")
        .arg(&artifact_dir)
        .arg("--target")
        .arg(triton_target)
        .arg("--num-warps")
        .arg(spec.num_warps.to_string())
        .arg("--num-stages")
        .arg(spec.num_stages.to_string())
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "failed to run Triton AOT generator for {}: {err}",
                spec.kernel_name
            )
        });

    assert!(
        output.status.success(),
        "Triton AOT generator failed for {}. stdout: {} stderr: {}",
        spec.kernel_name,
        String::from_utf8_lossy(&output.stdout).trim(),
        String::from_utf8_lossy(&output.stderr).trim(),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut func_name = None;
    let mut c_path = None;
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix("FUNC_NAME=") {
            func_name = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("C_PATH=") {
            c_path = Some(PathBuf::from(value.trim()));
        }
    }

    let func_name = func_name.expect("Triton generator did not print FUNC_NAME");
    let c_path = c_path.expect("Triton generator did not print C_PATH");
    (func_name, c_path)
}

fn write_wrapper(generated_c: &Path, file_name: &str, wrapper_src: String) -> PathBuf {
    let wrapper_path = generated_c
        .parent()
        .expect("generated Triton source should have a parent directory")
        .join(file_name);
    std::fs::write(&wrapper_path, wrapper_src).expect("failed to write Triton wrapper source");
    wrapper_path
}

fn compile_triton_aot_kernels(cuda_path: &str, out_dir: &Path, sm_targets: &[String]) {
    let python = find_triton_python().unwrap_or_else(|message| panic!("{message}"));
    let triton_target = triton_target(sm_targets);
    let mut generated_sources = Vec::new();
    let root = crate_root();
    let chunkwise_kernel_path = root.join("tools/triton/gated_delta_rule_chunkwise_kernels.py");

    let flash_attn_prefill_hd256_spec = TritonKernelSpec {
        artifact_dir: "flash_attention_prefill_hd256",
        kernel_path: "tools/triton/flash_attention_prefill_hd256_kernel.py",
        kernel_name: "flash_attention_prefill_hd256_kernel",
        signature: "*bf16,*bf16,*bf16,*bf16,i32,i32,i32,i32,*i32,i32,64,64,256",
        grid: "(seq_len + 63) / 64,num_q_heads,1",
        out_name: "triton_flash_attention_prefill_hd256",
        num_warps: 4,
        num_stages: 2,
    };
    let (flash_attn_hd256_func, flash_attn_hd256_c) = generate_triton_artifacts(
        &python,
        out_dir,
        &triton_target,
        &flash_attn_prefill_hd256_spec,
    );
    let flash_attn_hd256_wrapper = write_wrapper(
        &flash_attn_hd256_c,
        "triton_flash_attention_prefill_hd256_wrapper.c",
        format!(
            "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr Q, CUdeviceptr K_cache, CUdeviceptr V_cache, CUdeviceptr Output, int32_t num_q_heads, int32_t num_kv_heads, int32_t gqa_ratio, int32_t seq_len, CUdeviceptr start_pos_ptr, int32_t q_dim);\n\nCUresult flash_attention_prefill_hd256_cuda(const uint16_t* Q, const uint16_t* K_cache, const uint16_t* V_cache, uint16_t* Output, int32_t num_q_heads, int32_t num_kv_heads, int32_t gqa_ratio, int32_t seq_len, const int32_t* start_pos_ptr, int32_t q_dim, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)Q, (CUdeviceptr)K_cache, (CUdeviceptr)V_cache, (CUdeviceptr)Output, num_q_heads, num_kv_heads, gqa_ratio, seq_len, (CUdeviceptr)start_pos_ptr, q_dim);\n}}\n",
            func = flash_attn_hd256_func
        ),
    );
    generated_sources.push(flash_attn_hd256_c);
    generated_sources.push(flash_attn_hd256_wrapper);

    if chunkwise_kernel_path.exists() {
        let gdr_prepare_spec = TritonKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_prepare",
            kernel_path: "tools/triton/gated_delta_rule_chunkwise_kernels.py",
            kernel_name: "gdr_prepare_qkv_gbeta_qwen35_kernel",
            signature: "*bf16,*bf16,*bf16,*bf16,*fp32,*bf16,*bf16,*bf16,*fp32,*fp32,i32,i32,i32,i32,128,128",
            grid: "seq_len,num_value_heads,1",
            out_name: "triton_gated_delta_rule_chunk_prepare",
            num_warps: 4,
            num_stages: 2,
        };
        let (gdr_prepare_func, gdr_prepare_c) =
            generate_triton_artifacts(&python, out_dir, &triton_target, &gdr_prepare_spec);
        let gdr_prepare_wrapper = write_wrapper(
            &gdr_prepare_c,
            "triton_gated_delta_rule_chunk_prepare_wrapper.c",
            format!(
                "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr qkv, CUdeviceptr b_proj, CUdeviceptr a_proj, CUdeviceptr dt_bias, CUdeviceptr a_log, CUdeviceptr q_out, CUdeviceptr k_out, CUdeviceptr v_out, CUdeviceptr g_out, CUdeviceptr beta_out, int32_t num_key_heads, int32_t num_value_heads, int32_t qkv_dim, int32_t seq_len);\n\nCUresult gated_delta_rule_prefill_chunk_prepare_cuda(const uint16_t* qkv, const uint16_t* b_proj, const uint16_t* a_proj, const uint16_t* dt_bias, const float* a_log, uint16_t* q_out, uint16_t* k_out, uint16_t* v_out, float* g_out, float* beta_out, int32_t num_key_heads, int32_t num_value_heads, int32_t qkv_dim, int32_t seq_len, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)qkv, (CUdeviceptr)b_proj, (CUdeviceptr)a_proj, (CUdeviceptr)dt_bias, (CUdeviceptr)a_log, (CUdeviceptr)q_out, (CUdeviceptr)k_out, (CUdeviceptr)v_out, (CUdeviceptr)g_out, (CUdeviceptr)beta_out, num_key_heads, num_value_heads, qkv_dim, seq_len);\n}}\n",
                func = gdr_prepare_func
            ),
        );
        generated_sources.push(gdr_prepare_c);
        generated_sources.push(gdr_prepare_wrapper);

        let gdr_cumsum_spec = TritonKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_cumsum",
            kernel_path: "tools/triton/gated_delta_rule_chunkwise_kernels.py",
            kernel_name: "gdr_chunk_local_cumsum_qwen35_kernel",
            signature: "*fp32,*fp32,i32,i32,64",
            grid: "(seq_len + 63) / 64,num_value_heads,1",
            out_name: "triton_gated_delta_rule_chunk_cumsum",
            num_warps: 1,
            num_stages: 1,
        };
        let (gdr_cumsum_func, gdr_cumsum_c) =
            generate_triton_artifacts(&python, out_dir, &triton_target, &gdr_cumsum_spec);
        let gdr_cumsum_wrapper = write_wrapper(
            &gdr_cumsum_c,
            "triton_gated_delta_rule_chunk_cumsum_wrapper.c",
            format!(
                "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr g_in, CUdeviceptr g_out, int32_t seq_len, int32_t num_value_heads);\n\nCUresult gated_delta_rule_prefill_chunk_cumsum_cuda(const float* g_in, float* g_out, int32_t seq_len, int32_t num_value_heads, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)g_in, (CUdeviceptr)g_out, seq_len, num_value_heads);\n}}\n",
                func = gdr_cumsum_func
            ),
        );
        generated_sources.push(gdr_cumsum_c);
        generated_sources.push(gdr_cumsum_wrapper);

        let gdr_a_spec = TritonKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_a",
            kernel_path: "tools/triton/gated_delta_rule_chunkwise_kernels.py",
            kernel_name: "gdr_chunk_scaled_dot_kkt_qwen35_kernel",
            signature: "*bf16,*fp32,*fp32,*fp32,i32,i32,64,64,128",
            grid: "(seq_len + 63) / 64,num_value_heads,1",
            out_name: "triton_gated_delta_rule_chunk_a",
            num_warps: 4,
            num_stages: 2,
        };
        let (gdr_a_func, gdr_a_c) =
            generate_triton_artifacts(&python, out_dir, &triton_target, &gdr_a_spec);
        let gdr_a_wrapper = write_wrapper(
            &gdr_a_c,
            "triton_gated_delta_rule_chunk_a_wrapper.c",
            format!(
                "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr k, CUdeviceptr g_cumsum, CUdeviceptr beta, CUdeviceptr a_tril, int32_t seq_len, int32_t num_value_heads);\n\nCUresult gated_delta_rule_prefill_chunk_a_cuda(const uint16_t* k, const float* g_cumsum, const float* beta, float* a_tril, int32_t seq_len, int32_t num_value_heads, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)k, (CUdeviceptr)g_cumsum, (CUdeviceptr)beta, (CUdeviceptr)a_tril, seq_len, num_value_heads);\n}}\n",
                func = gdr_a_func
            ),
        );
        generated_sources.push(gdr_a_c);
        generated_sources.push(gdr_a_wrapper);

        let gdr_solve_spec = TritonKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_solve",
            kernel_path: "tools/triton/gated_delta_rule_chunkwise_kernels.py",
            kernel_name: "gdr_solve_tril_64_qwen35_kernel",
            signature: "*fp32,*bf16,i32,i32",
            grid: "(seq_len + 63) / 64,num_value_heads,1",
            out_name: "triton_gated_delta_rule_chunk_solve",
            num_warps: 4,
            num_stages: 2,
        };
        let (gdr_solve_func, gdr_solve_c) =
            generate_triton_artifacts(&python, out_dir, &triton_target, &gdr_solve_spec);
        let gdr_solve_wrapper = write_wrapper(
            &gdr_solve_c,
            "triton_gated_delta_rule_chunk_solve_wrapper.c",
            format!(
                "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr a_tril, CUdeviceptr a_inv, int32_t seq_len, int32_t num_value_heads);\n\nCUresult gated_delta_rule_prefill_chunk_solve_cuda(const float* a_tril, uint16_t* a_inv, int32_t seq_len, int32_t num_value_heads, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)a_tril, (CUdeviceptr)a_inv, seq_len, num_value_heads);\n}}\n",
                func = gdr_solve_func
            ),
        );
        generated_sources.push(gdr_solve_c);
        generated_sources.push(gdr_solve_wrapper);

        let gdr_recompute_spec = TritonKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_recompute",
            kernel_path: "tools/triton/gated_delta_rule_chunkwise_kernels.py",
            kernel_name: "gdr_recompute_w_u_qwen35_kernel",
            signature: "*bf16,*bf16,*fp32,*bf16,*bf16,*bf16,*fp32,i32,i32,128,128,64,64,64",
            grid: "(seq_len + 63) / 64,num_value_heads,1",
            out_name: "triton_gated_delta_rule_chunk_recompute",
            num_warps: 4,
            num_stages: 2,
        };
        let (gdr_recompute_func, gdr_recompute_c) =
            generate_triton_artifacts(&python, out_dir, &triton_target, &gdr_recompute_spec);
        let gdr_recompute_wrapper = write_wrapper(
            &gdr_recompute_c,
            "triton_gated_delta_rule_chunk_recompute_wrapper.c",
            format!(
                "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr k, CUdeviceptr v, CUdeviceptr beta, CUdeviceptr w, CUdeviceptr u, CUdeviceptr a_inv, CUdeviceptr g_cumsum, int32_t seq_len, int32_t num_value_heads);\n\nCUresult gated_delta_rule_prefill_chunk_recompute_cuda(const uint16_t* k, const uint16_t* v, const float* beta, uint16_t* w, uint16_t* u, const uint16_t* a_inv, const float* g_cumsum, int32_t seq_len, int32_t num_value_heads, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)k, (CUdeviceptr)v, (CUdeviceptr)beta, (CUdeviceptr)w, (CUdeviceptr)u, (CUdeviceptr)a_inv, (CUdeviceptr)g_cumsum, seq_len, num_value_heads);\n}}\n",
                func = gdr_recompute_func
            ),
        );
        generated_sources.push(gdr_recompute_c);
        generated_sources.push(gdr_recompute_wrapper);

        let gdr_chunk_state_spec = TritonKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_state",
            kernel_path: "tools/triton/gated_delta_rule_chunkwise_kernels.py",
            kernel_name: "gdr_chunk_state_qwen35_kernel",
            signature: "*bf16,*bf16,*bf16,*fp32,*fp32,*fp32,*bf16,*fp32,i32,i32,32,64,128,128,64",
            grid: "4,num_value_heads,1",
            out_name: "triton_gated_delta_rule_chunk_state",
            num_warps: 4,
            num_stages: 2,
        };
        let (gdr_chunk_state_func, gdr_chunk_state_c) =
            generate_triton_artifacts(&python, out_dir, &triton_target, &gdr_chunk_state_spec);
        let gdr_chunk_state_wrapper = write_wrapper(
            &gdr_chunk_state_c,
            "triton_gated_delta_rule_chunk_state_wrapper.c",
            format!(
                "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr k, CUdeviceptr w, CUdeviceptr u, CUdeviceptr g_cumsum, CUdeviceptr initial_state, CUdeviceptr chunk_state, CUdeviceptr v_new, CUdeviceptr final_state, int32_t seq_len, int32_t num_value_heads);\n\nCUresult gated_delta_rule_prefill_chunk_state_cuda(const uint16_t* k, const uint16_t* w, const uint16_t* u, const float* g_cumsum, const float* initial_state, float* chunk_state, uint16_t* v_new, float* final_state, int32_t seq_len, int32_t num_value_heads, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)k, (CUdeviceptr)w, (CUdeviceptr)u, (CUdeviceptr)g_cumsum, (CUdeviceptr)initial_state, (CUdeviceptr)chunk_state, (CUdeviceptr)v_new, (CUdeviceptr)final_state, seq_len, num_value_heads);\n}}\n",
                func = gdr_chunk_state_func
            ),
        );
        generated_sources.push(gdr_chunk_state_c);
        generated_sources.push(gdr_chunk_state_wrapper);

        let gdr_chunk_o_spec = TritonKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_o",
            kernel_path: "tools/triton/gated_delta_rule_chunkwise_kernels.py",
            kernel_name: "gdr_chunk_o_qwen35_kernel",
            signature: "*bf16,*bf16,*bf16,*fp32,*fp32,*bf16,i32,i32,fp32,64,32,64,128,128",
            grid: "4,(seq_len + 63) / 64,num_value_heads",
            out_name: "triton_gated_delta_rule_chunk_o",
            num_warps: 4,
            num_stages: 2,
        };
        let (gdr_chunk_o_func, gdr_chunk_o_c) =
            generate_triton_artifacts(&python, out_dir, &triton_target, &gdr_chunk_o_spec);
        let gdr_chunk_o_wrapper = write_wrapper(
            &gdr_chunk_o_c,
            "triton_gated_delta_rule_chunk_o_wrapper.c",
            format!(
                "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr q, CUdeviceptr k, CUdeviceptr v_new, CUdeviceptr chunk_state, CUdeviceptr g_cumsum, CUdeviceptr output, int32_t seq_len, int32_t num_value_heads, float scale);\n\nCUresult gated_delta_rule_prefill_chunk_o_cuda(const uint16_t* q, const uint16_t* k, const uint16_t* v_new, const float* chunk_state, const float* g_cumsum, uint16_t* output, int32_t seq_len, int32_t num_value_heads, float scale, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)q, (CUdeviceptr)k, (CUdeviceptr)v_new, (CUdeviceptr)chunk_state, (CUdeviceptr)g_cumsum, (CUdeviceptr)output, seq_len, num_value_heads, scale);\n}}\n",
                func = gdr_chunk_o_func
            ),
        );
        generated_sources.push(gdr_chunk_o_c);
        generated_sources.push(gdr_chunk_o_wrapper);
    } else {
        println!(
            "cargo:warning=Skipping chunk-wise GDR Triton AOT scaffolding because {} is not present yet.",
            chunkwise_kernel_path.display()
        );
    }

    let mut build = cc::Build::new();
    build
        .cuda(false)
        .include(format!("{}/include", cuda_path))
        .flag("-std=c11")
        .warnings(false);
    for source in &generated_sources {
        build.file(source);
    }
    build.compile("triton_kernels_aot");

    println!("cargo:rustc-link-lib=cuda");
    println!(
        "cargo:warning=Using Triton AOT for Qwen3.5 HD256 FlashAttention prefill and GDR chunkwise; basic ops (add, silu_mul, embedding) use native CUDA"
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("tools/triton/flash_attention_prefill_hd256_kernel.py")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("tools/triton/gated_delta_rule_chunkwise_kernels.py")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("tools/triton/gen_triton_aot.py").display()
    );
    println!("cargo:rerun-if-env-changed=PEGAINFER_TRITON_PYTHON");
}

fn main() {
    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let sm_targets = detect_sm_targets();
    let arch_args = nvcc_arch_args(&sm_targets);
    println!(
        "cargo:warning=Compiling CUDA kernels for targets: {}",
        sm_targets
            .iter()
            .map(|sm| format!("sm_{sm}"))
            .collect::<Vec<_>>()
            .join(",")
    );

    let replaced_cuda_files =
        BTreeSet::from(["activation.cu", "embedding.cu", "fused_attention.cu"]);

    let root = crate_root();
    let csrc_dir = root.join("csrc");
    let cu_files: Vec<_> = std::fs::read_dir(csrc_dir)
        .expect("Failed to read csrc/ directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let file_name = path.file_name()?.to_str()?;
            if path.extension().and_then(|e| e.to_str()) == Some("cu")
                && !replaced_cuda_files.contains(file_name)
            {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    println!(
        "cargo:warning=Legacy CUDA translation units retired from the runtime build: {}",
        replaced_cuda_files
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut obj_files = Vec::new();
    for cu_file in &cu_files {
        let stem = cu_file.file_stem().unwrap().to_str().unwrap();
        let obj_file = out_dir.join(format!("{}_cuda.o", stem));

        let mut nvcc_args = vec![
            "-c".to_string(),
            cu_file.to_string_lossy().to_string(),
            "-o".to_string(),
            obj_file.to_string_lossy().to_string(),
            "-O3".to_string(),
        ];
        nvcc_args.extend(arch_args.clone());
        nvcc_args.extend(["--compiler-options".to_string(), "-fPIC".to_string()]);

        // Files that include FlashInfer headers (C++17, header-only)
        if stem == "paged_attention"
            || stem == "flashinfer_norm"
            || stem == "flashinfer_sampling"
            || stem == "flashinfer_top1"
        {
            nvcc_args.extend([
                "--std=c++17".to_string(),
                "-I".to_string(),
                root.join("third_party/flashinfer/include")
                    .to_string_lossy()
                    .to_string(),
            ]);
        }

        let status = Command::new(&nvcc)
            .args(&nvcc_args)
            .status()
            .unwrap_or_else(|_| panic!("Failed to run nvcc for {}", cu_file.display()));

        assert!(
            status.success(),
            "nvcc compilation failed for {}",
            cu_file.display()
        );

        obj_files.push(obj_file);
    }

    let cuda_lib = out_dir.join("libkernels_cuda.a");
    let mut ar_args = vec!["rcs".to_string(), cuda_lib.to_string_lossy().to_string()];
    ar_args.extend(
        obj_files
            .into_iter()
            .map(|path| path.to_string_lossy().to_string()),
    );

    let status = Command::new("ar")
        .args(&ar_args)
        .status()
        .expect("Failed to run ar");

    assert!(status.success(), "ar failed");

    compile_triton_aot_kernels(&cuda_path, &out_dir, &sm_targets);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    println!("cargo:rustc-link-lib=static=kernels_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed={}", root.join("csrc").display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=PEGAINFER_CUDA_SM");
    println!("cargo:rerun-if-env-changed=CUDA_SM");
}
