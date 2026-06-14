use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

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

struct NvccTask {
    cu_file: PathBuf,
    obj_file: PathBuf,
    args: Vec<String>,
}

struct TileLangArtifacts {
    cu_files: Vec<PathBuf>,
    template_include: PathBuf,
    cutlass_include: PathBuf,
}

struct CuTeDslArtifacts {
    obj_files: Vec<PathBuf>,
    wrapper_files: Vec<PathBuf>,
    include_dir: PathBuf,
    runtime_lib_dirs: Vec<PathBuf>,
}

struct FlashInferIncludes {
    include: PathBuf,
    csrc: PathBuf,
    cutlass: PathBuf,
    cutlass_util: PathBuf,
    spdlog: PathBuf,
    /// Vendored CCCL (cub, libcudacxx, thrust). Must be passed as `-I` while the
    /// CTK include dir is `-isystem`, so these override the toolkit's older CCCL
    /// copy — FlashInfer v0.6+ uses APIs (e.g. `cuda::fast_mod_div`) that older
    /// CTK CCCL lacks. Mirrors upstream flashinfer/jit/cpp_ext.py ordering.
    cccl: Vec<PathBuf>,
}

fn workspace_root() -> PathBuf {
    crate_root().join("..")
}

fn crate_root() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"))
}

fn build_timing_enabled() -> bool {
    std::env::var("OPENINFER_BUILD_TIMING").is_ok_and(|value| {
        let value = value.trim().to_ascii_lowercase();
        !(value.is_empty() || value == "0" || value == "false" || value == "off")
    })
}

fn time_phase<T>(label: impl AsRef<str>, f: impl FnOnce() -> T) -> T {
    if !build_timing_enabled() {
        return f();
    }

    let started = Instant::now();
    let result = f();
    println!(
        "cargo:warning=build-timing {} {:.3}s",
        label.as_ref(),
        started.elapsed().as_secs_f64()
    );
    result
}

fn parse_job_count_env(name: &str) -> Option<usize> {
    let value = std::env::var(name).ok()?;
    match value.trim().parse::<usize>() {
        Ok(jobs) if jobs > 0 => Some(jobs),
        _ => {
            println!("cargo:warning=Ignoring invalid {name}={value}; expected a positive integer.");
            None
        }
    }
}

fn nvcc_job_count() -> usize {
    if let Some(jobs) = parse_job_count_env("OPENINFER_NVCC_JOBS") {
        return jobs;
    }

    parse_job_count_env("NUM_JOBS")
        .or_else(|| thread::available_parallelism().ok().map(usize::from))
        .unwrap_or(1)
        .max(1)
}

fn nvcc_task_priority(cu_file: &Path) -> usize {
    match cu_file.file_stem().and_then(|stem| stem.to_str()) {
        Some("paged_attention") => 0,
        Some("flashinfer_sampling") => 1,
        Some("flashinfer_top1") => 2,
        Some("flashinfer_norm") => 3,
        _ => 10,
    }
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
        let digit_count = minor.chars().take_while(char::is_ascii_digit).count();
        let (minor_digits, suffix) = minor.split_at(digit_count);
        if major.chars().all(|c| c.is_ascii_digit())
            && !minor_digits.is_empty()
            && suffix.chars().all(|c| c.is_ascii_alphabetic())
        {
            return Some(format!("{major}{minor_digits}{suffix}"));
        }
        return None;
    }

    let digit_count = token.chars().take_while(char::is_ascii_digit).count();
    let (digits, suffix) = token.split_at(digit_count);
    if !digits.is_empty() && suffix.chars().all(|c| c.is_ascii_alphabetic()) {
        if digits.len() == 1 {
            return Some(format!("{digits}0{suffix}"));
        }
        return Some(format!("{digits}{suffix}"));
    }

    None
}

fn nvcc_supported_arches(nvcc: &str) -> Option<BTreeSet<String>> {
    let output = Command::new(nvcc).arg("--list-gpu-arch").output().ok()?;
    if !output.status.success() {
        return None;
    }

    Some(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("compute_"))
            .map(str::to_string)
            .collect(),
    )
}

fn normalize_nvcc_sm(sm: &str, supported_arches: Option<&BTreeSet<String>>) -> String {
    let preferred = match sm {
        "120" | "120f" => Some("120f"),
        "121" | "121a" => Some("121a"),
        _ => None,
    };
    if let Some(preferred) = preferred {
        if supported_arches.is_some_and(|arches| arches.contains(&format!("compute_{preferred}"))) {
            return preferred.to_string();
        }
        let raw = sm_numeric_prefix(sm).map_or_else(|| sm.to_string(), |sm| sm.to_string());
        println!(
            "cargo:warning=nvcc does not list compute_{preferred}; compiling CUDA kernels for raw sm_{raw}"
        );
        return raw;
    }
    sm.to_string()
}

fn normalize_nvcc_sms(sm_targets: &[String], nvcc: &str) -> Vec<String> {
    let supported_arches = nvcc_supported_arches(nvcc);
    sm_targets
        .iter()
        .map(|sm| normalize_nvcc_sm(sm, supported_arches.as_ref()))
        .collect()
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
    if let Ok(env) = std::env::var("OPENINFER_CUDA_SM").or_else(|_| std::env::var("CUDA_SM")) {
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
        "cargo:warning=Failed to detect GPU SMs via nvidia-smi. Set OPENINFER_CUDA_SM/CUDA_SM environment variable to override."
    );
    panic!("GPU detection failed");
}

fn nvcc_arch_args(normalized_sms: &[String]) -> Vec<String> {
    let mut args = Vec::new();
    for sm in normalized_sms {
        args.push("-gencode".to_string());
        args.push(format!("arch=compute_{sm},code=sm_{sm}"));
    }

    if let Some(max_sm) = normalized_sms
        .iter()
        .max_by_key(|sm| sm_numeric_prefix(sm).unwrap_or(0))
    {
        args.push("-gencode".to_string());
        args.push(format!("arch=compute_{max_sm},code=compute_{max_sm}"));
    }

    args
}

fn collect_files_recursively(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("Failed to read {}: {err}", dir.display()))
        .map(|entry| {
            entry.unwrap_or_else(|err| panic!("Failed to read entry in {}: {err}", dir.display()))
        })
        .collect();
    entries.sort_by_key(std::fs::DirEntry::path);

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursively(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn is_deepseek_v4_source(csrc_dir: &Path, path: &Path) -> bool {
    match path.strip_prefix(csrc_dir) {
        Ok(relative) => relative
            .components()
            .any(|part| part.as_os_str() == "deepseek_v4"),
        Err(_) => false,
    }
}

fn is_kimi_k2_source(csrc_dir: &Path, path: &Path) -> bool {
    match path.strip_prefix(csrc_dir) {
        Ok(relative) => relative
            .components()
            .any(|part| part.as_os_str() == "kimi_k2"),
        Err(_) => false,
    }
}

/// DeepEP elastic shim (csrc/deepep/): Kimi-K2's EP all-to-all backend.
/// Compiled only with the `kimi-k2` feature; needs NCCL >= 2.30.4 headers/lib.
fn is_deepep_source(csrc_dir: &Path, path: &Path) -> bool {
    match path.strip_prefix(csrc_dir) {
        Ok(relative) => relative
            .components()
            .any(|part| part.as_os_str() == "deepep"),
        Err(_) => false,
    }
}

/// NCCL >= 2.30.4 root (include/nccl.h + lib/libnccl.so.2) for the DeepEP
/// shim's device API (ncclDevComm / windows / GIN). cudarc dlopens whatever
/// libnccl.so.2 it finds at runtime, so build and runtime must point at the
/// same install: set OPENINFER_NCCL_ROOT and put `$OPENINFER_NCCL_ROOT/lib`
/// on LD_LIBRARY_PATH. The nvidia-nccl-cu13 wheel layout works directly:
///   pip download 'nvidia-nccl-cu13>=2.30.4' --no-deps -d /tmp/nccl \
///     && unzip /tmp/nccl/*.whl 'nvidia/nccl/*' -d /tmp/nccl \
///     && export OPENINFER_NCCL_ROOT=/tmp/nccl/nvidia/nccl
fn deepep_nccl_root() -> PathBuf {
    let Ok(root) = std::env::var("OPENINFER_NCCL_ROOT").map(PathBuf::from) else {
        panic!(
            "The kimi-k2 feature builds the DeepEP shim, which needs NCCL >= 2.30.4. \
             Set OPENINFER_NCCL_ROOT to an install with include/nccl.h and lib/libnccl.so.2 \
             (e.g. the unpacked nvidia-nccl-cu13 wheel)."
        )
    };

    let header = root.join("include/nccl.h");
    let contents = fs::read_to_string(&header).unwrap_or_else(|err| {
        panic!(
            "OPENINFER_NCCL_ROOT: cannot read {}: {err}",
            header.display()
        )
    });
    let version_component = |name: &str| -> u32 {
        contents
            .lines()
            .find_map(|line| line.strip_prefix(&format!("#define {name} ")))
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or_else(|| panic!("{} does not define {name}", header.display()))
    };
    let version = version_component("NCCL_MAJOR") * 10000
        + version_component("NCCL_MINOR") * 100
        + version_component("NCCL_PATCH");
    assert!(
        version >= 23004,
        "OPENINFER_NCCL_ROOT points at NCCL {version} (< 2.30.4); the DeepEP shim needs the \
         NCCL device API"
    );

    let lib = root.join("lib/libnccl.so.2");
    assert!(
        lib.is_file(),
        "OPENINFER_NCCL_ROOT: {} not found",
        lib.display()
    );
    root
}

/// The wheel ships only libnccl.so.2; give the linker the unversioned name it
/// wants via a symlink in OUT_DIR (already on the link search path). The
/// resulting DT_NEEDED is the soname (libnccl.so.2), resolved at runtime from
/// LD_LIBRARY_PATH.
fn link_deepep_nccl(nccl_root: &Path, out_dir: &Path) {
    let link_name = out_dir.join("libnccl.so");
    let target = nccl_root.join("lib/libnccl.so.2");
    match fs::read_link(&link_name) {
        Ok(existing) if existing == target => {}
        _ => {
            let _ = fs::remove_file(&link_name);
            std::os::unix::fs::symlink(&target, &link_name)
                .unwrap_or_else(|err| panic!("Failed to symlink {}: {err}", link_name.display()));
        }
    }
    println!("cargo:rustc-link-lib=dylib=nccl");
}

fn cuda_object_name(csrc_dir: &Path, cu_file: &Path) -> String {
    let Some(relative) = cu_file.strip_prefix(csrc_dir).ok() else {
        return format!("{}_cuda.o", cu_file.file_stem().unwrap().to_string_lossy());
    };

    let mut parts: Vec<String> = relative
        .components()
        .filter_map(|component| component.as_os_str().to_str().map(ToOwned::to_owned))
        .collect();
    if let Some(last) = parts.last_mut() {
        if let Some(stem) = Path::new(last).file_stem().and_then(|stem| stem.to_str()) {
            *last = stem.to_string();
        }
    }

    format!("{}_cuda.o", parts.join("_"))
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
    if let Ok(candidate) = std::env::var("OPENINFER_TRITON_PYTHON") {
        let candidate = candidate.trim();
        if candidate.is_empty() {
            return Err(
                "OPENINFER_TRITON_PYTHON is set but empty. See openinfer-kernels/tools/triton/README.md.".to_string(),
            );
        }
        return probe_triton_python(candidate).map_err(|message| {
            format!(
                "OPENINFER_TRITON_PYTHON=`{candidate}` could not import Triton. {message}. See openinfer-kernels/tools/triton/README.md."
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
        "Could not find a Python interpreter with Triton installed. Set OPENINFER_TRITON_PYTHON, bootstrap .venv, or ensure `python3 -c 'import triton'` works. Probe results: {}.",
        diagnostics.join(" | ")
    ))
}

fn probe_tilelang_python(candidate: &str) -> Result<String, String> {
    let output = Command::new(candidate)
        .args(["-c", "import tilelang"])
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

fn find_tilelang_python() -> Result<String, String> {
    if let Ok(candidate) = std::env::var("OPENINFER_TILELANG_PYTHON") {
        let candidate = candidate.trim();
        if candidate.is_empty() {
            return Err("OPENINFER_TILELANG_PYTHON is set but empty.".to_string());
        }
        return probe_tilelang_python(candidate).map_err(|message| {
            format!("OPENINFER_TILELANG_PYTHON=`{candidate}` could not import TileLang: {message}")
        });
    }

    let local_venv = workspace_root().join("../.venv/bin/python");
    let workspace_venv = workspace_root().join(".venv/bin/python");
    let mut diagnostics = Vec::new();
    let mut candidates = Vec::new();
    if local_venv.exists() {
        candidates.push(local_venv.to_string_lossy().to_string());
    }
    if workspace_venv.exists() {
        candidates.push(workspace_venv.to_string_lossy().to_string());
    }
    candidates.extend(["python3".to_string(), "python".to_string()]);

    for candidate in candidates {
        match probe_tilelang_python(&candidate) {
            Ok(path) => return Ok(path),
            Err(message) => diagnostics.push(message),
        }
    }

    Err(format!(
        "Could not find a Python interpreter with TileLang installed. Probe results: {}.",
        diagnostics.join(" | ")
    ))
}

fn probe_cutedsl_python(candidate: &str) -> Result<String, String> {
    let output = Command::new(candidate)
        .args(["-c", "import cutlass, cutlass.cute"])
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

fn find_cutedsl_python() -> Result<String, String> {
    if let Ok(candidate) = std::env::var("OPENINFER_CUTEDSL_PYTHON") {
        let candidate = candidate.trim();
        if candidate.is_empty() {
            return Err("OPENINFER_CUTEDSL_PYTHON is set but empty.".to_string());
        }
        return probe_cutedsl_python(candidate).map_err(|message| {
            format!("OPENINFER_CUTEDSL_PYTHON=`{candidate}` could not import CuTe DSL: {message}")
        });
    }

    let workspace_venv = workspace_root().join(".venv/bin/python");
    let parent_venv = workspace_root().join("../.venv/bin/python");
    let mut diagnostics = Vec::new();
    let mut candidates = Vec::new();
    if workspace_venv.exists() {
        candidates.push(workspace_venv.to_string_lossy().to_string());
    }
    if parent_venv.exists() {
        candidates.push(parent_venv.to_string_lossy().to_string());
    }
    candidates.extend(["python3".to_string(), "python".to_string()]);

    for candidate in candidates {
        match probe_cutedsl_python(&candidate) {
            Ok(path) => return Ok(path),
            Err(message) => diagnostics.push(message),
        }
    }

    Err(format!(
        "Could not find a Python interpreter with CuTe DSL installed. Set OPENINFER_CUTEDSL_PYTHON. Probe results: {}.",
        diagnostics.join(" | ")
    ))
}

fn generate_deepseek_tilelang_artifacts(out_dir: &Path) -> TileLangArtifacts {
    let python = find_tilelang_python().unwrap_or_else(|message| {
        panic!("DeepSeek V4 TileLang kernels require TileLang at build time: {message}")
    });

    let root = crate_root();
    let generator_path = root.join("tools/tilelang/deepseek_v4/generate.py");
    assert!(
        generator_path.exists(),
        "DeepSeek V4 TileLang generator is missing: {}",
        generator_path.display()
    );

    let artifact_dir = out_dir.join("tilelang").join("deepseek_v4");
    let output = time_phase("tilelang-gen deepseek_v4", || {
        Command::new(&python)
            .arg(&generator_path)
            .arg("--out-dir")
            .arg(&artifact_dir)
            .output()
            .unwrap_or_else(|err| panic!("failed to run DeepSeek TileLang generator: {err}"))
    });
    assert!(
        output.status.success(),
        "DeepSeek TileLang generator failed. stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout).trim(),
        String::from_utf8_lossy(&output.stderr).trim(),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut cu_path = None;
    let mut template_include = None;
    let mut cutlass_include = None;
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix("CU_PATH=") {
            cu_path = Some(PathBuf::from(value.trim()));
        } else if let Some(value) = line.strip_prefix("TILELANG_TEMPLATE_PATH=") {
            template_include = Some(PathBuf::from(value.trim()));
        } else if let Some(value) = line.strip_prefix("CUTLASS_INCLUDE_DIR=") {
            cutlass_include = Some(PathBuf::from(value.trim()));
        }
    }

    let cu_path = cu_path.expect("DeepSeek TileLang generator did not print CU_PATH");
    let template_include =
        template_include.expect("DeepSeek TileLang generator did not print TILELANG_TEMPLATE_PATH");
    let cutlass_include =
        cutlass_include.expect("DeepSeek TileLang generator did not print CUTLASS_INCLUDE_DIR");

    println!(
        "cargo:warning=Using DeepSeek V4 TileLang generated CUDA: {}",
        cu_path.display()
    );
    println!("cargo:rerun-if-changed={}", generator_path.display());
    println!("cargo:rerun-if-env-changed=OPENINFER_TILELANG_PYTHON");

    TileLangArtifacts {
        cu_files: vec![cu_path],
        template_include,
        cutlass_include,
    }
}

fn generate_deepseek_cutedsl_artifacts(out_dir: &Path) -> CuTeDslArtifacts {
    let python = find_cutedsl_python().unwrap_or_else(|message| {
        panic!("DeepSeek V4 CuTe DSL kernels require CuTe DSL at build time: {message}")
    });

    let root = crate_root();
    let repo_root = workspace_root();
    let generator_path = root.join("tools/cutedsl/deepseek_v4/generate.py");
    assert!(
        generator_path.exists(),
        "DeepSeek V4 CuTe DSL generator is missing: {}",
        generator_path.display()
    );

    let artifact_dir = out_dir.join("cutedsl").join("deepseek_v4");
    let mut command = Command::new(&python);
    command
        .arg(&generator_path)
        .arg("--out-dir")
        .arg(&artifact_dir)
        .arg("--repo-root")
        .arg(&repo_root);
    if let Ok(cutlass_root) = std::env::var("OPENINFER_CUTEDSL_CUTLASS_ROOT") {
        command.arg("--cutlass-root").arg(cutlass_root);
    }

    let output = time_phase("cutedsl-gen deepseek_v4", || {
        command
            .output()
            .unwrap_or_else(|err| panic!("failed to run DeepSeek CuTe DSL generator: {err}"))
    });
    assert!(
        output.status.success(),
        "DeepSeek CuTe DSL generator failed. stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout).trim(),
        String::from_utf8_lossy(&output.stderr).trim(),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut obj_files = Vec::new();
    let mut wrapper_files = Vec::new();
    let mut include_dir = None;
    let mut runtime_lib_dirs = Vec::new();
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix("OBJ_PATH=") {
            obj_files.push(PathBuf::from(value.trim()));
        } else if let Some(value) = line.strip_prefix("WRAPPER_PATH=") {
            wrapper_files.push(PathBuf::from(value.trim()));
        } else if let Some(value) = line.strip_prefix("HEADER_DIR=") {
            include_dir = Some(PathBuf::from(value.trim()));
        } else if let Some(value) = line.strip_prefix("RUNTIME_LIB_DIR=") {
            runtime_lib_dirs.push(PathBuf::from(value.trim()));
        }
    }

    let include_dir = include_dir.expect("DeepSeek CuTe DSL generator did not print HEADER_DIR");
    assert!(
        !obj_files.is_empty() && !wrapper_files.is_empty(),
        "DeepSeek CuTe DSL generator did not emit OBJ_PATH/WRAPPER_PATH"
    );
    for path in obj_files.iter().chain(wrapper_files.iter()) {
        assert!(
            path.exists(),
            "generated CuTe DSL artifact missing: {}",
            path.display()
        );
    }

    println!(
        "cargo:warning=Using DeepSeek V4 CuTe DSL AOT artifacts from {}",
        artifact_dir.display()
    );
    println!("cargo:rerun-if-changed={}", generator_path.display());
    println!("cargo:rerun-if-env-changed=OPENINFER_CUTEDSL_PYTHON");
    println!("cargo:rerun-if-env-changed=OPENINFER_CUTEDSL_CUTLASS_ROOT");

    CuTeDslArtifacts {
        obj_files,
        wrapper_files,
        include_dir,
        runtime_lib_dirs,
    }
}

fn first_existing_dir(candidates: &[PathBuf], fallback: PathBuf) -> PathBuf {
    candidates
        .iter()
        .find(|path| path.exists())
        .cloned()
        .unwrap_or(fallback)
}

fn flashinfer_includes() -> FlashInferIncludes {
    let crate_root = crate_root();
    let root = workspace_root();

    if let Ok(path) = std::env::var("OPENINFER_FLASHINFER_INCLUDE") {
        let path = PathBuf::from(path);
        if path.join("flashinfer/sampling.cuh").exists() {
            return flashinfer_includes_from_include(path);
        }
        println!(
            "cargo:warning=OPENINFER_FLASHINFER_INCLUDE={} does not contain flashinfer/sampling.cuh; falling back.",
            path.display()
        );
    }

    let candidates = [
        crate_root.join("third_party/flashinfer/include"),
        root.join("third_party/flashinfer/include"),
        root.join(".venv/lib/python3.13/site-packages/flashinfer/data/include"),
        root.join(".venv/lib/python3.12/site-packages/flashinfer/data/include"),
        root.join(".venv/lib/python3.11/site-packages/flashinfer/data/include"),
        root.join(".venv/lib/python3.10/site-packages/flashinfer/data/include"),
        root.join("../.venv/lib/python3.13/site-packages/flashinfer/data/include"),
        root.join("../.venv/lib/python3.12/site-packages/flashinfer/data/include"),
        root.join("../.venv/lib/python3.11/site-packages/flashinfer/data/include"),
        root.join("../.venv/lib/python3.10/site-packages/flashinfer/data/include"),
    ];

    for candidate in candidates {
        if candidate.join("flashinfer/sampling.cuh").exists() {
            return flashinfer_includes_from_include(candidate);
        }
    }

    flashinfer_includes_from_include(crate_root.join("third_party/flashinfer/include"))
}

fn flashinfer_includes_from_include(include: PathBuf) -> FlashInferIncludes {
    let flashinfer_root = include
        .parent()
        .expect("FlashInfer include dir must have a parent")
        .to_path_buf();

    let csrc = flashinfer_root.join("csrc");
    let cutlass = first_existing_dir(
        &[
            flashinfer_root.join("3rdparty/cutlass/include"),
            flashinfer_root.join("cutlass/include"),
        ],
        flashinfer_root.join("cutlass/include"),
    );
    let cutlass_util = first_existing_dir(
        &[
            flashinfer_root.join("3rdparty/cutlass/tools/util/include"),
            flashinfer_root.join("cutlass/tools/util/include"),
        ],
        flashinfer_root.join("cutlass/tools/util/include"),
    );
    let spdlog = first_existing_dir(
        &[
            flashinfer_root.join("3rdparty/spdlog/include"),
            flashinfer_root.join("spdlog/include"),
        ],
        flashinfer_root.join("spdlog/include"),
    );
    let cccl_root = first_existing_dir(
        &[
            flashinfer_root.join("3rdparty/cccl"),
            flashinfer_root.join("cccl"),
        ],
        flashinfer_root.join("3rdparty/cccl"),
    );
    let cccl = vec![
        cccl_root.join("cub"),
        cccl_root.join("libcudacxx/include"),
        cccl_root.join("thrust"),
    ];

    FlashInferIncludes {
        include,
        csrc,
        cutlass,
        cutlass_util,
        spdlog,
        cccl,
    }
}

fn triton_target(sm_targets: &[String]) -> String {
    let max_sm = sm_targets
        .iter()
        .filter_map(|sm| sm_numeric_prefix(sm))
        .max()
        .expect("expected at least one CUDA SM target for Triton AOT");

    if sm_targets.len() > 1 {
        println!(
            "cargo:warning=Triton AOT currently emits one cubin per kernel spec; using highest detected target sm_{max_sm}. Set OPENINFER_CUDA_SM to pin one target explicitly."
        );
    }

    format!("cuda:{max_sm}:32")
}

fn sm_numeric_prefix(sm: &str) -> Option<u32> {
    let digits = sm
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<u32>().ok()
    }
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

    let output = time_phase(format!("triton-gen {}", spec.kernel_name), || {
        Command::new(python)
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
            })
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
    assert!(
        chunkwise_kernel_path.exists(),
        "Qwen3.5 GDR Triton kernel source is missing: {}",
        chunkwise_kernel_path.display()
    );

    {
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
    time_phase("cc triton_kernels_aot", || {
        build.compile("triton_kernels_aot");
    });

    println!("cargo:rustc-link-lib=cuda");
    println!(
        "cargo:warning=Using Triton AOT for Qwen3.5 GDR chunkwise prefill; basic ops (add, silu_mul, embedding) use native CUDA"
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
    println!("cargo:rerun-if-env-changed=OPENINFER_TRITON_PYTHON");
}

fn main() {
    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let cuda_include = Path::new(&cuda_path).join("include");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let sm_targets = detect_sm_targets();
    let nvcc_sm_targets = normalize_nvcc_sms(&sm_targets, &nvcc);
    let arch_args = nvcc_arch_args(&nvcc_sm_targets);
    let deepseek_enabled = cfg!(feature = "deepseek-v4");
    let kimi_k2_enabled = cfg!(feature = "kimi-k2");
    let qwen35_enabled = cfg!(feature = "qwen35-4b");
    let cutedsl_enabled = cfg!(feature = "deepseek-v4");
    let tilelang_artifacts = if deepseek_enabled {
        Some(generate_deepseek_tilelang_artifacts(&out_dir))
    } else {
        None
    };
    let cutedsl_artifacts = if cutedsl_enabled {
        Some(generate_deepseek_cutedsl_artifacts(&out_dir))
    } else {
        None
    };
    println!(
        "cargo:warning=Detected CUDA SM targets: {}",
        sm_targets
            .iter()
            .map(|sm| format!("sm_{sm}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "cargo:warning=Compiling CUDA kernels for nvcc targets: {}",
        nvcc_sm_targets
            .iter()
            .map(|sm| format!("sm_{sm}"))
            .collect::<Vec<_>>()
            .join(",")
    );

    let replaced_cuda_files =
        BTreeSet::from(["activation.cu", "embedding.cu", "fused_attention.cu"]);

    let root = crate_root();
    let csrc_dir = root.join("csrc");
    let mut csrc_files = Vec::new();
    collect_files_recursively(&csrc_dir, &mut csrc_files);
    let mut cu_files: Vec<_> = csrc_files
        .iter()
        .filter_map(|path| {
            let file_name = path.file_name()?.to_str()?;
            if !deepseek_enabled && is_deepseek_v4_source(&csrc_dir, path) {
                return None;
            }
            if !kimi_k2_enabled
                && (is_kimi_k2_source(&csrc_dir, path) || is_deepep_source(&csrc_dir, path))
            {
                return None;
            }
            if path.extension().and_then(|e| e.to_str()) == Some("cu")
                && !replaced_cuda_files.contains(file_name)
            {
                Some(path.clone())
            } else {
                None
            }
        })
        .collect();
    cu_files.sort();
    for path in &csrc_files {
        let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
            continue;
        };
        if matches!(extension, "cu" | "cuh") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    println!(
        "cargo:warning=Legacy CUDA translation units retired from the runtime build: {}",
        replaced_cuda_files
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .join(", ")
    );

    let nvcc_jobs = nvcc_job_count();
    println!("cargo:warning=Compiling CUDA translation units with {nvcc_jobs} nvcc job(s)");
    let flashinfer = flashinfer_includes();
    println!(
        "cargo:warning=Using FlashInfer include dir: {}",
        flashinfer.include.display()
    );

    let deepep_nccl = kimi_k2_enabled.then(deepep_nccl_root);

    let mut nvcc_tasks = Vec::new();
    for cu_file in &cu_files {
        let stem = cu_file.file_stem().unwrap().to_str().unwrap();
        let obj_file = out_dir.join(cuda_object_name(&csrc_dir, cu_file));

        let mut nvcc_args = vec![
            "-c".to_string(),
            cu_file.to_string_lossy().to_string(),
            "-o".to_string(),
            obj_file.to_string_lossy().to_string(),
            "-O3".to_string(),
            "-isystem".to_string(),
            cuda_include.to_string_lossy().to_string(),
            "-I".to_string(),
            csrc_dir.to_string_lossy().to_string(),
        ];
        nvcc_args.extend(arch_args.clone());
        nvcc_args.extend(["--compiler-options".to_string(), "-fPIC".to_string()]);

        // Files that include FlashInfer headers (C++17, header-only)
        if stem == "paged_attention"
            || stem == "flashinfer_norm"
            || stem == "flashinfer_sampling"
            || stem == "flashinfer_top1"
            || stem.starts_with("deepseek_")
        {
            for dir in &flashinfer.cccl {
                nvcc_args.extend(["-I".to_string(), dir.to_string_lossy().to_string()]);
            }
            nvcc_args.extend([
                "--std=c++17".to_string(),
                "-I".to_string(),
                flashinfer.include.to_string_lossy().to_string(),
                "-I".to_string(),
                flashinfer.csrc.to_string_lossy().to_string(),
                "-I".to_string(),
                flashinfer.cutlass.to_string_lossy().to_string(),
                "-I".to_string(),
                flashinfer.cutlass_util.to_string_lossy().to_string(),
                "-I".to_string(),
                flashinfer.spdlog.to_string_lossy().to_string(),
                "-I".to_string(),
                csrc_dir
                    .join("kimi_k2/vllm_marlin")
                    .to_string_lossy()
                    .to_string(),
            ]);
        }

        if stem == "deepseek_quant" {
            nvcc_args.extend([
                "--expt-relaxed-constexpr".to_string(),
                "-static-global-template-stub=false".to_string(),
                "-DFLASHINFER_ENABLE_FP8_E8M0".to_string(),
                "-DFLASHINFER_ENABLE_FP4_E2M1".to_string(),
                "-DCUTLASS_ENABLE_GDC_FOR_SM100=1".to_string(),
            ]);
        }

        // DeepEP elastic shim: mirrors the upstream JIT compile flags
        // (DeepEP csrc/jit/compiler.hpp) minus the cubin plumbing.
        if is_deepep_source(&csrc_dir, cu_file) {
            let nccl_root = deepep_nccl
                .as_ref()
                .expect("deepep sources are collected only with the kimi-k2 feature");
            nvcc_args.extend(
                [
                    "--std=c++20",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--diag-suppress=39,68,161,174,177,186,940,3012",
                    "-Xptxas",
                    "--register-usage-level=10",
                    "-DEP_NUM_TOPK_IDX_BITS=32",
                ]
                .map(str::to_string),
            );
            nvcc_args.extend([
                "-I".to_string(),
                root.join("third_party/DeepEP/deep_ep/include")
                    .to_string_lossy()
                    .to_string(),
                "-I".to_string(),
                nccl_root.join("include").to_string_lossy().to_string(),
            ]);
        }

        if stem.starts_with("kimi_") {
            for dir in &flashinfer.cccl {
                nvcc_args.extend(["-I".to_string(), dir.to_string_lossy().to_string()]);
            }
            nvcc_args.extend([
                "--std=c++17".to_string(),
                "--expt-relaxed-constexpr".to_string(),
                "-I".to_string(),
                flashinfer.include.to_string_lossy().to_string(),
                "-I".to_string(),
                flashinfer.csrc.to_string_lossy().to_string(),
                "-I".to_string(),
                flashinfer.cutlass.to_string_lossy().to_string(),
                "-I".to_string(),
                flashinfer.cutlass_util.to_string_lossy().to_string(),
                "-I".to_string(),
                flashinfer.spdlog.to_string_lossy().to_string(),
                "-I".to_string(),
                csrc_dir
                    .join("kimi_k2/vllm_marlin")
                    .to_string_lossy()
                    .to_string(),
            ]);
        }

        nvcc_tasks.push(NvccTask {
            cu_file: cu_file.clone(),
            obj_file,
            args: nvcc_args,
        });
    }

    if !deepseek_enabled {
        println!(
            "cargo:warning=DeepSeek V4 CUDA/TileLang kernels disabled; enable the openinfer-kernels `deepseek-v4` feature to build them"
        );
    }
    if !kimi_k2_enabled {
        println!(
            "cargo:warning=Kimi-K2 CUDA kernels disabled; enable the openinfer-kernels `kimi-k2` feature to build them"
        );
    }

    if let Some(tilelang_artifacts) = tilelang_artifacts {
        for cu_file in tilelang_artifacts.cu_files {
            let stem = cu_file.file_stem().unwrap().to_str().unwrap();
            let obj_file = out_dir.join(format!("{stem}_cuda.o"));
            let mut nvcc_args = vec![
                "-c".to_string(),
                cu_file.to_string_lossy().to_string(),
                "-o".to_string(),
                obj_file.to_string_lossy().to_string(),
                "-O3".to_string(),
                "-I".to_string(),
                cuda_include.to_string_lossy().to_string(),
            ];
            nvcc_args.extend(arch_args.clone());
            nvcc_args.extend([
                "--std=c++20".to_string(),
                "--compiler-options".to_string(),
                "-fPIC".to_string(),
                "-w".to_string(),
                "-Xcudafe".to_string(),
                "--diag_suppress=177".to_string(),
                "-I".to_string(),
                tilelang_artifacts
                    .template_include
                    .to_string_lossy()
                    .to_string(),
                "-I".to_string(),
                tilelang_artifacts
                    .cutlass_include
                    .to_string_lossy()
                    .to_string(),
            ]);
            nvcc_tasks.push(NvccTask {
                cu_file,
                obj_file,
                args: nvcc_args,
            });
        }
    }

    let mut cutedsl_obj_files = Vec::new();
    let mut cutedsl_runtime_lib_dirs = Vec::new();
    if let Some(cutedsl_artifacts) = cutedsl_artifacts {
        cutedsl_obj_files.extend(cutedsl_artifacts.obj_files);
        cutedsl_runtime_lib_dirs.extend(cutedsl_artifacts.runtime_lib_dirs);
        for cu_file in cutedsl_artifacts.wrapper_files {
            let stem = cu_file.file_stem().unwrap().to_str().unwrap();
            let obj_file = out_dir.join(format!("{stem}_cuda.o"));
            let mut nvcc_args = vec![
                "-c".to_string(),
                cu_file.to_string_lossy().to_string(),
                "-o".to_string(),
                obj_file.to_string_lossy().to_string(),
                "-O3".to_string(),
                "-I".to_string(),
                cuda_include.to_string_lossy().to_string(),
            ];
            nvcc_args.extend(arch_args.clone());
            nvcc_args.extend([
                "--std=c++17".to_string(),
                "--compiler-options".to_string(),
                "-fPIC".to_string(),
                "-I".to_string(),
                cutedsl_artifacts.include_dir.to_string_lossy().to_string(),
            ]);
            nvcc_tasks.push(NvccTask {
                cu_file,
                obj_file,
                args: nvcc_args,
            });
        }
    }

    nvcc_tasks.sort_by_key(|task| nvcc_task_priority(&task.cu_file));

    let task_queue = Mutex::new(VecDeque::from(nvcc_tasks));
    let nvcc_workers = nvcc_jobs.min(task_queue.lock().expect("task queue poisoned").len());
    let mut obj_files = Vec::new();
    thread::scope(|scope| {
        let handles: Vec<_> = (0..nvcc_workers)
            .map(|_| {
                let task_queue = &task_queue;
                let nvcc = nvcc.clone();
                scope.spawn(move || {
                    let mut completed = Vec::new();
                    loop {
                        let task = task_queue.lock().expect("task queue poisoned").pop_front();
                        let Some(task) = task else {
                            break;
                        };

                        let status = time_phase(format!("nvcc {}", task.cu_file.display()), || {
                            Command::new(&nvcc)
                                .args(&task.args)
                                .status()
                                .unwrap_or_else(|_| {
                                    panic!("Failed to run nvcc for {}", task.cu_file.display())
                                })
                        });
                        completed.push((task.cu_file, task.obj_file, status));
                    }
                    completed
                })
            })
            .collect();

        for handle in handles {
            for (cu_file, obj_file, status) in handle.join().expect("nvcc worker panicked") {
                assert!(
                    status.success(),
                    "nvcc compilation failed for {}",
                    cu_file.display()
                );
                obj_files.push(obj_file);
            }
        }
    });
    obj_files.sort();
    obj_files.extend(cutedsl_obj_files);
    obj_files.sort();

    let cuda_lib = out_dir.join("libkernels_cuda.a");
    let _ = fs::remove_file(&cuda_lib);
    let mut ar_args = vec!["rcs".to_string(), cuda_lib.to_string_lossy().to_string()];
    ar_args.extend(
        obj_files
            .into_iter()
            .map(|path| path.to_string_lossy().to_string()),
    );

    let status = time_phase("ar libkernels_cuda.a", || {
        Command::new("ar")
            .args(&ar_args)
            .status()
            .expect("Failed to run ar")
    });

    assert!(status.success(), "ar failed");

    if qwen35_enabled {
        compile_triton_aot_kernels(&cuda_path, &out_dir, &sm_targets);
    } else {
        println!(
            "cargo:warning=Qwen3.5 Triton AOT kernels disabled; enable the openinfer-kernels `qwen35-4b` feature to build them (needs Python + Triton at build time)"
        );
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    for dir in &cutedsl_runtime_lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=static=kernels_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    if let Some(nccl_root) = &deepep_nccl {
        link_deepep_nccl(nccl_root, &out_dir);
    }
    if !cutedsl_runtime_lib_dirs.is_empty() {
        println!("cargo:rustc-link-lib=static=cuda_dialect_runtime_static");
    }
    if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed={}", root.join("csrc").display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=OPENINFER_CUDA_SM");
    println!("cargo:rerun-if-env-changed=CUDA_SM");
    println!("cargo:rerun-if-env-changed=OPENINFER_FLASHINFER_INCLUDE");
    println!("cargo:rerun-if-env-changed=OPENINFER_BUILD_TIMING");
    println!("cargo:rerun-if-env-changed=OPENINFER_NVCC_JOBS");
    println!("cargo:rerun-if-env-changed=OPENINFER_NCCL_ROOT");
    println!(
        "cargo:rerun-if-changed={}",
        root.join("third_party/DeepEP/deep_ep/include").display()
    );
}
