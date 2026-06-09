use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Default feature is OFF: stay completely silent. Do not invoke Python to
    // discover Torch, do not probe CUDA paths, do not emit any
    // `cargo:rustc-link-*` directives. Anything below only runs when the
    // `hw-cuda` feature is active.
    if env::var_os("CARGO_FEATURE_HW_CUDA").is_none() {
        return;
    }

    let cmake_prefix_path = resolve_torch_cmake_prefix_path();
    let torch_path = resolve_torch_install_root(&cmake_prefix_path);
    let torch_include = torch_path.join("include");
    let torch_lib = torch_path.join("lib");

    let config = pkg_config::Config::new().probe("python3").expect(
        "torch-lib build error: pkg-config could not find `python3`. \
         Hint: install python3-dev / libpython3-dev.",
    );

    cxx_build::bridge("src/hw_cuda_impl.rs")
        .file("src/torch_lib.cc")
        .flag("-Wno-unused-parameter")
        .includes(config.include_paths)
        .include(torch_include)
        .include("/usr/local/cuda/include")
        .std("c++20")
        .compile("torch-lib");

    println!("cargo:rerun-if-changed=src/torch_lib.cc");
    println!("cargo:rerun-if-changed=src/torch_lib.h");

    println!("cargo:rustc-link-search=native={}", torch_lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", torch_lib.display());
    println!("cargo:rustc-link-lib=torch_python");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=c10");
}

/// Resolve PyTorch's CMake prefix path.
///
/// If `TORCH_CMAKE_PREFIX_PATH` is set, the value is used directly (after
/// trimming) but must be non-empty. Otherwise we query the active Python
/// interpreter for `torch.utils.cmake_prefix_path`. Every failure mode
/// reports a `torch-lib build error: ...` message that names the missing
/// component so the caller knows whether to install PyTorch, point at a
/// different interpreter, or set the env var explicitly.
fn resolve_torch_cmake_prefix_path() -> String {
    if let Ok(env_path) = env::var("TORCH_CMAKE_PREFIX_PATH") {
        let trimmed = env_path.trim().to_string();
        if trimmed.is_empty() {
            panic!(
                "torch-lib build error: TORCH_CMAKE_PREFIX_PATH is set but empty. \
                 Hint: unset it to fall back to `python3 -c 'import torch'`, or \
                 set it to the path returned by `python3 -c \
                 'import torch; print(torch.utils.cmake_prefix_path)'`."
            );
        }
        return trimmed;
    }

    let output = Command::new("python3")
        .arg("-W")
        .arg("ignore")
        .arg("-c")
        .arg("import torch; print(torch.utils.cmake_prefix_path)")
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "torch-lib build error: failed to spawn `python3` while \
                 discovering the Torch CMake prefix path: {e}. \
                 Hint: install python3, point CARGO at a working interpreter, \
                 or set TORCH_CMAKE_PREFIX_PATH to skip this probe."
            )
        });

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "torch-lib build error: `python3 -c 'import torch; \
             print(torch.utils.cmake_prefix_path)'` exited with {status}. \
             stderr: {stderr}. \
             Hint: install PyTorch into the active interpreter (e.g. `pip install \
             torch`) or set TORCH_CMAKE_PREFIX_PATH to a valid Torch install.",
            status = output.status,
            stderr = stderr.trim(),
        );
    }

    let stdout = String::from_utf8(output.stdout).unwrap_or_else(|e| {
        panic!(
            "torch-lib build error: `python3 -c 'import torch; \
             print(torch.utils.cmake_prefix_path)'` produced non-UTF-8 stdout: {e}. \
             Hint: set TORCH_CMAKE_PREFIX_PATH explicitly to bypass the probe."
        )
    });

    let path = stdout.trim();
    if path.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "torch-lib build error: `python3 -c 'import torch; \
             print(torch.utils.cmake_prefix_path)'` exited 0 but produced empty \
             stdout. stderr: {stderr}. \
             Hint: verify `python3 -c 'import torch'` works, then re-run; or \
             set TORCH_CMAKE_PREFIX_PATH explicitly.",
            stderr = stderr.trim(),
        );
    }
    path.to_string()
}

/// Walk up two directory levels from the CMake prefix path
/// (`<torch>/share/cmake`) to land on the Torch install root.
///
/// We require both parents to exist on the path so a malformed value
/// caught here, not at link time. The message names which step failed
/// so the caller can correct either the env var or the Python probe
/// result.
fn resolve_torch_install_root(cmake_prefix_path: &str) -> PathBuf {
    let cmake_dir = Path::new(cmake_prefix_path);
    let share_dir = cmake_dir.parent().unwrap_or_else(|| {
        panic!(
            "torch-lib build error: Torch CMake prefix path {cmake_prefix_path:?} \
             has no parent directory. Expected a path of shape \
             `<torch>/share/cmake`. \
             Hint: check TORCH_CMAKE_PREFIX_PATH or the `torch.utils.cmake_prefix_path` \
             output."
        )
    });
    let torch_root = share_dir.parent().unwrap_or_else(|| {
        panic!(
            "torch-lib build error: Torch CMake prefix path {cmake_prefix_path:?} \
             has no grandparent directory. Expected a path of shape \
             `<torch>/share/cmake`. \
             Hint: check TORCH_CMAKE_PREFIX_PATH or the `torch.utils.cmake_prefix_path` \
             output."
        )
    });
    torch_root.to_path_buf()
}
