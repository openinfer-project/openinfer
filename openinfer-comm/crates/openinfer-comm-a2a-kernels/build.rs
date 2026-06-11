use std::env;
use std::path::PathBuf;

fn main() {
    // Default feature is OFF: stay completely silent. Do not invoke nvcc, do
    // not probe CUDA paths, do not emit `cargo:rustc-link-*`. Anything below
    // only runs when the `hw-cuda` feature is active.
    if env::var_os("CARGO_FEATURE_HW_CUDA").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let toolkit = openinfer_build::CudaToolkit::discover();

    if env::var_os("NVCC").is_none() {
        // SAFETY: build scripts run single-threaded before any thread spawns.
        unsafe { env::set_var("NVCC", &toolkit.nvcc) };
    }

    cxx_build::bridge("src/hw_cuda_impl.rs")
        .debug(false)
        .cuda(true)
        .flag("-t0")
        .flag("-O3")
        .flag("-cudart=shared")
        .flag("-gencode=arch=compute_90a,code=sm_90a")
        .flag("-gencode=arch=compute_100a,code=sm_100a")
        .flag(format!("-I{}/src", manifest_dir.display()))
        .file("src/a2a/a2a_dispatch_recv.cu")
        .file("src/a2a/a2a_combine_send.cu")
        .file("src/a2a/a2a_combine_recv.cu")
        .file("src/a2a/a2a_dispatch_send.cu")
        .compile("liba2a_kernels.a");

    openinfer_build::emit_rerun_if_changed_files("src", &["cu", "cuh", "h"]);

    toolkit.link_search();
    println!("cargo:rustc-link-lib=cudart");
}
