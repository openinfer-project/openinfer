use std::path::{Path, PathBuf};

fn cuda_root() -> PathBuf {
    std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .map_or_else(|_| PathBuf::from("/usr/local/cuda"), PathBuf::from)
}

fn add_if_exists(build: &mut cc::Build, path: &Path) {
    if path.exists() {
        build.include(path);
    }
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let cuda_root = cuda_root();
    let cuda_include = cuda_root.join("include");
    let cuda_target_include = cuda_root.join("targets/x86_64-linux/include");
    let cupti_include = cuda_root.join("extras/CUPTI/include");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .file(manifest_dir.join("csrc/range_profiler.cpp"));
    add_if_exists(&mut build, &cuda_include);
    add_if_exists(&mut build, &cuda_target_include);
    add_if_exists(&mut build, &cupti_include);
    build.compile("openinfer_cupti_range_profiler");

    let cuda_lib64 = cuda_root.join("lib64");
    let cuda_target_lib = cuda_root.join("targets/x86_64-linux/lib");
    let cupti_lib64 = cuda_root.join("extras/CUPTI/lib64");
    for path in [&cuda_lib64, &cuda_target_lib, &cupti_lib64] {
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cupti");
    if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed=csrc/range_profiler.cpp");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
