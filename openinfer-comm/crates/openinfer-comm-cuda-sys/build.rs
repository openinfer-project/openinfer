use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default feature is OFF: stay completely silent so a barebones dev box
    // (no CUDA SDK installed) can still run `cargo check --workspace`. Do not
    // probe filesystem paths, do not emit `cargo:rerun-if-*`, do not emit
    // `cargo:rustc-link-*`. Anything below this line only runs when the
    // sys-crate-internal `system-bindings` feature is active.
    if env::var_os("CARGO_FEATURE_SYSTEM_BINDINGS").is_none() {
        return Ok(());
    }

    let headers = openinfer_build::cuda_headers("cuda.h");
    let headers: Vec<&str> = headers.iter().map(String::as_str).collect();
    let (cuda_home, cuda_h) = openinfer_build::find_package(
        "cuda-sys",
        "CUDA_HOME",
        &["/usr/local/cuda"],
        &headers,
    );
    let bindings = bindgen::Builder::default()
        .header(cuda_h.to_string_lossy())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"(cu|CU).*")
        .derive_default(true)
        .generate()
        .map_err(|e| {
            format!(
                "cuda-sys build error: failed to generate CUDA driver bindings via bindgen \
                 (looked under CUDA_HOME={}). Underlying error: {}. \
                 Hint: install the CUDA SDK and/or set CUDA_HOME to its install root.",
                cuda_home.display(),
                e
            )
        })?;
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_dir.join("cuda-bindings.rs")).map_err(|e| {
        format!("cuda-sys build error: cannot write cuda-bindings.rs: {}", e)
    })?;

    openinfer_build::link_cuda(&cuda_home, Some("stubs"));
    println!("cargo:rustc-link-lib=cuda");

    Ok(())
}
