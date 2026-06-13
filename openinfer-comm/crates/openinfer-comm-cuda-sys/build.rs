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

    let toolkit = openinfer_build::CudaToolkit::discover();
    let cuda_h = toolkit
        .header_dir("cuda.h")
        .map(|dir| dir.join("cuda.h"))
        .unwrap_or_else(|| {
            panic!(
                "cuda-sys build error: cuda.h not found under {}. \
                 Hint: install the CUDA SDK and/or set CUDA_HOME to its install root.",
                toolkit.root.display()
            )
        });
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
                 (looked under {}). Underlying error: {}. \
                 Hint: install the CUDA SDK and/or set CUDA_HOME to its install root.",
                toolkit.root.display(),
                e
            )
        })?;
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_dir.join("cuda-bindings.rs")).map_err(|e| {
        format!("cuda-sys build error: cannot write cuda-bindings.rs: {}", e)
    })?;

    toolkit.link_search_stubs();
    println!("cargo:rustc-link-lib=cuda");

    Ok(())
}
