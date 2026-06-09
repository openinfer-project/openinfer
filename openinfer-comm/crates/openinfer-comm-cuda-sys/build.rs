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

    let cuda_home = build_utils::find_package(
        "cuda-sys",
        "CUDA_HOME",
        &["/usr/local/cuda"],
        "include/cuda.h",
    );
    let bindings = bindgen::Builder::default()
        .header(cuda_home.join("include/cuda.h").to_string_lossy())
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

    // Dynamic link dependencies
    println!("cargo:rustc-link-search=native={}/lib64/stubs", cuda_home.display());
    println!("cargo:rustc-link-lib=cuda");

    Ok(())
}
