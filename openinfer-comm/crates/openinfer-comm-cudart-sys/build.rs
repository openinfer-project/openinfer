use bindgen::callbacks::{ItemInfo, ParseCallbacks};
use std::{env, path::PathBuf};

#[derive(Debug)]
struct RenameCallback;

impl ParseCallbacks for RenameCallback {
    fn item_name(&self, item_info: ItemInfo) -> Option<String> {
        match item_info.name {
            // CUDA 12 defines cudaGetDeviceProperties as cudaGetDeviceProperties_v2.
            // CUDA 13 dropped the _v2 suffix.
            "cudaGetDeviceProperties_v2" => Some("cudaGetDeviceProperties".into()),

            // No rename needed.
            _ => None,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default feature is OFF: stay completely silent.
    if env::var_os("CARGO_FEATURE_SYSTEM_BINDINGS").is_none() {
        return Ok(());
    }

    let toolkit = openinfer_build::CudaToolkit::discover();
    let cuda_include = toolkit.header_dir("cuda.h").unwrap_or_else(|| {
        panic!(
            "cudart-sys build error: cuda.h not found under {}. \
             Hint: install the CUDA SDK and/or set CUDA_HOME to its install root.",
            toolkit.root.display()
        )
    });
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", cuda_include.display()))
        .parse_callbacks(Box::new(RenameCallback))
        .prepend_enum_name(false)
        .allowlist_item(r"cuda.*")
        .derive_default(true)
        .generate()
        .map_err(|e| {
            format!(
                "cudart-sys build error: failed to generate CUDA runtime bindings via bindgen \
                 (looked under {}). Underlying error: {}. \
                 Hint: install the CUDA SDK and/or set CUDA_HOME to its install root.",
                toolkit.root.display(),
                e
            )
        })?;
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_dir.join("cudart-bindings.rs")).map_err(|e| {
        format!("cudart-sys build error: cannot write cudart-bindings.rs: {}", e)
    })?;

    toolkit.link_search();
    println!("cargo:rustc-link-lib=cudart");

    Ok(())
}
