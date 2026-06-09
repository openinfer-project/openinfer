use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default feature is OFF: stay completely silent.
    if env::var_os("CARGO_FEATURE_SYSTEM_BINDINGS").is_none() {
        return Ok(());
    }

    let gdrapi_home = build_utils::find_package(
        "gdrapi-sys",
        "GDRAPI_HOME",
        &["/usr"],
        "include/gdrapi.h",
    );
    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include <gdrapi.h>")
        .clang_arg(format!("-I{}/include", gdrapi_home.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"gdr.*")
        .derive_default(true)
        .layout_tests(false)
        .generate()
        .map_err(|e| {
            format!(
                "gdrapi-sys build error: failed to generate gdrapi bindings via bindgen \
                 (looked under GDRAPI_HOME={}). Underlying error: {}. \
                 Hint: install GDRCopy (rdma-core/gdrcopy) and/or set GDRAPI_HOME.",
                gdrapi_home.display(),
                e
            )
        })?;
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_dir.join("gdrapi-bindings.rs")).map_err(|e| {
        format!("gdrapi-sys build error: cannot write gdrapi-bindings.rs: {}", e)
    })?;

    // Dynamic link dependencies
    println!("cargo:rustc-link-lib=gdrapi");
    println!("cargo:rustc-link-search=native={}/lib", gdrapi_home.display());

    Ok(())
}
