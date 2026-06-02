#![cfg(feature = "tvm-ffi-interop")]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, ensure};
use tvm_ffi::{Error, Function, Module, Result as TvmResult, VALUE_ERROR, into_typed_fn};

// Manual command:
// cargo test --release -p pegainfer-kernels --features tvm-ffi-interop tvm_ffi_bidirectional -- --ignored --nocapture

static FIXTURE_LIB: OnceLock<PathBuf> = OnceLock::new();

fn tvm<T>(result: TvmResult<T>, context: impl AsRef<str>) -> Result<T> {
    result.map_err(|err| anyhow!("{}: {err}", context.as_ref()))
}

fn fixture_extension() -> &'static str {
    if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    }
}

fn shell_quote(path: &Path) -> String {
    let raw = path.to_string_lossy();
    format!("'{}'", raw.replace('\'', "'\"'\"'"))
}

fn run_command(command: &str, context: &str) -> Result<std::process::Output> {
    let output = Command::new("sh")
        .arg("-lc")
        .arg(command)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("failed to launch shell for {context}"))?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(anyhow!(
            "{context} failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn require_tvm_ffi_config() -> Result<PathBuf> {
    let output = Command::new("tvm-ffi-config")
        .arg("--libdir")
        .stdin(Stdio::null())
        .output()
        .context(
            "failed to run tvm-ffi-config --libdir; install apache-tvm-ffi and ensure tvm-ffi-config is on PATH before enabling tvm-ffi-interop",
        )?;
    ensure!(
        output.status.success(),
        "tvm-ffi-config --libdir failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let libdir = String::from_utf8(output.stdout)
        .context("tvm-ffi-config produced non-UTF8 output")?
        .trim()
        .to_string();
    ensure!(
        !libdir.is_empty(),
        "tvm-ffi-config --libdir returned an empty path"
    );
    Ok(PathBuf::from(libdir))
}

fn build_fixture_library() -> Result<PathBuf> {
    if let Some(path) = FIXTURE_LIB.get() {
        return Ok(path.clone());
    }

    let _libdir = require_tvm_ffi_config()?;
    let cxx = std::env::var("PEGAINFER_TVM_FFI_CXX").unwrap_or_else(|_| "c++".to_string());
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tvm_ffi_fixture.cc");
    ensure!(
        source.is_file(),
        "fixture source missing: {}",
        source.display()
    );

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_nanos();
    let build_dir = std::env::temp_dir().join(format!("pegainfer-tvm-ffi-{stamp}"));
    std::fs::create_dir_all(&build_dir)
        .with_context(|| format!("failed to create {}", build_dir.display()))?;
    let output = build_dir.join(format!("tvm_ffi_fixture.{}", fixture_extension()));

    let command = format!(
        "{cxx} -shared -O3 -std=c++17 -fPIC -fvisibility=hidden \
         -o {output} {source} \
         $(tvm-ffi-config --cxxflags) \
         $(tvm-ffi-config --ldflags) \
         $(tvm-ffi-config --libs)",
        output = shell_quote(&output),
        source = shell_quote(&source),
    );
    run_command(&command, "building TVM FFI fixture library").with_context(|| {
        format!(
            "failed to build TVM FFI fixture with compiler {cxx}; override with PEGAINFER_TVM_FFI_CXX if needed"
        )
    })?;

    let _ = FIXTURE_LIB.set(output.clone());
    Ok(output)
}

#[test]
#[ignore = "requires tvm-ffi runtime/tooling on PATH; run manually on a host with apache-tvm-ffi installed"]
fn tvm_ffi_bidirectional_smoke() -> Result<()> {
    let fixture = build_fixture_library()?;
    let fixture_path = fixture.to_string_lossy();
    let module = tvm(
        Module::load_from_file(fixture_path.as_ref()),
        format!(
            "failed to load fixture module {}; verify libtvm_ffi runtime libraries are visible",
            fixture.display()
        ),
    )?;

    let add_one = tvm(
        module.get_function("add_one_scalar"),
        "missing add_one_scalar export in TVM FFI fixture module",
    )?;
    let add_one = into_typed_fn!(add_one, Fn(i64) -> TvmResult<i64>);
    assert_eq!(tvm(add_one(41), "calling add_one_scalar")?, 42);

    let apply_callback = tvm(
        module.get_function("apply_callback"),
        "missing apply_callback export in TVM FFI fixture module",
    )?;
    let apply_callback = into_typed_fn!(apply_callback, Fn(Function, i64) -> TvmResult<i64>);
    let host_add_five = Function::from_typed(|x: i64| -> TvmResult<i64> { Ok(x + 5) });
    assert_eq!(
        tvm(
            apply_callback(host_add_five, 7),
            "calling apply_callback with a Rust callback",
        )?,
        12
    );

    tvm(
        Function::register_global(
            "pegainfer.testing.add_three",
            Function::from_typed(|x: i64| -> TvmResult<i64> { Ok(x + 3) }),
        ),
        "failed to register pegainfer.testing.add_three",
    )?;
    let call_registered = tvm(
        module.get_function("call_registered_host_add_three"),
        "missing call_registered_host_add_three export in TVM FFI fixture module",
    )?;
    let call_registered = into_typed_fn!(call_registered, Fn(i64) -> TvmResult<i64>);
    assert_eq!(
        tvm(call_registered(9), "calling call_registered_host_add_three")?,
        12
    );

    tvm(
        Function::register_global(
            "pegainfer.testing.fail_if_negative",
            Function::from_typed(|x: i64| -> TvmResult<i64> {
                if x < 0 {
                    Err(Error::new(
                        VALUE_ERROR,
                        "negative input rejected by Rust callback",
                        "",
                    ))
                } else {
                    Ok(x)
                }
            }),
        ),
        "failed to register pegainfer.testing.fail_if_negative",
    )?;
    let fail_callback = tvm(
        module.get_function("call_registered_host_fail_if_negative"),
        "missing call_registered_host_fail_if_negative export in TVM FFI fixture module",
    )?;
    let fail_callback = into_typed_fn!(fail_callback, Fn(i64) -> TvmResult<i64>);
    let err = tvm(
        fail_callback(-1),
        "calling call_registered_host_fail_if_negative",
    )
    .expect_err("negative callback should propagate an error");
    let err_text = err.to_string();
    assert!(
        err_text.contains("negative input rejected by Rust callback"),
        "unexpected callback error: {err_text}"
    );

    let missing = match module.get_function("does_not_exist") {
        Ok(_) => panic!("missing TVM FFI symbol lookup should fail"),
        Err(err) => err,
    };
    let missing_text = missing.to_string();
    assert!(
        missing_text.contains("Cannot convert from type `None` to `ffi.Function`"),
        "unexpected missing-symbol error: {missing_text}"
    );

    Ok(())
}
