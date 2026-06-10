use std::{
    env,
    path::{Path, PathBuf},
};

/// Finds a package's install root: probes `$env_var` first, then each of
/// `default_paths`, for any of the `check_files` — several cover layout
/// variants like `include/` vs `targets/<arch>/include/`. Returns the
/// matched root and check file.
///
/// # Panics
/// When nothing matches.
pub fn find_package(
    provider: &str,
    env_var: &str,
    default_paths: &[&str],
    check_files: &[&str],
) -> (PathBuf, PathBuf) {
    println!("cargo:rerun-if-env-changed={}", env_var);
    let env_root = env::var_os(env_var).map(PathBuf::from);
    let roots: Vec<PathBuf> = env_root
        .clone()
        .into_iter()
        .chain(default_paths.iter().map(PathBuf::from))
        .collect();
    for root in &roots {
        for check in check_files {
            let found = root.join(check);
            if found.is_file() {
                if let Some(env_root) = &env_root
                    && env_root != root
                {
                    println!(
                        "cargo:warning={provider}: ${env_var} ({}) contains none of \
                         {check_files:?}; using {} instead",
                        env_root.display(),
                        root.display()
                    );
                }
                return (root.clone(), found);
            }
        }
    }
    panic!(
        "{provider} build error: none of {check_files:?} found. \
         Looked at `${env_var}` ({env_status}) and default paths {default_paths:?}. \
         Hint: install the provider headers or set `{env_var}` to their install root.",
        env_status = env_root
            .map(|root| format!("set to {root:?}"))
            .unwrap_or_else(|| "unset".to_string()),
    )
}

/// `targets/<dir>` names for the build target; aarch64 toolkits ship as
/// either `aarch64-linux` or `sbsa-linux`. Host arch outside build scripts.
fn target_dirs() -> Vec<String> {
    let arch =
        env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| std::env::consts::ARCH.to_string());
    match arch.as_str() {
        "aarch64" => vec!["aarch64-linux".to_string(), "sbsa-linux".to_string()],
        arch => vec![format!("{arch}-linux")],
    }
}

/// Relative candidate paths for a CUDA header across the layouts of
/// [`cuda_libs`].
pub fn cuda_headers(header: &str) -> Vec<String> {
    let mut headers = vec![format!("include/{header}")];
    for target in target_dirs() {
        headers.push(format!("targets/{target}/include/{header}"));
    }
    headers
}

/// Emits `cargo:rustc-link-search` for [`cuda_libs`], warning when nothing
/// matched so a failing link points back at the probed root.
pub fn link_cuda(root: &Path, suffix: Option<&str>) {
    let dirs = cuda_libs(root, suffix);
    if dirs.is_empty() {
        println!(
            "cargo:warning=no CUDA library dir found under {} (suffix: {})",
            root.display(),
            suffix.unwrap_or("none"),
        );
    }
    for dir in dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
}

/// The existing CUDA library dirs under `root`: `lib64` (classic), `lib`
/// (conda), `targets/<arch>/lib`, and the NVIDIA HPC SDK `math_libs/<ver>`
/// sibling tree, where cuBLAS lives outside the cuda dir. `suffix` narrows
/// each candidate, e.g. `Some("stubs")` for driver stubs. Kept free of cargo
/// output so layout coverage is unit-testable on synthetic trees.
pub fn cuda_libs(root: &Path, suffix: Option<&str>) -> Vec<PathBuf> {
    let mut subdirs = vec!["lib64".to_string(), "lib".to_string()];
    for target in target_dirs() {
        subdirs.push(format!("targets/{target}/lib"));
    }
    let mut dirs: Vec<PathBuf> = subdirs.iter().map(|sub| root.join(sub)).collect();
    // HPC SDK roots look like .../hpc_sdk/<os>/<release>/cuda/<ver>; the math
    // libraries live in the <release>/math_libs/<ver> sibling tree.
    if let (Some(version), Some(release)) = (root.file_name(), root.parent().and_then(Path::parent))
    {
        let math = release.join("math_libs").join(version);
        dirs.push(math.join("lib64"));
        dirs.push(math.join("lib"));
    }
    dirs.into_iter()
        .map(|dir| match suffix {
            Some(suffix) => dir.join(suffix),
            None => dir,
        })
        .filter(|dir| dir.is_dir())
        .collect()
}

/// Recursively emits `cargo:rerun-if-changed` for all files under `src_dir`
/// with one of the given `extensions`.
pub fn emit_rerun_if_changed_files(src_dir: &str, extensions: &[&str]) {
    fn visit_dir(dir: &Path, extensions: &[&str]) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dir(&path, extensions)?;
            } else if let Some(ext) = path.extension().and_then(|s| s.to_str())
                && extensions.contains(&ext)
            {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
        Ok(())
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.join(src_dir);

    if let Err(err) = visit_dir(&root, extensions) {
        eprintln!("cargo:warning=Failed to scan {}: {}", root.display(), err);
    }

    // Also watch the directory itself so new files trigger rebuilds
    println!("cargo:rerun-if-changed={}", root.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempTree(PathBuf);

    impl TempTree {
        fn new(name: &str) -> Self {
            let root =
                std::env::temp_dir().join(format!("openinfer-build-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            Self(root)
        }

        fn mkdirs(&self, rel: &str) -> PathBuf {
            let dir = self.0.join(rel);
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }

        fn touch(&self, rel: &str) {
            let file = self.0.join(rel);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, b"").unwrap();
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn target_dir() -> String {
        target_dirs().remove(0)
    }

    #[test]
    fn classic_layout_finds_header_and_lib64() {
        let tree = TempTree::new("classic");
        tree.touch("include/cuda.h");
        let lib64 = tree.mkdirs("lib64");
        tree.mkdirs("lib64/stubs");

        let candidates = cuda_headers("cuda.h");
        let candidates: Vec<&str> = candidates.iter().map(String::as_str).collect();
        let root_str = tree.0.to_str().unwrap().to_string();
        let (root, header) = find_package(
            "test",
            "OPENINFER_TEST_UNSET_ENV",
            &[&root_str],
            &candidates,
        );
        assert_eq!(root, tree.0);
        assert_eq!(header, tree.0.join("include/cuda.h"));

        assert_eq!(cuda_libs(&tree.0, None), vec![lib64.clone()]);
        assert_eq!(cuda_libs(&tree.0, Some("stubs")), vec![lib64.join("stubs")]);
    }

    #[test]
    fn conda_layout_finds_targets_header_and_lib() {
        let tree = TempTree::new("conda");
        let target = target_dir();
        tree.touch(&format!("targets/{target}/include/cuda.h"));
        let lib = tree.mkdirs("lib");
        let targets_lib = tree.mkdirs(&format!("targets/{target}/lib"));

        let candidates = cuda_headers("cuda.h");
        let candidates: Vec<&str> = candidates.iter().map(String::as_str).collect();
        let root_str = tree.0.to_str().unwrap().to_string();
        let (_, header) = find_package(
            "test",
            "OPENINFER_TEST_UNSET_ENV",
            &[&root_str],
            &candidates,
        );
        assert_eq!(
            header,
            tree.0.join(format!("targets/{target}/include/cuda.h"))
        );

        assert_eq!(cuda_libs(&tree.0, None), vec![lib, targets_lib]);
    }

    #[test]
    fn hpc_sdk_layout_adds_math_libs_sibling() {
        let tree = TempTree::new("hpcsdk");
        let cuda_root = tree.mkdirs("release/cuda/12.6");
        let cuda_lib64 = tree.mkdirs("release/cuda/12.6/lib64");
        let math_lib64 = tree.mkdirs("release/math_libs/12.6/lib64");

        assert_eq!(cuda_libs(&cuda_root, None), vec![cuda_lib64, math_lib64]);
    }

    #[test]
    fn unknown_layout_yields_no_dirs() {
        let tree = TempTree::new("unknown");
        tree.mkdirs("weird/place");
        assert!(cuda_libs(&tree.0, None).is_empty());
    }

    #[test]
    #[should_panic(expected = "none of")]
    fn missing_header_panics_with_all_candidates() {
        let tree = TempTree::new("empty");
        let root_str = tree.0.to_str().unwrap().to_string();
        find_package(
            "test",
            "OPENINFER_TEST_UNSET_ENV",
            &[&root_str],
            &["include/cuda.h"],
        );
    }
}
