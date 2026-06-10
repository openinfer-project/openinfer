use std::{
    collections::HashSet,
    env, fs,
    time::{SystemTime, UNIX_EPOCH},
};

use super::{add_python_env_root, nccl_python_wheel_lib_dirs_from_root};

#[test]
fn finds_nccl_python_wheel_lib_dir_from_python_executable() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    let root = env::temp_dir().join(format!(
        "openinfer-nccl-wheel-test-{}-{unique}",
        std::process::id()
    ));
    let python_dir = root.join("bin");
    let wheel_dir = root.join("lib/python3.11/site-packages/nvidia/nccl/lib");
    fs::create_dir_all(&python_dir).expect("create python bin dir");
    fs::create_dir_all(&wheel_dir).expect("create NCCL wheel dir");
    fs::write(wheel_dir.join("libnccl.so.2"), []).expect("create fake NCCL lib marker");

    let mut roots = Vec::new();
    let mut seen = HashSet::new();
    add_python_env_root(&mut roots, &mut seen, &python_dir.join("python"));

    assert_eq!(roots, vec![root.clone()]);
    assert_eq!(nccl_python_wheel_lib_dirs_from_root(&root), vec![wheel_dir]);

    fs::remove_dir_all(root).expect("remove temp root");
}
