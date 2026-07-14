use std::{collections::HashSet, fs};

#[cfg(unix)]
use std::os::unix::fs::symlink;

use super::{
    AllReduceChunk, NCCL_BF16_ALL_REDUCE_MAX_ELEMS_PER_CALL,
    NCCL_F32_ALL_REDUCE_MAX_ELEMS_PER_CALL, bf16_all_reduce_chunks, f32_all_reduce_chunks,
    format_nccl_version, validate_nccl_version_for_compute_capabilities,
};
use super::{add_python_env_root, nccl_python_wheel_lib_dirs_from_root};

#[test]
fn f32_all_reduce_chunks_preserve_short_counts_and_split_long_counts() {
    assert!(f32_all_reduce_chunks(0).is_empty());
    assert_eq!(
        f32_all_reduce_chunks(47_104),
        vec![AllReduceChunk {
            offset: 0,
            len: 47_104,
        }]
    );
    assert_eq!(
        f32_all_reduce_chunks(NCCL_F32_ALL_REDUCE_MAX_ELEMS_PER_CALL),
        vec![AllReduceChunk {
            offset: 0,
            len: NCCL_F32_ALL_REDUCE_MAX_ELEMS_PER_CALL,
        }]
    );
    assert_eq!(
        f32_all_reduce_chunks(NCCL_F32_ALL_REDUCE_MAX_ELEMS_PER_CALL + 16_384),
        vec![
            AllReduceChunk {
                offset: 0,
                len: NCCL_F32_ALL_REDUCE_MAX_ELEMS_PER_CALL,
            },
            AllReduceChunk {
                offset: NCCL_F32_ALL_REDUCE_MAX_ELEMS_PER_CALL,
                len: 16_384,
            },
        ]
    );
}

#[test]
fn bf16_all_reduce_chunks_preserve_24_word_count_and_split_long_counts() {
    assert_eq!(
        bf16_all_reduce_chunks(NCCL_BF16_ALL_REDUCE_MAX_ELEMS_PER_CALL),
        vec![AllReduceChunk {
            offset: 0,
            len: NCCL_BF16_ALL_REDUCE_MAX_ELEMS_PER_CALL,
        }]
    );
    assert_eq!(
        bf16_all_reduce_chunks(NCCL_BF16_ALL_REDUCE_MAX_ELEMS_PER_CALL + 45_056),
        vec![
            AllReduceChunk {
                offset: 0,
                len: NCCL_BF16_ALL_REDUCE_MAX_ELEMS_PER_CALL,
            },
            AllReduceChunk {
                offset: NCCL_BF16_ALL_REDUCE_MAX_ELEMS_PER_CALL,
                len: 45_056,
            },
        ]
    );
}

#[test]
fn finds_nccl_python_wheel_lib_dir_from_python_executable() {
    let root = tempfile::tempdir().expect("create temp root");
    let python_dir = root.path().join("bin");
    let wheel_dir = root
        .path()
        .join("lib/python3.11/site-packages/nvidia/nccl/lib");
    fs::create_dir_all(&python_dir).expect("create python bin dir");
    fs::create_dir_all(&wheel_dir).expect("create NCCL wheel dir");
    fs::write(wheel_dir.join("libnccl.so.2"), []).expect("create fake NCCL lib marker");

    let mut roots = Vec::new();
    let mut seen = HashSet::new();
    add_python_env_root(&mut roots, &mut seen, &python_dir.join("python"));

    assert_eq!(roots, vec![root.path().to_path_buf()]);
    assert_eq!(
        nccl_python_wheel_lib_dirs_from_root(root.path()),
        vec![wheel_dir]
    );
}

#[cfg(unix)]
#[test]
fn keeps_symlinked_python_venv_root_before_resolved_root() {
    let real_root = tempfile::tempdir().expect("create real Python root");
    let link_root = tempfile::tempdir().expect("create symlink root");
    let real_bin = real_root.path().join("bin");
    let link_bin = link_root.path().join("bin");
    fs::create_dir_all(&real_bin).expect("create real bin dir");
    fs::create_dir_all(&link_bin).expect("create symlink bin dir");
    let real_python = real_bin.join("python3.12");
    fs::write(&real_python, []).expect("create Python marker");
    let linked_python = link_bin.join("python3");
    symlink(&real_python, &linked_python).expect("create Python symlink");

    let mut roots = Vec::new();
    let mut seen = HashSet::new();
    add_python_env_root(&mut roots, &mut seen, &linked_python);

    assert_eq!(
        roots,
        vec![
            link_root.path().to_path_buf(),
            real_root.path().to_path_buf()
        ]
    );
}

#[test]
fn finds_nccl_python_wheel_lib_dir_with_unversioned_soname() {
    let root = tempfile::tempdir().expect("create temp root");
    let wheel_dir = root
        .path()
        .join("lib/python3.11/site-packages/nvidia/nccl/lib");
    fs::create_dir_all(&wheel_dir).expect("create NCCL wheel dir");
    fs::write(wheel_dir.join("libnccl.so"), []).expect("create fake NCCL lib marker");

    assert_eq!(
        nccl_python_wheel_lib_dirs_from_root(root.path()),
        vec![wheel_dir]
    );
}

#[test]
fn formats_nccl_version_code() {
    assert_eq!(format_nccl_version(22_602), "2.26.2");
    assert_eq!(format_nccl_version(22_707), "2.27.7");
    assert_eq!(format_nccl_version(22_501), "2.25.1");
}

#[test]
fn sm120_rejects_nccl_before_shared_memory_fix() {
    let error = validate_nccl_version_for_compute_capabilities(22_601, &[(12, 0)])
        .expect_err("NCCL 2.26.1 predates the sm_120 shared-memory fix");
    let message = error.to_string();
    assert!(message.contains("requires NCCL >= 2.26.2"));
    assert!(message.contains("loaded 2.26.1"));
    assert!(message.contains("OPENINFER_NCCL_LIB_DIR"));
}

#[test]
fn sm120_accepts_nccl_2_26_2() {
    validate_nccl_version_for_compute_capabilities(22_602, &[(12, 0), (12, 0)])
        .expect("NCCL 2.26.2 contains the sm_120 shared-memory fix");
}

#[test]
fn non_sm120_capabilities_do_not_inherit_the_sm120_floor() {
    validate_nccl_version_for_compute_capabilities(22_501, &[(8, 0), (10, 0), (12, 1)])
        .expect("the sm_120 workaround must stay scoped to compute capability 12.0");
}
