use std::{
    collections::HashSet,
    env, fs,
    time::{SystemTime, UNIX_EPOCH},
};

use super::{
    AllReduceChunk, NCCL_BF16_ALL_REDUCE_MAX_ELEMS_PER_CALL,
    NCCL_F32_ALL_REDUCE_MAX_ELEMS_PER_CALL, bf16_all_reduce_chunks, f32_all_reduce_chunks,
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

#[test]
fn finds_nccl_python_wheel_lib_dir_with_unversioned_soname() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    let root = env::temp_dir().join(format!(
        "openinfer-nccl-wheel-unversioned-test-{}-{unique}",
        std::process::id()
    ));
    let wheel_dir = root.join("lib/python3.11/site-packages/nvidia/nccl/lib");
    fs::create_dir_all(&wheel_dir).expect("create NCCL wheel dir");
    fs::write(wheel_dir.join("libnccl.so"), []).expect("create fake NCCL lib marker");

    assert_eq!(nccl_python_wheel_lib_dirs_from_root(&root), vec![wheel_dir]);

    fs::remove_dir_all(root).expect("remove temp root");
}
