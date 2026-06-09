// fabric-debug: an interactive RDMA Verbs debugging binary.
//
// The full implementation lives in `hw_rdma_impl` and only compiles when the
// `hw-rdma` feature is enabled. Default-off builds compile a tiny stub `main`
// so the binary still appears in `cargo build --workspace` output without
// dragging in CUDA / libibverbs / GDRCopy.

#[cfg(feature = "hw-rdma")]
mod hw_rdma_impl;

#[cfg(feature = "hw-rdma")]
fn main() -> anyhow::Result<()> {
    hw_rdma_impl::run()
}

#[cfg(not(feature = "hw-rdma"))]
fn main() {
    eprintln!(
        "fabric-debug: built without `hw-rdma` feature; rebuild with \
         `--features hw-rdma` on a host that has CUDA + libibverbs installed."
    );
    std::process::exit(2);
}
