//! Python extension module for `pplx-garden` (upstream-derived).
//!
//! When the `hw-rdma` feature is enabled, this cdylib exposes the full Python
//! API (CudaMem, fabric, p2p-all-to-all). When the feature is disabled (the
//! default), the cdylib is a near-empty PyO3 module that just registers a
//! `HW_RDMA_ENABLED = False` attribute so importing the wheel never fails on
//! a non-hardware host (useful for `cargo build --workspace` and packaging
//! smoke tests).

use pyo3::{
    Bound, PyResult, pymodule,
    types::{PyModule, PyModuleMethods},
};

#[cfg(feature = "hw-rdma")]
mod py_cumem;
#[cfg(feature = "hw-rdma")]
mod py_device;
#[cfg(feature = "hw-rdma")]
mod py_fabric_lib;
#[cfg(feature = "hw-rdma")]
mod py_p2p_all_to_all;

#[cfg(feature = "hw-rdma")]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = logging_lib::init(&logging_lib::LoggingOpts {
        log_color: logging_lib::LogColor::Auto,
        log_format: logging_lib::LogFormat::Text,
        log_directives: None,
    });

    py_cumem::init(m)?;
    py_p2p_all_to_all::init(m)?;
    py_fabric_lib::init(m)?;

    Ok(())
}

#[cfg(not(feature = "hw-rdma"))]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("HW_RDMA_ENABLED", false)?;
    Ok(())
}
