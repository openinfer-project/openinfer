pub mod config;
pub mod weights; // weight name mapping (pure logic)

#[cfg(feature = "higgs-audio")]
pub mod backbone; // GPU forward, only compiled under feature
