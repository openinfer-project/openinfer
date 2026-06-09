use std::borrow::Cow;

use crate::{provider::RdmaDomainInfo, verbs::VerbsDeviceInfo};

// NOTE: This enum was originally a dispatch point between multiple RDMA
// providers (EFA + Verbs upstream). The non-Verbs provider was removed during
// the port, but we keep the enum so that future fabric providers can be added
// without breaking the public API surface.
#[derive(Clone)]
pub enum DomainInfo {
    Verbs(VerbsDeviceInfo),
}

impl RdmaDomainInfo for DomainInfo {
    fn name(&self) -> Cow<'_, str> {
        match self {
            DomainInfo::Verbs(info) => info.name(),
        }
    }

    fn link_speed(&self) -> u64 {
        match self {
            DomainInfo::Verbs(info) => info.link_speed(),
        }
    }
}
