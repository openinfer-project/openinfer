// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Composition layer wiring together [`BlockSequence`](crate::BlockSequence),
//! [`LogicalBlockAssignments`](crate::LogicalBlockAssignments), and
//! [`BlockManager`](crate::BlockManager) into higher-level request lifecycle types.

mod request;
mod scheduled;

pub use request::RequestSequence;
pub use scheduled::ApplyError;
pub use scheduled::DecodeOutcome;
pub use scheduled::NoopDelegate;
pub use scheduled::SchedulableSequence;
pub use scheduled::SchedulableSequenceBuilder;
pub use scheduled::ScheduleError;
pub use scheduled::SequenceDelegate;
pub use scheduled::SequenceEvent;
pub use scheduled::SequenceState;
