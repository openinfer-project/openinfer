// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod external;
mod logical;

#[cfg(test)]
mod tests;

pub use external::ExternalBlockAssignments;
pub use external::zip_assigned;
pub use external::zip_assigned_pending;
pub use logical::LogicalBlockAssignmentError;
pub use logical::LogicalBlockAssignments;
