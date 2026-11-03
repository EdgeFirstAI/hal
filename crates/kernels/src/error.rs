// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use core::fmt;
use std::borrow::Cow;

/// Errors returned by kernel plan construction and `run`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum KernelError {
    /// Valid input that this plan does not handle.
    Unsupported(Cow<'static, str>),
    /// Bad parameters, or run views that do not match the plan.
    InvalidArgument(&'static str),
    /// A forced tier needs a CPU feature the running CPU lacks.
    ForcedTierUnavailable {
        tier: &'static str,
        missing_feature: &'static str,
    },
}

/// Result alias using [`KernelError`].
pub type Result<T> = core::result::Result<T, KernelError>;

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
            Self::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Self::ForcedTierUnavailable {
                tier,
                missing_feature,
            } => write!(
                f,
                "forced kernel tier '{tier}' unavailable: CPU lacks {missing_feature}"
            ),
        }
    }
}

impl std::error::Error for KernelError {}
