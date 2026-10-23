// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The `content_region=` argument shared by image drawing and the decoder:
//! where the image content sits in the model input.

use pyo3::prelude::*;

/// A content region as normalized `[xmin, ymin, xmax, ymax]`: a 4-sequence,
/// or any object with a `normalized` attribute such as
/// `edgefirst.image.ContentRegion` (duck-typed so it crosses packages).
#[derive(FromPyObject)]
pub enum ContentRegionArg {
    Values([f32; 4]),
    Region {
        #[pyo3(attribute("normalized"))]
        normalized: [f32; 4],
    },
}

impl ContentRegionArg {
    /// The normalized rectangle.
    pub fn normalized(&self) -> [f32; 4] {
        match self {
            Self::Values(v) => *v,
            Self::Region { normalized } => *normalized,
        }
    }
}

/// The normalized rectangle of an optional argument.
pub fn normalized(arg: Option<ContentRegionArg>) -> Option<[f32; 4]> {
    arg.map(|a| a.normalized())
}
