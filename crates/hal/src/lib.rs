// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The EdgeFirst Hardware Abstraction Layer, as one dependency.
//!
//! This crate is a re-export shell. Every type you interact with is defined in
//! a sub-crate, and depending on this umbrella costs nothing over depending on
//! them individually — you get one version number and one feature surface
//! instead of five.
//!
//! | Module | Crate | What it does |
//! |---|---|---|
//! | [`tensor`] | [`edgefirst_tensor`] | Zero-copy tensor memory: platform GPU buffer, SHM, PBO, heap |
//! | [`codec`] | [`edgefirst_codec`] | Decodes JPEG/PNG **into** pre-allocated tensors |
//! | [`image`] | [`edgefirst_image`] | Colour conversion, resize, rotation via OpenGL, G2D, or CPU |
//! | [`decoder`] | [`edgefirst_decoder`] | Decodes model **output** tensors into detections |
//! | `tracker` | `edgefirst-tracker` | ByteTrack multi-object tracking (feature `tracker`, off by default) |
//! | [`trace`] | local | Process-wide span capture to Chrome JSON (feature `tracing`) |
//!
//! `codec` and `decoder` are unrelated despite the names: `codec` sits at the
//! front of the pipeline turning image bytes into tensors, `decoder` at the
//! back turning model outputs into detections.
//!
//! # Feature flags
//!
//! `ndarray`, `opengl`, and `tracing` are on by default. `tracker` is opt-in
//! because it pulls in additional dependencies.
//!
//! # Tracing
//!
//! The sub-crates emit [`tracing`] spans on their hot paths that cost a single
//! relaxed atomic load while no subscriber is installed. [`trace::start_tracing`]
//! is what turns them into a trace file:
//!
//! ```no_run
//! # #[cfg(feature = "tracing")]
//! # {
//! edgefirst_hal::trace::start_tracing("/tmp/trace.json").expect("start tracing");
//! // ... run the pipeline ...
//! edgefirst_hal::trace::stop_tracing();
//! # }
//! ```
//!
//! Load the result at <https://ui.perfetto.dev/>. Only one session is
//! supported per process lifetime.

pub use edgefirst_codec as codec;
pub use edgefirst_decoder as decoder;
pub use edgefirst_image as image;
pub use edgefirst_tensor as tensor;
#[cfg(feature = "tracker")]
pub use edgefirst_tracker as tracker;

#[cfg(feature = "tracing")]
pub mod trace;
