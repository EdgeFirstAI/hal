// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `EDGEFIRST_DISABLE_V4L2` turns the V4L2 JPEG probe off.
#![cfg(all(target_os = "linux", feature = "v4l2"))]

#[path = "common/sessions.rs"]
mod sessions;

/// `v4l2_available` answers with or without a decoder device, and
/// `EDGEFIRST_DISABLE_V4L2` forces `false`.
#[test]
fn v4l2_available_probe_honours_disable_env() {
    let _sessions = sessions::take(1);
    std::env::set_var("EDGEFIRST_DISABLE_V4L2", "1");
    assert!(!edgefirst_codec::v4l2_available());
    std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
    let _present: bool = edgefirst_codec::v4l2_available();
}
