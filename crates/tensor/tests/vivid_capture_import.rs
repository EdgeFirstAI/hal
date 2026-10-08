// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A V4L2 capture buffer with padded rows, exported with `EXPBUF` and
//! imported through `Tensor::from_fd`, maps at its row stride and reads the
//! pixels the driver wrote.
//!
//! The device is the kernel's virtual `vivid` driver, which honours a
//! requested `bytesperline`. Load it with
//! `sudo modprobe vivid n_devs=2 node_types=0x1,0x1 multiplanar=1,2` and
//! give the user read-write access to its nodes. Without it the tests skip;
//! `HAL_TEST_REQUIRE_VIVID=1` makes that a failure.
#![cfg(all(target_os = "linux", feature = "static"))]

use std::time::Duration;

use edgefirst_tensor::{PixelFormat, Tensor, TensorMapTrait, TensorMemory, TensorTrait};
use edgefirst_v4l2::device::{self, Device};
use edgefirst_v4l2::queue::{BufType, Memory, Plane, Queue};
use edgefirst_v4l2::uapi;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
/// YUYV is 2 bytes per pixel, so the natural pitch is 1280 bytes.
const NATURAL: usize = WIDTH as usize * 2;
/// A pitch with 256 bytes of padding per row.
const PADDED: u32 = 1536;

/// The first vivid node offering `buf_type`, or a skip.
fn vivid(buf_type: BufType) -> Option<Device> {
    let found = device::enumerate().ok().and_then(|nodes| {
        nodes.into_iter().find_map(|node| {
            let caps = node.capabilities.ok()?;
            (caps.driver == "vivid" && caps.capture_buf_type() == Some(buf_type))
                .then(|| Device::open(&node.path).ok())
                .flatten()
        })
    });
    if found.is_none() {
        let why = format!("no vivid {buf_type:?} node");
        assert!(
            std::env::var("HAL_TEST_REQUIRE_VIVID").as_deref() != Ok("1"),
            "HAL_TEST_REQUIRE_VIVID=1 but the test would have skipped: {why}"
        );
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), "SKIPPED: {why}");
    }
    found
}

/// Set YUYV at the padded pitch; returns the pitch and image size the
/// driver chose.
fn set_padded_yuyv(dev: &Device, buf_type: BufType) -> (usize, usize) {
    let mut fmt = uapi::v4l2_format {
        type_: buf_type.raw(),
        ..Default::default()
    };
    // SAFETY: `type_` selects the payload each branch writes and reads.
    unsafe {
        if buf_type == BufType::VideoCaptureMplane {
            let p = fmt.pix_mp();
            p.width = WIDTH;
            p.height = HEIGHT;
            p.pixelformat = uapi::V4L2_PIX_FMT_YUYV;
            p.field = uapi::V4L2_FIELD_NONE;
            p.num_planes = 1;
            p.plane_fmt[0].bytesperline = PADDED;
        } else {
            let p = fmt.pix();
            p.width = WIDTH;
            p.height = HEIGHT;
            p.pixelformat = uapi::V4L2_PIX_FMT_YUYV;
            p.field = uapi::V4L2_FIELD_NONE;
            p.bytesperline = PADDED;
        }
    }
    dev.set_format(&mut fmt).expect("S_FMT YUYV");
    // SAFETY: as above.
    let (w, h, pitch, size) = unsafe {
        if buf_type == BufType::VideoCaptureMplane {
            let p = fmt.pix_mp();
            (
                p.width,
                p.height,
                p.plane_fmt[0].bytesperline,
                p.plane_fmt[0].sizeimage,
            )
        } else {
            let p = fmt.pix();
            (p.width, p.height, p.bytesperline, p.sizeimage)
        }
    };
    assert_eq!((w, h), (WIDTH, HEIGHT));
    assert_eq!(pitch, PADDED, "vivid did not keep the padded bytesperline");
    (pitch as usize, size as usize)
}

fn padded_capture_imports_and_maps_at_its_stride(buf_type: BufType) {
    let Some(dev) = vivid(buf_type) else {
        return;
    };
    let (pitch, size) = set_padded_yuyv(&dev, buf_type);
    assert!(size >= pitch * HEIGHT as usize);

    let mut queue = Queue::new(&dev, buf_type).expect("queue");
    let count = queue.request(Memory::Mmap, 2).expect("REQBUFS");
    let mmaps: Vec<_> = (0..count)
        .map(|i| queue.map(i).expect("mmap").remove(0))
        .collect();
    for i in 0..count {
        queue.enqueue(i, &[Plane::mmap()], None).expect("QBUF");
    }
    queue.stream_on().expect("STREAMON");
    assert!(
        queue.wait(Some(Duration::from_secs(5))).expect("poll"),
        "no frame from vivid within 5 s"
    );
    let frame = queue.dequeue().expect("DQBUF").expect("a ready buffer");

    let fd = queue.export(frame.index, 0).expect("EXPBUF");
    let mut t = Tensor::<u8>::from_fd(fd, &[HEIGHT as usize, WIDTH as usize, 2], None)
        .expect("import the exported buffer");
    assert_eq!(t.memory(), TensorMemory::DmaBuf);
    t.set_format(PixelFormat::Yuyv).expect("YUYV");
    t.set_row_stride(pitch).expect("padded row stride");
    assert_eq!(t.effective_row_stride(), Some(pitch));

    let map = t.map().expect("strided map of the imported capture buffer");
    let pixels = map.as_slice();
    assert!(pixels.len() >= pitch * HEIGHT as usize);
    // SAFETY: the buffer is dequeued, so the driver no longer writes it.
    let reference = unsafe { mmaps[frame.index as usize].as_slice() };
    for y in 0..HEIGHT as usize {
        let row = y * pitch;
        assert_eq!(
            &pixels[row..row + NATURAL],
            &reference[row..row + NATURAL],
            "row {y} of the imported buffer differs from the driver's mapping"
        );
    }
    assert!(
        pixels[..NATURAL].iter().any(|&b| b != 0),
        "vivid wrote an empty first row"
    );
    drop(map);
    drop(t);
    queue.stream_off().expect("STREAMOFF");
}

#[test]
fn vivid_padded_capture_import_single_planar() {
    padded_capture_imports_and_maps_at_its_stride(BufType::VideoCapture);
}

#[test]
fn vivid_padded_capture_import_multi_planar() {
    padded_capture_imports_and_maps_at_its_stride(BufType::VideoCaptureMplane);
}
