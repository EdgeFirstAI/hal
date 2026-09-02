// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_tensor::{DetectBox, ProtoData, Segmentation};
use std::panic::AssertUnwindSafe;
use std::ptr::NonNull;
use std::thread::JoinHandle;
use tokio::sync::mpsc::{Sender, WeakSender};

use super::processor::GLProcessorST;
use super::shaders::check_gl_error;
use super::{EglDisplayKind, Int8InterpolationMode, TransferBackend};
use crate::{Crop, Error, Flip, ImageProcessorTrait, Rotation};
use edgefirst_tensor::TensorDyn;

#[allow(clippy::type_complexity)]
enum GLProcessorMessage {
    ImageConvert(
        SendablePtr<TensorDyn>,
        SendablePtr<TensorDyn>,
        Rotation,
        Flip,
        Crop,
        bool, // defer: skip the per-tile glFinish (batch); flush syncs once
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    /// Convert without the terminal CPU sync, replying with a native
    /// fence fd guarding the GPU work (the GL→NPU handoff). `Ok(None)` =
    /// converted with blocking semantics (no fence on this platform).
    ImageConvertFenced(
        SendablePtr<TensorDyn>,
        SendablePtr<TensorDyn>,
        Rotation,
        Flip,
        Crop,
        tokio::sync::oneshot::Sender<Result<Option<super::CompletionFence>, Error>>,
    ),
    /// Complete any deferred batch render with a single GPU sync.
    Flush(tokio::sync::oneshot::Sender<Result<(), Error>>),
    SetColors(
        Vec<[u8; 4]>,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    DrawDecodedMasks(
        SendablePtr<TensorDyn>,
        SendablePtr<DetectBox>,
        SendablePtr<Segmentation>,
        f32,                            // opacity
        Option<SendablePtr<TensorDyn>>, // background
        Option<[f32; 4]>,               // letterbox
        crate::ColorMode,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    DrawProtoMasks(
        SendablePtr<TensorDyn>,
        SendablePtr<DetectBox>,
        SendablePtr<ProtoData>,
        f32,                            // opacity
        Option<SendablePtr<TensorDyn>>, // background
        Option<[f32; 4]>,               // letterbox
        crate::ColorMode,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    SetInt8Interpolation(
        Int8InterpolationMode,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    /// Set the colorimetry/performance trade-off (`ColorimetryMode`).
    SetColorimetryMode(
        crate::ColorimetryMode,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    /// Snapshot the EGLImage cache counters (steady-state import assertions).
    EglCacheStats(tokio::sync::oneshot::Sender<Result<super::cache::GlCacheStats, Error>>),
    /// Snapshot the per-convert source-feed counters (zero-copy observability).
    ConvertStats(tokio::sync::oneshot::Sender<Result<super::cache::ConvertStats, Error>>),
    PboCreate(
        usize, // buffer size in bytes
        tokio::sync::oneshot::Sender<Result<u32, Error>>,
    ),
    PboMap(
        u32,   // buffer_id
        usize, // size
        tokio::sync::oneshot::Sender<Result<edgefirst_tensor::PboMapping, Error>>,
    ),
    PboUnmap(
        u32, // buffer_id
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    PboDelete(u32), // fire-and-forget, no reply
    CudaRegisterBuffer(u32, tokio::sync::oneshot::Sender<Option<usize>>),
    CudaMap(usize, tokio::sync::oneshot::Sender<Option<(usize, usize)>>),
    CudaUnmap(usize),      // fire-and-forget
    CudaUnregister(usize), // fire-and-forget
}

/// Compute the flat element count for a PBO image buffer of the given format.
///
/// NV12 and NV16 are semiplanar with non-trivial element counts; all other
/// formats use `width * height * channels`.
///
/// Returns `None` on `usize` overflow. A wrapped element count would size
/// the PBO too small and corrupt memory on readback, so callers must treat
/// `None` as an invalid-dimensions error (the same way they already reject
/// a zero count).
fn pbo_elem_count(
    width: usize,
    height: usize,
    format: edgefirst_tensor::PixelFormat,
) -> Option<usize> {
    // Element count == product of the canonical image shape (the PBO holds u8,
    // one byte per element). Delegating to `allocation_shape` keeps the combined
    // semi-planar height in lockstep with every other consumer (and is exact
    // for odd heights — `height*3/2` truncation under-sized odd-H NV12 by a
    // row, and the old NV24 arm fell through to `channels` = far too small).
    // `checked_mul` so a wrapped count can't silently undersize the PBO.
    format
        .allocation_shape(width, height)?
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Compute the tensor shape for a PBO image of the given format — the canonical
/// [`PixelFormat::allocation_shape`] (Planar `[C,H,W]`, SemiPlanar `[total_h, W]`
/// with the odd-height-exact combined-plane height, Packed `[H,W,C]`).
fn pbo_shape(width: usize, height: usize, format: edgefirst_tensor::PixelFormat) -> Vec<usize> {
    format
        .allocation_shape(width, height)
        .unwrap_or_else(|| vec![height, width, format.channels()])
}

/// Best-effort registration of a freshly-created PBO with CUDA GL interop.
///
/// When libcudart is present, asks the GL worker (where the GL context is
/// current) to call `cudaGraphicsGLRegisterBuffer` for `buffer_id`, then
/// attaches a [`CudaHandle`] so `cuda_map()` can later yield a linear,
/// 256-byte-aligned device pointer for TensorRT. On any failure (no CUDA,
/// channel closed, registration rejected) the handle is left unset and the
/// PBO is still usable as a normal CPU/GL buffer.
fn register_pbo_cuda<T>(
    tensor: &mut edgefirst_tensor::Tensor<T>,
    buffer_id: u32,
    size: usize,
    sender: &Sender<GLProcessorMessage>,
) where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
{
    if !edgefirst_tensor::is_cuda_available() {
        return;
    }
    let (tx, rx) = tokio::sync::oneshot::channel();
    if sender
        .blocking_send(GLProcessorMessage::CudaRegisterBuffer(buffer_id, tx))
        .is_ok()
    {
        if let Ok(Some(resource)) = rx.blocking_recv() {
            let ops = std::sync::Arc::new(GlCudaOps {
                sender: sender.downgrade(),
            });
            tensor.set_cuda_handle(edgefirst_tensor::CudaHandle::new_gl(
                resource as *mut std::ffi::c_void,
                size,
                ops,
            ));
        }
    }
}

/// Implements PboOps by sending commands to the GL thread.
///
/// Uses a `WeakSender` so that PBO images don't keep the GL thread's channel
/// alive. When the `GLProcessorThreaded` is dropped, its `Sender` is the last
/// strong reference — dropping it closes the channel and lets the GL thread
/// exit. PBO operations after that return `PboDisconnected`.
struct GlPboOps {
    sender: WeakSender<GLProcessorMessage>,
}

// SAFETY: GlPboOps sends all GL operations to the dedicated GL thread via a
// channel. `map_buffer` returns a CPU-visible pointer from `glMapBufferRange`
// that remains valid until `unmap_buffer` calls `glUnmapBuffer` on the GL thread.
// `delete_buffer` sends a fire-and-forget deletion command to the GL thread.
unsafe impl edgefirst_tensor::PboOps for GlPboOps {
    fn map_buffer(
        &self,
        buffer_id: u32,
        size: usize,
    ) -> edgefirst_tensor::Result<edgefirst_tensor::PboMapping> {
        let sender = self
            .sender
            .upgrade()
            .ok_or(edgefirst_tensor::Error::PboDisconnected)?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .blocking_send(GLProcessorMessage::PboMap(buffer_id, size, tx))
            .map_err(|_| edgefirst_tensor::Error::PboDisconnected)?;
        rx.blocking_recv()
            .map_err(|_| edgefirst_tensor::Error::PboDisconnected)?
            .map_err(|e| {
                edgefirst_tensor::Error::NotImplemented(format!("GL PBO map failed: {e:?}"))
            })
    }

    fn unmap_buffer(&self, buffer_id: u32) -> edgefirst_tensor::Result<()> {
        let sender = self
            .sender
            .upgrade()
            .ok_or(edgefirst_tensor::Error::PboDisconnected)?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .blocking_send(GLProcessorMessage::PboUnmap(buffer_id, tx))
            .map_err(|_| edgefirst_tensor::Error::PboDisconnected)?;
        rx.blocking_recv()
            .map_err(|_| edgefirst_tensor::Error::PboDisconnected)?
            .map_err(|e| {
                edgefirst_tensor::Error::NotImplemented(format!("GL PBO unmap failed: {e:?}"))
            })
    }

    fn delete_buffer(&self, buffer_id: u32) {
        if let Some(sender) = self.sender.upgrade() {
            let _ = sender.blocking_send(GLProcessorMessage::PboDelete(buffer_id));
        }
    }
}

/// Routes CUDA GL-interop map/unmap/unregister to the GL thread.
///
/// Mirrors [`GlPboOps`]: holds a [`WeakSender`] so a registered PBO doesn't
/// keep the GL thread's channel alive. The CUDA primitives
/// (`cudaGraphicsMapResources` etc.) require the GL context to be current,
/// so they must execute on the dedicated GL worker thread.
struct GlCudaOps {
    sender: WeakSender<GLProcessorMessage>,
}

impl edgefirst_tensor::CudaGlOps for GlCudaOps {
    fn map(&self, resource: *mut std::ffi::c_void) -> Option<(*mut std::ffi::c_void, usize)> {
        let sender = self.sender.upgrade()?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .blocking_send(GLProcessorMessage::CudaMap(resource as usize, tx))
            .ok()?;
        rx.blocking_recv()
            .ok()?
            .map(|(p, l)| (p as *mut std::ffi::c_void, l))
    }

    fn unmap(&self, resource: *mut std::ffi::c_void) {
        if let Some(sender) = self.sender.upgrade() {
            let _ = sender.blocking_send(GLProcessorMessage::CudaUnmap(resource as usize));
        }
    }

    fn unregister(&self, resource: *mut std::ffi::c_void) {
        if let Some(sender) = self.sender.upgrade() {
            let _ = sender.blocking_send(GLProcessorMessage::CudaUnregister(resource as usize));
        }
    }
}

/// OpenGL multi-threaded image converter. The actual conversion is done in a
/// separate rendering thread, as OpenGL contexts are not thread-safe. This can
/// be safely sent between threads. The `convert()` call sends the conversion
/// request to the rendering thread and waits for the result.
#[derive(Debug)]
pub struct GLProcessorThreaded {
    // This is only None when the converter is being dropped.
    handle: Option<JoinHandle<()>>,

    // This is only None when the converter is being dropped.
    sender: Option<Sender<GLProcessorMessage>>,
    /// Immutable capability surface (transfer backend, float render
    /// support, serialization policy), captured once from the worker at
    /// construction. See `PlatformCaps` in `platform/mod.rs`.
    caps: super::platform::PlatformCaps,
}

unsafe impl Send for GLProcessorThreaded {}
unsafe impl Sync for GLProcessorThreaded {}

struct SendablePtr<T: Send> {
    ptr: NonNull<T>,
    len: usize,
}

unsafe impl<T> Send for SendablePtr<T> where T: Send {}

/// Extract a human-readable message from a `catch_unwind` panic payload.
fn panic_message(info: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = info.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = info.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn reply_caught<T>(
    result: std::thread::Result<crate::Result<T>>,
    resp: tokio::sync::oneshot::Sender<crate::Result<T>>,
    poisoned: &mut bool,
    what: &str,
) {
    let _ = resp.send(match result {
        Ok(res) => res,
        Err(e) => {
            *poisoned = true;
            Err(crate::Error::Internal(format!(
                "GL thread panicked during {what}: {}",
                panic_message(e.as_ref()),
            )))
        }
    });
}

fn reject_poisoned_message(msg: GLProcessorMessage) {
    let poison_err =
        crate::Error::Internal("GL context is poisoned after a prior panic".to_string());
    match msg {
        GLProcessorMessage::ImageConvertFenced(.., resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::ImageConvert(.., resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::Flush(resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::DrawDecodedMasks(.., resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::DrawProtoMasks(.., resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::SetColors(_, resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::SetInt8Interpolation(_, resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::SetColorimetryMode(_, resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::EglCacheStats(resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::ConvertStats(resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::PboCreate(_, resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::PboMap(_, _, resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::PboUnmap(_, resp) => {
            let _ = resp.send(Err(poison_err));
        }
        GLProcessorMessage::PboDelete(_) => {}
        GLProcessorMessage::CudaRegisterBuffer(_, resp) => {
            let _ = resp.send(None);
        }
        GLProcessorMessage::CudaMap(_, resp) => {
            let _ = resp.send(None);
        }
        // GL context is poisoned/dead — the GL buffer is unusable, so unmapping/
        // unregistering the CUDA interop is moot (the resource is reclaimed at
        // process exit). Same accepted "GL is gone" no-op as PboDelete above.
        GLProcessorMessage::CudaUnmap(_) => {}
        GLProcessorMessage::CudaUnregister(_) => {}
    }
}

fn run_gl_worker(
    kind: Option<EglDisplayKind>,
    capacity: Option<usize>,
    mut recv: tokio::sync::mpsc::Receiver<GLProcessorMessage>,
    create_ctx_send: tokio::sync::oneshot::Sender<Result<super::platform::PlatformCaps, Error>>,
) {
    // Creation runs under the global lifecycle lock on Linux
    // (bring-up races on some embedded drivers); ANGLE serializes
    // display-level entry points internally, so macOS needs none
    // (validated by the A0 lifecycle-churn spike leg).
    #[cfg(target_os = "linux")]
    let init_result = {
        let _guard = super::context::GL_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        GLProcessorST::new(kind, capacity)
    };
    // macOS: ANGLE/Metal serializes display-level entry points
    // internally. Android: the system EGL is specification-conformant
    // thread-safe. Neither needs the Linux lifecycle lock.
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
    let init_result = GLProcessorST::new(kind, capacity);
    // Windows: ANGLE's D3D11 backend drives one immediate device context
    // (and one state manager) per display from every GL context and is not
    // safe to use from two threads at once: a processor bringing up its
    // context (shader compiles, texture allocation) while another converted
    // crashed with an access violation. Lifecycle and messages therefore
    // share NON_LINUX_GL_MUTEX there; per-message locking is the ANGLE Full
    // policy anyway.
    #[cfg(target_os = "windows")]
    let init_result = {
        let _guard = NON_LINUX_GL_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        GLProcessorST::new(kind, capacity)
    };
    let mut gl_converter = match init_result {
        Ok(gl) => gl,
        Err(e) => {
            let _ = create_ctx_send.send(Err(e));
            return;
        }
    };
    // Capability surface captured ONCE before the message loop —
    // immutable for the worker's life, never re-probed per message.
    let caps = gl_converter.platform_caps();
    let _ = create_ctx_send.send(Ok(caps));
    let mut poisoned = false;
    // Tracks which PBO buffer_ids currently have an outstanding GL
    // mapping, scoped to this one GL context/thread -- the ONE
    // place every PboOps call against this context's buffers
    // actually serializes, regardless of which `PboHandle`
    // (edgefirst-tensor) issued the `PboMap`/`PboUnmap` message.
    // `PboHandle::acquire_map`'s own per-handle `Mutex<MapState>`
    // enforces "at most one mapping" WITHIN one handle, including
    // read-sharing (several readers reuse ONE mapping without ever
    // re-sending `PboMap` here) -- but it has no visibility into a
    // SEPARATE `PboHandle` for the same real `buffer_id`, which
    // task 11's cross-cdylib PBO import made newly reachable: a
    // producer's own tensor and a reconstructed cross-package
    // consumer tensor for the same `buffer_id` are two independent
    // `PboHandle`s, each believing it is the only mapper.
    // `glMapBufferRange` on an already-mapped buffer is undefined
    // behaviour per the GL spec, so THIS set -- not the per-handle
    // mutex, which never sees another handle's calls -- is what
    // actually prevents that.
    let mut mapped_pbo_buffers: std::collections::HashSet<u32> = std::collections::HashSet::new();
    // Per-message serialization policy: Full where the platform
    // demands it (`caps.serialize_gl` — Vivante/galcore is not
    // thread-safe for concurrent GL across contexts), LifecycleOnly
    // everywhere else — multiple processors then run GL work in
    // parallel, each on its own thread + context. See the GL_MUTEX
    // doc comment in context.rs. EDGEFIRST_GL_SERIALIZE overrides
    // per-driver defaults (full = escape hatch for unknown-bad
    // drivers; lifecycle = force-parallel).
    let serialize_per_msg = match std::env::var("EDGEFIRST_GL_SERIALIZE").as_deref() {
        Ok("full") => true,
        Ok("lifecycle") => false,
        _ => caps.serialize_gl,
    };
    log::debug!(
        "GL serialization policy: {}",
        if serialize_per_msg {
            "Full (per-message global lock)"
        } else {
            "LifecycleOnly (parallel across processors)"
        }
    );
    while let Some(msg) = recv.blocking_recv() {
        // Full policy: one processor's message at a time, process-wide.
        // Linux locks GL_MUTEX (messages must also exclude the locked
        // lifecycle operations — on Vivante a convert racing a context
        // bring-up is part of the driver bug class). macOS has no
        // lifecycle lock (ANGLE serializes display entry points
        // internally), so message-vs-message exclusion suffices —
        // needed on virtualized GPUs, where concurrent GL across
        // contexts mis-renders (paravirtual Metal, macOS CI runners).
        #[cfg(target_os = "linux")]
        let _guard = serialize_per_msg.then(|| {
            super::context::GL_MUTEX
                .lock()
                .unwrap_or_else(|e| e.into_inner())
        });
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        let _guard =
            serialize_per_msg.then(|| NON_LINUX_GL_MUTEX.lock().unwrap_or_else(|e| e.into_inner()));

        // After a panic, the GL context is in an undefined state. Reject
        // all subsequent messages with an error rather than risking wrong
        // output or a GPU hang from corrupted GL state. This follows the
        // same pattern as std::sync::Mutex poisoning.
        if poisoned {
            reject_poisoned_message(msg);
            continue;
        }

        // Platform pre-message hook (ANGLE/D3D11 re-syncs this context's
        // state when another processor's context issued the last commands;
        // a no-op elsewhere). Runs under the lock taken above.
        gl_converter.begin_gpu_pass();
        handle_gl_message(
            msg,
            &mut gl_converter,
            &mut mapped_pbo_buffers,
            &mut poisoned,
        );
        // Per-pass platform texture attachments (macOS pbuffer
        // binds) are released once the message's GPU work has
        // synced; deferred batches keep theirs until Flush.
        gl_converter.end_gpu_pass_if_synced();
    }
    // Explicitly drop under the mutex so EGL teardown is serialized
    // (Linux, and Windows for the D3D11 reason above; ANGLE teardown is
    // display-internal on macOS).
    #[cfg(target_os = "linux")]
    let _guard = super::context::GL_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    #[cfg(target_os = "windows")]
    let _guard = NON_LINUX_GL_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    drop(gl_converter);
}

/// Per-message serialization lock for the platforms without a Linux-style
/// `GL_MUTEX` (macOS/iOS, Android, Windows) — taken per message under the
/// Full policy, and on Windows also around processor bring-up/teardown.
#[cfg(not(target_os = "linux"))]
static NON_LINUX_GL_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn handle_gl_message(
    msg: GLProcessorMessage,
    gl_converter: &mut GLProcessorST,
    mapped_pbo_buffers: &mut std::collections::HashSet<u32>,
    poisoned: &mut bool,
) {
    match msg {
        GLProcessorMessage::ImageConvert(src, dst, rotation, flip, crop, defer, resp) => {
            handle_image_convert(
                src,
                dst,
                (rotation, flip, crop, defer),
                resp,
                gl_converter,
                poisoned,
            );
        }
        GLProcessorMessage::ImageConvertFenced(src, dst, rotation, flip, crop, resp) => {
            handle_image_convert_fenced(
                src,
                dst,
                (rotation, flip, crop),
                resp,
                gl_converter,
                poisoned,
            );
        }
        GLProcessorMessage::Flush(resp) => {
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| gl_converter.flush()));
            reply_caught(result, resp, poisoned, "Flush");
        }
        GLProcessorMessage::DrawDecodedMasks(
            dst,
            det,
            seg,
            opacity,
            bg,
            letterbox,
            color_mode,
            resp,
        ) => {
            handle_draw_decoded_masks(
                dst,
                det,
                seg,
                (opacity, bg, letterbox, color_mode),
                resp,
                gl_converter,
                poisoned,
            );
        }
        GLProcessorMessage::DrawProtoMasks(
            dst,
            det,
            proto_data,
            opacity,
            bg,
            letterbox,
            color_mode,
            resp,
        ) => {
            handle_draw_proto_masks(
                dst,
                det,
                proto_data,
                (opacity, bg, letterbox, color_mode),
                resp,
                gl_converter,
                poisoned,
            );
        }
        GLProcessorMessage::SetColors(colors, resp) => {
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                gl_converter.set_class_colors(&colors)
            }));
            reply_caught(result, resp, poisoned, "SetColors");
        }
        GLProcessorMessage::SetColorimetryMode(mode, resp) => {
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                gl_converter.set_colorimetry_mode(mode);
                Ok(())
            }));
            reply_caught(result, resp, poisoned, "SetColorimetryMode");
        }
        GLProcessorMessage::SetInt8Interpolation(mode, resp) => {
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                gl_converter.set_int8_interpolation_mode(mode);
                Ok(())
            }));
            reply_caught(result, resp, poisoned, "SetInt8Interpolation");
        }
        GLProcessorMessage::EglCacheStats(resp) => {
            let result =
                std::panic::catch_unwind(AssertUnwindSafe(|| Ok(gl_converter.egl_cache_stats())));
            reply_caught(result, resp, poisoned, "EglCacheStats");
        }
        GLProcessorMessage::ConvertStats(resp) => {
            let result =
                std::panic::catch_unwind(AssertUnwindSafe(|| Ok(gl_converter.convert_stats())));
            reply_caught(result, resp, poisoned, "ConvertStats");
        }
        GLProcessorMessage::PboCreate(size, resp) => {
            handle_pbo_create(size, resp, poisoned);
        }
        GLProcessorMessage::PboMap(buffer_id, size, resp) => {
            handle_pbo_map(buffer_id, size, resp, mapped_pbo_buffers, poisoned);
        }
        GLProcessorMessage::PboUnmap(buffer_id, resp) => {
            handle_pbo_unmap(buffer_id, resp, mapped_pbo_buffers, poisoned);
        }
        GLProcessorMessage::PboDelete(buffer_id) => {
            handle_pbo_delete(buffer_id, mapped_pbo_buffers, poisoned);
        }
        GLProcessorMessage::CudaRegisterBuffer(buffer_id, resp) => {
            // CUDA GL-interop must run on the GL-context thread.
            let _ = resp.send(edgefirst_tensor::gl_register_buffer(buffer_id));
        }
        GLProcessorMessage::CudaMap(resource, resp) => {
            // Auto-flush: if a batched `convert_deferred` is still
            // owed its single GPU sync, complete it before CUDA maps
            // the buffer — otherwise the device could read a render
            // still in flight. This makes `cuda_map()` correct on a
            // batched output with no API change (the caller pays the
            // sync it would have owed anyway).
            gl_converter.flush_pending();
            let _ = resp.send(edgefirst_tensor::gl_map_resource(resource));
        }
        GLProcessorMessage::CudaUnmap(resource) => {
            edgefirst_tensor::gl_unmap_resource(resource);
        }
        GLProcessorMessage::CudaUnregister(resource) => {
            edgefirst_tensor::gl_unregister_resource(resource);
        }
    }
}

fn handle_image_convert(
    src: SendablePtr<TensorDyn>,
    mut dst: SendablePtr<TensorDyn>,
    geom: (Rotation, Flip, Crop, bool),
    resp: tokio::sync::oneshot::Sender<Result<(), Error>>,
    gl_converter: &mut GLProcessorST,
    poisoned: &mut bool,
) {
    let (rotation, flip, crop, defer) = geom;
    // SAFETY: This is safe because the convert() function waits for the resp to
    // be sent before dropping the borrow for src and dst
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let src = unsafe { src.ptr.as_ref() };
        let dst = unsafe { dst.ptr.as_mut() };
        // A deferred convert skips the per-tile glFinish and
        // leaves a single sync owed at Flush; an eager convert
        // finishes as usual.
        if defer {
            gl_converter.convert_deferred(src, dst, rotation, flip, crop)
        } else {
            gl_converter.convert(src, dst, rotation, flip, crop)
        }
    }));
    reply_caught(result, resp, poisoned, "ImageConvert");
}

fn handle_image_convert_fenced(
    src: SendablePtr<TensorDyn>,
    mut dst: SendablePtr<TensorDyn>,
    geom: (Rotation, Flip, Crop),
    resp: tokio::sync::oneshot::Sender<Result<Option<super::CompletionFence>, Error>>,
    gl_converter: &mut GLProcessorST,
    poisoned: &mut bool,
) {
    let (rotation, flip, crop) = geom;
    // SAFETY: as ImageConvert — convert_with_fence()
    // blocks on the reply before the borrows drop.
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let src = unsafe { src.ptr.as_ref() };
        let dst = unsafe { dst.ptr.as_mut() };
        gl_converter.convert_fenced(src, dst, rotation, flip, crop)
    }));
    reply_caught(result, resp, poisoned, "ImageConvertFenced");
}

fn handle_draw_decoded_masks(
    mut dst: SendablePtr<TensorDyn>,
    det: SendablePtr<DetectBox>,
    seg: SendablePtr<Segmentation>,
    overlay: (
        f32,
        Option<SendablePtr<TensorDyn>>,
        Option<[f32; 4]>,
        crate::ColorMode,
    ),
    resp: tokio::sync::oneshot::Sender<Result<(), Error>>,
    gl_converter: &mut GLProcessorST,
    poisoned: &mut bool,
) {
    let (opacity, bg, letterbox, color_mode) = overlay;
    // SAFETY: This is safe because the draw_decoded_masks() function waits for the
    // resp to be sent before dropping the borrow for dst, detect,
    // segmentation, and background
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let dst = unsafe { dst.ptr.as_mut() };
        let det = unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
        let seg = unsafe { std::slice::from_raw_parts(seg.ptr.as_ptr(), seg.len) };
        let bg_ref = bg.map(|p| unsafe { &*p.ptr.as_ptr() });
        gl_converter.draw_decoded_masks(
            dst,
            det,
            seg,
            crate::MaskOverlay {
                background: bg_ref,
                opacity,
                letterbox,
                color_mode,
            },
        )
    }));
    reply_caught(result, resp, poisoned, "DrawDecodedMasks");
}

fn handle_draw_proto_masks(
    mut dst: SendablePtr<TensorDyn>,
    det: SendablePtr<DetectBox>,
    proto_data: SendablePtr<ProtoData>,
    overlay: (
        f32,
        Option<SendablePtr<TensorDyn>>,
        Option<[f32; 4]>,
        crate::ColorMode,
    ),
    resp: tokio::sync::oneshot::Sender<Result<(), Error>>,
    gl_converter: &mut GLProcessorST,
    poisoned: &mut bool,
) {
    let (opacity, bg, letterbox, color_mode) = overlay;
    // SAFETY: Same safety invariant as DrawDecodedMasks — caller
    // blocks on resp before dropping borrows.
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let dst = unsafe { dst.ptr.as_mut() };
        let det = unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
        let bg_ref = bg.map(|p| unsafe { &*p.ptr.as_ptr() });
        let proto_data = unsafe { proto_data.ptr.as_ref() };
        gl_converter.draw_proto_masks(
            dst,
            det,
            proto_data,
            crate::MaskOverlay {
                background: bg_ref,
                opacity,
                letterbox,
                color_mode,
            },
        )
    }));
    reply_caught(result, resp, poisoned, "DrawProtoMasks");
}

fn handle_pbo_create(
    size: usize,
    resp: tokio::sync::oneshot::Sender<Result<u32, Error>>,
    poisoned: &mut bool,
) {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        let mut id: u32 = 0;
        edgefirst_gl::gl::GenBuffers(1, &mut id);
        edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, id);
        edgefirst_gl::gl::BufferData(
            edgefirst_gl::gl::PIXEL_PACK_BUFFER,
            size as isize,
            std::ptr::null(),
            edgefirst_gl::gl::STREAM_COPY,
        );
        edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, 0);
        match check_gl_error("PboCreate", 0) {
            Ok(()) => Ok(id),
            Err(e) => {
                edgefirst_gl::gl::DeleteBuffers(1, &id);
                Err(e)
            }
        }
    }));
    reply_caught(result, resp, poisoned, "PboCreate");
}

fn handle_pbo_map(
    buffer_id: u32,
    size: usize,
    resp: tokio::sync::oneshot::Sender<Result<edgefirst_tensor::PboMapping, Error>>,
    mapped_pbo_buffers: &mut std::collections::HashSet<u32>,
    poisoned: &mut bool,
) {
    if !mapped_pbo_buffers.insert(buffer_id) {
        // Already mapped by SOME PboHandle against this
        // context -- refuse rather than call
        // glMapBufferRange a second time (undefined
        // behaviour). See `mapped_pbo_buffers`'s own
        // doc comment above.
        let _ = resp.send(Err(crate::Error::OpenGl(format!(
            "PBO buffer {buffer_id} is already mapped in this GL context \
             (a producer and a reconstructed cross-package consumer may \
             both hold a live tensor over it)"
        ))));
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, buffer_id);
        let ptr = edgefirst_gl::gl::MapBufferRange(
            edgefirst_gl::gl::PIXEL_PACK_BUFFER,
            0,
            size as isize,
            edgefirst_gl::gl::MAP_READ_BIT | edgefirst_gl::gl::MAP_WRITE_BIT,
        );
        edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, 0);
        if ptr.is_null() {
            Err(crate::Error::OpenGl(
                "glMapBufferRange returned null".to_string(),
            ))
        } else {
            Ok(edgefirst_tensor::PboMapping {
                ptr: ptr as *mut u8,
                size,
            })
        }
    }));
    let mapped_ok = matches!(result, Ok(Ok(_)));
    if !mapped_ok {
        // The map did not actually take -- release the
        // reservation so a retry (or the OTHER handle
        // that lost this race) is not permanently
        // blocked by a mapping that never happened.
        mapped_pbo_buffers.remove(&buffer_id);
    }
    reply_caught(result, resp, poisoned, "PboMap");
}

fn handle_pbo_unmap(
    buffer_id: u32,
    resp: tokio::sync::oneshot::Sender<Result<(), Error>>,
    mapped_pbo_buffers: &mut std::collections::HashSet<u32>,
    poisoned: &mut bool,
) {
    // Always release the reservation, regardless of
    // whether the GL unmap itself reports success below
    // -- mirrors `PboHandle::release_map`'s own
    // "state returns to Unmapped either way" contract
    // (edgefirst-tensor), so a failed unmap cannot
    // permanently strand this buffer_id as "mapped".
    mapped_pbo_buffers.remove(&buffer_id);
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, buffer_id);
        let ok = edgefirst_gl::gl::UnmapBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER);
        edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, 0);
        if ok == edgefirst_gl::gl::FALSE {
            Err(Error::OpenGl(
                "PBO data was corrupted during mapping".into(),
            ))
        } else {
            check_gl_error("PboUnmap", 0)
        }
    }));
    reply_caught(result, resp, poisoned, "PboUnmap");
}

fn handle_pbo_delete(
    buffer_id: u32,
    mapped_pbo_buffers: &mut std::collections::HashSet<u32>,
    poisoned: &mut bool,
) {
    // Defensive: a live mapping should never reach here
    // (PboHandle::Drop unmaps before deleting), but a
    // stale reservation for a since-deleted (and
    // possibly buffer-id-recycled) buffer would
    // otherwise wrongly block every future map of a NEW
    // buffer that reuses the same id.
    mapped_pbo_buffers.remove(&buffer_id);
    if let Err(e) = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        edgefirst_gl::gl::DeleteBuffers(1, &buffer_id);
    })) {
        *poisoned = true;
        log::error!(
            "GL thread panicked during PboDelete: {}",
            panic_message(e.as_ref()),
        );
    }
}

impl GLProcessorThreaded {
    /// Creates a new OpenGL multi-threaded image converter whose import
    /// caches use the environment/default capacity.
    pub fn new(kind: Option<EglDisplayKind>) -> Result<Self, Error> {
        Self::with_cache_capacity(kind, None)
    }

    /// As [`new`](Self::new), with an explicit per-cache import capacity —
    /// [`ImageProcessorConfig::egl_cache_capacity`], which documents what the
    /// number bounds and takes precedence over the environment.
    ///
    /// [`ImageProcessorConfig::egl_cache_capacity`]: crate::ImageProcessorConfig::egl_cache_capacity
    pub fn with_cache_capacity(
        kind: Option<EglDisplayKind>,
        capacity: Option<usize>,
    ) -> Result<Self, Error> {
        let (send, recv) = tokio::sync::mpsc::channel::<GLProcessorMessage>(1);

        let (create_ctx_send, create_ctx_recv) = tokio::sync::oneshot::channel();

        let handle = std::thread::spawn(move || {
            run_gl_worker(kind, capacity, recv, create_ctx_send);
        });

        let caps = match create_ctx_recv.blocking_recv() {
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(Error::Internal(
                    "GL converter error messaging closed without update".to_string(),
                ));
            }
            Ok(Ok(caps)) => caps,
        };

        Ok(Self {
            handle: Some(handle),
            sender: Some(send),
            caps,
        })
    }
}

impl ImageProcessorTrait for GLProcessorThreaded {
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::ImageConvert(
                SendablePtr {
                    ptr: NonNull::from(src),
                    len: 1,
                },
                SendablePtr {
                    ptr: NonNull::from(dst),
                    len: 1,
                },
                rotation,
                flip,
                crop,
                false, // eager: finish per convert
                err_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    fn convert_deferred(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::ImageConvert(
                SendablePtr {
                    ptr: NonNull::from(src),
                    len: 1,
                },
                SendablePtr {
                    ptr: NonNull::from(dst),
                    len: 1,
                },
                rotation,
                flip,
                crop,
                true, // defer: skip glFinish; flush() syncs the batch once
                err_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    fn flush(&mut self) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::Flush(err_send))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    fn draw_decoded_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[crate::DetectBox],
        segmentation: &[crate::Segmentation],
        overlay: crate::MaskOverlay<'_>,
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::DrawDecodedMasks(
                SendablePtr {
                    ptr: NonNull::from(dst),
                    len: 1,
                },
                SendablePtr {
                    ptr: NonNull::new(detect.as_ptr() as *mut DetectBox).unwrap(),
                    len: detect.len(),
                },
                SendablePtr {
                    ptr: NonNull::new(segmentation.as_ptr() as *mut Segmentation).unwrap(),
                    len: segmentation.len(),
                },
                overlay.opacity,
                overlay.background.map(|bg| SendablePtr {
                    ptr: NonNull::from(bg).cast::<TensorDyn>(),
                    len: 1,
                }),
                overlay.letterbox,
                overlay.color_mode,
                err_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    fn draw_proto_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        overlay: crate::MaskOverlay<'_>,
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::DrawProtoMasks(
                SendablePtr {
                    ptr: NonNull::from(dst),
                    len: 1,
                },
                SendablePtr {
                    ptr: NonNull::new(detect.as_ptr() as *mut DetectBox).unwrap(),
                    len: detect.len(),
                },
                SendablePtr {
                    ptr: NonNull::from(proto_data).cast::<ProtoData>(),
                    len: 1,
                },
                overlay.opacity,
                overlay.background.map(|bg| SendablePtr {
                    ptr: NonNull::from(bg).cast::<TensorDyn>(),
                    len: 1,
                }),
                overlay.letterbox,
                overlay.color_mode,
                err_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> Result<(), crate::Error> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::SetColors(colors.to_vec(), err_send))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }
}

impl GLProcessorThreaded {
    /// Sets the colorimetry/performance trade-off (see
    /// [`crate::ColorimetryMode`]). The `EDGEFIRST_COLORIMETRY` environment
    /// variable takes precedence — when set, this call logs and keeps the
    /// env-selected mode.
    pub fn set_colorimetry_mode(&mut self, mode: crate::ColorimetryMode) -> Result<(), Error> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::SetColorimetryMode(mode, err_send))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    /// Sets the interpolation mode for int8 proto textures.
    pub fn set_int8_interpolation_mode(
        &mut self,
        mode: Int8InterpolationMode,
    ) -> Result<(), crate::Error> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::SetInt8Interpolation(mode, err_send))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    /// Snapshot the EGLImage cache counters (src, dst, NV R8) from the GL
    /// thread. Steady-state tests capture this after warmup and after an
    /// N-frame loop over a fixed buffer pool and assert
    /// [`total_misses`](super::cache::GlCacheStats::total_misses) stays flat —
    /// any increase means a convert re-imported a buffer it should have found
    /// cached (the cache-behavior equality gate for GL refactors).
    pub fn egl_cache_stats(&self) -> Result<super::cache::GlCacheStats, Error> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::EglCacheStats(send))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    /// Convert and return a native sync-fence fd that signals when the
    /// GPU work completes, instead of blocking the CPU on `glFinish` —
    /// the GL→NPU handoff (`EGL_ANDROID_native_fence_sync`).
    ///
    /// `Ok(None)` means the convert completed with the normal blocking
    /// semantics (platform/driver has no native fence, or a pure-CPU-
    /// readback path was taken) — the destination is already safe to
    /// read. `Ok(Some(fd))` means the caller (or the NPU runtime, via
    /// `ANeuralNetworksExecution_startComputeWithDependencies`) must wait
    /// on the fd before consuming the destination buffer.
    pub fn convert_with_fence(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<Option<super::CompletionFence>, Error> {
        use crate::ImageProcessorTrait as _;
        if !self.caps.native_fence_sync {
            // No fence on this display: the plain blocking convert IS the
            // fenced contract with `None` — skip the special round-trip.
            self.convert(src, dst, rotation, flip, crop)?;
            return Ok(None);
        }
        let (send, recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::ImageConvertFenced(
                SendablePtr {
                    ptr: NonNull::from(src),
                    len: 1,
                },
                SendablePtr {
                    ptr: NonNull::from(dst),
                    len: 1,
                },
                rotation,
                flip,
                crop,
                send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    /// Snapshot the per-convert source-feed counters from the GL thread.
    /// A pipeline that expects end-to-end zero-copy asserts
    /// [`src_uploads`](super::cache::ConvertStats::src_uploads) stays flat
    /// (every frame fed by import, never by a CPU map + upload).
    pub fn convert_stats(&self) -> Result<super::cache::ConvertStats, Error> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::ConvertStats(send))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    /// Create a PBO-backed [`Tensor<u8>`](edgefirst_tensor::Tensor) image on the GL thread.
    pub fn create_pbo_image(
        &self,
        width: usize,
        height: usize,
        format: edgefirst_tensor::PixelFormat,
    ) -> Result<edgefirst_tensor::Tensor<u8>, Error> {
        let sender = self
            .sender
            .as_ref()
            .ok_or(Error::OpenGl("GL processor is shutting down".to_string()))?;

        let size = pbo_elem_count(width, height, format)
            .filter(|&s| s != 0)
            .ok_or_else(|| Error::OpenGl("Invalid image dimensions".to_string()))?;

        // Allocate PBO on the GL thread
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .blocking_send(GLProcessorMessage::PboCreate(size, tx))
            .map_err(|_| Error::OpenGl("GL thread channel closed".to_string()))?;
        let buffer_id = rx
            .blocking_recv()
            .map_err(|_| Error::OpenGl("GL thread did not respond".to_string()))??;

        let ops: std::sync::Arc<dyn edgefirst_tensor::PboOps> = std::sync::Arc::new(GlPboOps {
            sender: sender.downgrade(),
        });

        let shape = pbo_shape(width, height, format);

        let pbo_tensor =
            edgefirst_tensor::PboTensor::<u8>::from_pbo(buffer_id, size, &shape, None, ops)
                .map_err(|e| Error::OpenGl(format!("PBO tensor creation failed: {e:?}")))?;
        let mut tensor = edgefirst_tensor::Tensor::from_pbo(pbo_tensor)
            .map_err(|e| Error::OpenGl(format!("PBO tensor wrap failed: {e:?}")))?;
        tensor
            .set_format(format)
            .map_err(|e| Error::OpenGl(format!("Failed to set format on PBO tensor: {e:?}")))?;
        // Register the PBO with CUDA (best-effort; no-op without libcudart) so
        // `cuda_map()` can yield a device pointer — matching the float PBO path
        // in `create_pbo_image_dtype`. Without this, u8 PBOs were never
        // CUDA-interop-capable, so the codec's nvJPEG backend (and any CUDA
        // consumer) could not use a `create_image`-allocated PBO destination.
        // The i8 transmute by the caller preserves the attached handle.
        register_pbo_cuda(&mut tensor, buffer_id, size, sender);
        Ok(tensor)
    }

    /// Create a PBO-backed [`TensorDyn`] image on the GL thread with the given dtype.
    ///
    /// Sizes the underlying GL buffer by `elems * dtype.size()` and wraps it in
    /// the appropriately-typed [`PboTensor`]. Supports `DType::U8`, `DType::F16`,
    /// and `DType::F32`; returns an error for other dtypes.
    pub(crate) fn create_pbo_image_dtype(
        &self,
        width: usize,
        height: usize,
        format: edgefirst_tensor::PixelFormat,
        dtype: edgefirst_tensor::DType,
    ) -> Result<TensorDyn, Error> {
        let sender = self
            .sender
            .as_ref()
            .ok_or(Error::OpenGl("GL processor is shutting down".to_string()))?;

        let elems = pbo_elem_count(width, height, format)
            .filter(|&e| e != 0)
            .ok_or_else(|| Error::OpenGl("Invalid image dimensions".to_string()))?;

        let size = elems
            .checked_mul(dtype.size())
            .ok_or_else(|| Error::OpenGl("PBO size overflow".to_string()))?;

        // Allocate PBO on the GL thread
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .blocking_send(GLProcessorMessage::PboCreate(size, tx))
            .map_err(|_| Error::OpenGl("GL thread channel closed".to_string()))?;
        let buffer_id = rx
            .blocking_recv()
            .map_err(|_| Error::OpenGl("GL thread did not respond".to_string()))??;

        let ops: std::sync::Arc<dyn edgefirst_tensor::PboOps> = std::sync::Arc::new(GlPboOps {
            sender: sender.downgrade(),
        });

        let shape = pbo_shape(width, height, format);

        let map_err = |e: edgefirst_tensor::Error| {
            Error::OpenGl(format!("PBO tensor creation failed: {e:?}"))
        };
        let set_err = |e: edgefirst_tensor::Error| {
            Error::OpenGl(format!("Failed to set format on PBO tensor: {e:?}"))
        };

        match dtype {
            edgefirst_tensor::DType::U8 => {
                let pbo =
                    edgefirst_tensor::PboTensor::<u8>::from_pbo(buffer_id, size, &shape, None, ops)
                        .map_err(map_err)?;
                let mut t = edgefirst_tensor::Tensor::from_pbo(pbo).map_err(map_err)?;
                t.set_format(format).map_err(set_err)?;
                register_pbo_cuda(&mut t, buffer_id, size, sender);
                Ok(TensorDyn::from(t))
            }
            edgefirst_tensor::DType::F16 => {
                let pbo = edgefirst_tensor::PboTensor::<edgefirst_tensor::f16>::from_pbo(
                    buffer_id, size, &shape, None, ops,
                )
                .map_err(map_err)?;
                let mut t = edgefirst_tensor::Tensor::from_pbo(pbo).map_err(map_err)?;
                t.set_format(format).map_err(set_err)?;
                register_pbo_cuda(&mut t, buffer_id, size, sender);
                Ok(TensorDyn::from(t))
            }
            edgefirst_tensor::DType::F32 => {
                let pbo = edgefirst_tensor::PboTensor::<f32>::from_pbo(
                    buffer_id, size, &shape, None, ops,
                )
                .map_err(map_err)?;
                let mut t = edgefirst_tensor::Tensor::from_pbo(pbo).map_err(map_err)?;
                t.set_format(format).map_err(set_err)?;
                register_pbo_cuda(&mut t, buffer_id, size, sender);
                Ok(TensorDyn::from(t))
            }
            other => Err(Error::OpenGl(format!("unsupported PBO dtype {other:?}"))),
        }
    }

    /// Returns the active transfer backend.
    pub(crate) fn transfer_backend(&self) -> TransferBackend {
        self.caps.transfer_backend
    }

    /// Report which float dtypes the GPU can render to.
    ///
    /// Values are probed once at construction time and adjusted for
    /// Vivante GC7000UL, whose float readback latency (170-320 ms) makes
    /// GL float destinations impractical; `ImageProcessor::convert()` falls
    /// back to CPU float output (normalized to `[0, 1]`) for these targets.
    pub(crate) fn supported_render_dtypes(&self) -> crate::RenderDtypeSupport {
        self.caps.render_dtypes
    }
}

impl Drop for GLProcessorThreaded {
    fn drop(&mut self) {
        drop(self.sender.take());
        let _ = self.handle.take().and_then(|h| h.join().ok());
    }
}

// `pbo_elem_count` / `pbo_shape` are pure (no GL), so they are unit-testable
// without a GPU. The overflow→None arm of `pbo_elem_count` guards against an
// undersized PBO allocation, so it is worth pinning explicitly.
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{pbo_elem_count, pbo_shape, GLProcessorThreaded, GlPboOps};
    use edgefirst_tensor::{PixelFormat, TensorTrait};

    #[test]
    fn elem_count_per_format() {
        // Packed RGBA: w*h*4.
        assert_eq!(pbo_elem_count(8, 4, PixelFormat::Rgba), Some(8 * 4 * 4));
        // Packed RGB: w*h*3.
        assert_eq!(pbo_elem_count(8, 4, PixelFormat::Rgb), Some(8 * 4 * 3));
        // NV12 semiplanar: w*h*3/2.
        assert_eq!(pbo_elem_count(8, 4, PixelFormat::Nv12), Some(8 * 4 * 3 / 2));
        // NV16 semiplanar: w*h*2.
        assert_eq!(pbo_elem_count(8, 4, PixelFormat::Nv16), Some(8 * 4 * 2));
    }

    #[test]
    fn elem_count_overflow_is_none() {
        // w*h already overflows usize → None (never a wrapped, undersized count).
        assert_eq!(pbo_elem_count(usize::MAX, 2, PixelFormat::Rgba), None);
        // w*h fits but *channels overflows → None.
        assert_eq!(pbo_elem_count(usize::MAX, 1, PixelFormat::Rgb), None);
    }

    #[test]
    fn shape_per_format() {
        // Planar: [channels, height, width].
        assert_eq!(pbo_shape(8, 4, PixelFormat::PlanarRgb), vec![3, 4, 8]);
        // SemiPlanar NV12: [height*3/2, width].
        assert_eq!(pbo_shape(8, 4, PixelFormat::Nv12), vec![4 * 3 / 2, 8]);
        // SemiPlanar NV16: [height*2, width].
        assert_eq!(pbo_shape(8, 4, PixelFormat::Nv16), vec![4 * 2, 8]);
        // Packed: [height, width, channels].
        assert_eq!(pbo_shape(8, 4, PixelFormat::Rgba), vec![4, 8, 4]);
    }

    /// Regression test for the cross-handle PBO map race task 11's review
    /// found: `PboHandle::acquire_map`'s per-handle `Mutex<MapState>`
    /// (`edgefirst-tensor`) has no visibility into a SEPARATE `PboHandle`
    /// for the same real `buffer_id` -- exactly what a cross-cdylib
    /// reconstructed consumer tensor now is, alongside the producer's own
    /// tensor, since task 11 wired a real `kind::PBO` `import_descriptor`
    /// arm. Simulates that with two independent `PboTensor`s built
    /// directly (rather than through the full descriptor/vtable round
    /// trip, which needs a second compiled `.so` to be meaningful) that
    /// both point at the SAME real GL buffer via a fresh [`GlPboOps`]
    /// wrapping the SAME sender -- exactly the relationship `import_descriptor`'s
    /// PBO arm produces between a producer's handle and a reconstructed
    /// one. Without `mapped_pbo_buffers`, the second map would reach
    /// `glMapBufferRange` on an already-mapped buffer (undefined
    /// behaviour); this asserts it is refused instead, and that the
    /// refusal is a real, releasable reservation rather than a permanently
    /// stuck `buffer_id`.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))] // PBO destinations: Linux + Windows
    fn concurrent_maps_from_two_independent_handles_on_one_buffer_do_not_corrupt_gl_state() {
        let gl = match GLProcessorThreaded::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                eprintln!(
                    "SKIPPED: concurrent_maps_from_two_independent_handles... - \
                     OpenGL not available: {e:?}"
                );
                return;
            }
        };
        let tensor_a = match gl.create_pbo_image(64, 64, PixelFormat::Rgba) {
            Ok(t) => t,
            Err(e) => {
                eprintln!(
                    "SKIPPED: concurrent_maps_from_two_independent_handles... - \
                     PBO not supported: {e:?}"
                );
                return;
            }
        };
        let pbo_a = tensor_a.as_pbo().expect("just created a PBO tensor");
        let buffer_id = pbo_a.buffer_id();
        let size = tensor_a.capacity_bytes();
        let shape = tensor_a.shape().to_vec();

        // A second, independent PboHandle for the IDENTICAL real buffer --
        // same relationship a producer's handle and a reconstructed
        // consumer's handle have to the same GL context.
        let ops_b: std::sync::Arc<dyn edgefirst_tensor::PboOps> = std::sync::Arc::new(GlPboOps {
            sender: gl.sender.as_ref().unwrap().downgrade(),
        });
        let tensor_b =
            edgefirst_tensor::PboTensor::<u8>::from_pbo(buffer_id, size, &shape, None, ops_b)
                .expect("from_pbo against a real, already-created buffer_id must succeed");

        // tensor_a maps first and holds the mapping open.
        let map_a = tensor_a.map().expect("first map must succeed");

        // tensor_b's own Mutex<MapState> starts at Unmapped and knows
        // nothing about tensor_a's outstanding mapping -- without the
        // GL-thread-level fix this reaches glMapBufferRange a second time.
        let map_b_result = tensor_b.map();
        // `glMapBufferRange` on an already-mapped buffer is UNDEFINED
        // BEHAVIOUR per the GL spec, not guaranteed to fail cleanly on
        // every driver -- so this asserts the SPECIFIC refusal
        // `mapped_pbo_buffers` produces (matching this file's own error
        // text), not merely "some error occurred", which a
        // driver-dependent null return could also produce and would make
        // this test pass even with the fix reverted.
        let Err(map_b_err) = map_b_result else {
            panic!(
                "a second, independent handle mapping the SAME real buffer while the \
                     first mapping is still open must be refused, not silently succeed"
            );
        };
        let msg = format!("{map_b_err:?}");
        assert!(
            msg.contains("already mapped"),
            "must be refused specifically by the GL-thread-level reservation check \
             (mapped_pbo_buffers), not some other error path: {msg}"
        );

        drop(map_a);

        // Once the first mapping releases, the SAME second handle must be
        // able to map successfully -- proves the refusal above was a real,
        // releasable reservation, not a permanently stuck buffer_id.
        let map_b_after = tensor_b.map();
        assert!(
            map_b_after.is_ok(),
            "after the first handle's mapping releases, the second handle must be able to \
             map the same buffer: {map_b_after:?}"
        );
    }
}
