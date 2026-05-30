// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_decoder::{DetectBox, ProtoData, Segmentation};
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
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
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
    let channels = format.channels();
    let wh = width.checked_mul(height)?;
    match format.layout() {
        edgefirst_tensor::PixelLayout::SemiPlanar => match format {
            // NV12 is 1.5 bytes/px: multiply by 3 (checked) before halving.
            edgefirst_tensor::PixelFormat::Nv12 => wh.checked_mul(3).map(|v| v / 2),
            edgefirst_tensor::PixelFormat::Nv16 => wh.checked_mul(2),
            _ => wh.checked_mul(channels),
        },
        edgefirst_tensor::PixelLayout::Packed | edgefirst_tensor::PixelLayout::Planar => {
            wh.checked_mul(channels)
        }
        _ => wh.checked_mul(channels),
    }
}

/// Compute the tensor shape for a PBO image of the given format.
///
/// Planar: `[channels, height, width]`.
/// SemiPlanar: `[total_h, width]` (NV12 total_h = height*3/2; NV16 total_h = height*2).
/// All others: `[height, width, channels]`.
fn pbo_shape(width: usize, height: usize, format: edgefirst_tensor::PixelFormat) -> Vec<usize> {
    let channels = format.channels();
    match format.layout() {
        edgefirst_tensor::PixelLayout::Planar => vec![channels, height, width],
        edgefirst_tensor::PixelLayout::SemiPlanar => {
            let total_h = match format {
                edgefirst_tensor::PixelFormat::Nv12 => height * 3 / 2,
                edgefirst_tensor::PixelFormat::Nv16 => height * 2,
                _ => height * 2,
            };
            vec![total_h, width]
        }
        _ => vec![height, width, channels],
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
    transfer_backend: TransferBackend,
    /// Float render dtype support, probed at construction time and
    /// adjusted for Vivante GC7000UL (whose float readback is 170-320 ms).
    render_dtype_support: crate::RenderDtypeSupport,
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

impl GLProcessorThreaded {
    /// Creates a new OpenGL multi-threaded image converter.
    pub fn new(kind: Option<EglDisplayKind>) -> Result<Self, Error> {
        let (send, mut recv) = tokio::sync::mpsc::channel::<GLProcessorMessage>(1);

        let (create_ctx_send, create_ctx_recv) = tokio::sync::oneshot::channel();

        let func = move || {
            let init_result = {
                let _guard = super::context::GL_MUTEX
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                GLProcessorST::new(kind)
            };
            let mut gl_converter = match init_result {
                Ok(gl) => gl,
                Err(e) => {
                    let _ = create_ctx_send.send(Err(e));
                    return;
                }
            };
            let _ = create_ctx_send.send(Ok((
                gl_converter.gl_context.transfer_backend,
                gl_converter.supported_render_dtypes(),
            )));
            let mut poisoned = false;
            while let Some(msg) = recv.blocking_recv() {
                // Serialize all GL operations across GLProcessorST instances.
                // See `GL_MUTEX` doc comment in context.rs for rationale.
                let _guard = super::context::GL_MUTEX
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());

                // After a panic, the GL context is in an undefined state. Reject
                // all subsequent messages with an error rather than risking wrong
                // output or a GPU hang from corrupted GL state. This follows the
                // same pattern as std::sync::Mutex poisoning.
                if poisoned {
                    let poison_err = crate::Error::Internal(
                        "GL context is poisoned after a prior panic".to_string(),
                    );
                    match msg {
                        GLProcessorMessage::ImageConvert(.., resp) => {
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
                    }
                    continue;
                }

                match msg {
                    GLProcessorMessage::ImageConvert(src, mut dst, rotation, flip, crop, resp) => {
                        // SAFETY: This is safe because the convert() function waits for the resp to
                        // be sent before dropping the borrow for src and dst
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                            let src = unsafe { src.ptr.as_ref() };
                            let dst = unsafe { dst.ptr.as_mut() };
                            gl_converter.convert(src, dst, rotation, flip, crop)
                        }));
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during ImageConvert: {}",
                                    panic_message(e.as_ref()),
                                )))
                            }
                        });
                    }
                    GLProcessorMessage::DrawDecodedMasks(
                        mut dst,
                        det,
                        seg,
                        opacity,
                        bg,
                        letterbox,
                        color_mode,
                        resp,
                    ) => {
                        // SAFETY: This is safe because the draw_decoded_masks() function waits for the
                        // resp to be sent before dropping the borrow for dst, detect,
                        // segmentation, and background
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                            let dst = unsafe { dst.ptr.as_mut() };
                            let det =
                                unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                            let seg =
                                unsafe { std::slice::from_raw_parts(seg.ptr.as_ptr(), seg.len) };
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
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during DrawDecodedMasks: {}",
                                    panic_message(e.as_ref()),
                                )))
                            }
                        });
                    }
                    GLProcessorMessage::DrawProtoMasks(
                        mut dst,
                        det,
                        proto_data,
                        opacity,
                        bg,
                        letterbox,
                        color_mode,
                        resp,
                    ) => {
                        // SAFETY: Same safety invariant as DrawDecodedMasks — caller
                        // blocks on resp before dropping borrows.
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                            let dst = unsafe { dst.ptr.as_mut() };
                            let det =
                                unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
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
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during DrawProtoMasks: {}",
                                    panic_message(e.as_ref()),
                                )))
                            }
                        });
                    }
                    GLProcessorMessage::SetColors(colors, resp) => {
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                            gl_converter.set_class_colors(&colors)
                        }));
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during SetColors: {}",
                                    panic_message(e.as_ref()),
                                )))
                            }
                        });
                    }
                    GLProcessorMessage::SetInt8Interpolation(mode, resp) => {
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                            gl_converter.set_int8_interpolation_mode(mode);
                            Ok(())
                        }));
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during SetInt8Interpolation: {}",
                                    panic_message(e.as_ref()),
                                )))
                            }
                        });
                    }
                    GLProcessorMessage::PboCreate(size, resp) => {
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
                            let mut id: u32 = 0;
                            gls::gl::GenBuffers(1, &mut id);
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, id);
                            gls::gl::BufferData(
                                gls::gl::PIXEL_PACK_BUFFER,
                                size as isize,
                                std::ptr::null(),
                                gls::gl::STREAM_COPY,
                            );
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                            match check_gl_error("PboCreate", 0) {
                                Ok(()) => Ok(id),
                                Err(e) => {
                                    gls::gl::DeleteBuffers(1, &id);
                                    Err(e)
                                }
                            }
                        }));
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during PboCreate: {}",
                                    panic_message(e.as_ref()),
                                )))
                            }
                        });
                    }
                    GLProcessorMessage::PboMap(buffer_id, size, resp) => {
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, buffer_id);
                            let ptr = gls::gl::MapBufferRange(
                                gls::gl::PIXEL_PACK_BUFFER,
                                0,
                                size as isize,
                                gls::gl::MAP_READ_BIT | gls::gl::MAP_WRITE_BIT,
                            );
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
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
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during PboMap: {}",
                                    panic_message(e.as_ref()),
                                )))
                            }
                        });
                    }
                    GLProcessorMessage::PboUnmap(buffer_id, resp) => {
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, buffer_id);
                            let ok = gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                            if ok == gls::gl::FALSE {
                                Err(Error::OpenGl(
                                    "PBO data was corrupted during mapping".into(),
                                ))
                            } else {
                                check_gl_error("PboUnmap", 0)
                            }
                        }));
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during PboUnmap: {}",
                                    panic_message(e.as_ref()),
                                )))
                            }
                        });
                    }
                    GLProcessorMessage::PboDelete(buffer_id) => {
                        if let Err(e) = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
                            gls::gl::DeleteBuffers(1, &buffer_id);
                        })) {
                            poisoned = true;
                            log::error!(
                                "GL thread panicked during PboDelete: {}",
                                panic_message(e.as_ref()),
                            );
                        }
                    }
                }
            }
            // Explicitly drop under the mutex so EGL teardown is serialized.
            let _guard = super::context::GL_MUTEX
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            drop(gl_converter);
        };

        // let handle = tokio::task::spawn(func());
        let handle = std::thread::spawn(func);

        let (transfer_backend, render_dtype_support) = match create_ctx_recv.blocking_recv() {
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(Error::Internal(
                    "GL converter error messaging closed without update".to_string(),
                ));
            }
            Ok(Ok((tb, rds))) => (tb, rds),
        };

        Ok(Self {
            handle: Some(handle),
            sender: Some(send),
            transfer_backend,
            render_dtype_support,
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
                err_send,
            ))
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

    /// Create a PBO-backed [`Tensor<u8>`] image on the GL thread.
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
        let mut tensor = edgefirst_tensor::Tensor::from_pbo(pbo_tensor);
        tensor
            .set_format(format)
            .map_err(|e| Error::OpenGl(format!("Failed to set format on PBO tensor: {e:?}")))?;
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
                let mut t = edgefirst_tensor::Tensor::from_pbo(pbo);
                t.set_format(format).map_err(set_err)?;
                Ok(TensorDyn::from(t))
            }
            edgefirst_tensor::DType::F16 => {
                let pbo = edgefirst_tensor::PboTensor::<edgefirst_tensor::f16>::from_pbo(
                    buffer_id, size, &shape, None, ops,
                )
                .map_err(map_err)?;
                let mut t = edgefirst_tensor::Tensor::from_pbo(pbo);
                t.set_format(format).map_err(set_err)?;
                Ok(TensorDyn::from(t))
            }
            edgefirst_tensor::DType::F32 => {
                let pbo = edgefirst_tensor::PboTensor::<f32>::from_pbo(
                    buffer_id, size, &shape, None, ops,
                )
                .map_err(map_err)?;
                let mut t = edgefirst_tensor::Tensor::from_pbo(pbo);
                t.set_format(format).map_err(set_err)?;
                Ok(TensorDyn::from(t))
            }
            other => Err(Error::OpenGl(format!("unsupported PBO dtype {other:?}"))),
        }
    }

    /// Returns the active transfer backend.
    pub(crate) fn transfer_backend(&self) -> TransferBackend {
        self.transfer_backend
    }

    /// Report which float dtypes the GPU can render to.
    ///
    /// Values are probed once at construction time and adjusted for
    /// Vivante GC7000UL, whose float readback latency (170-320 ms) makes
    /// GL float destinations impractical; `ImageProcessor::convert()` falls
    /// back to CPU float output (normalized to `[0, 1]`) for these targets.
    pub(crate) fn supported_render_dtypes(&self) -> crate::RenderDtypeSupport {
        self.render_dtype_support
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
    use super::{pbo_elem_count, pbo_shape};
    use edgefirst_tensor::PixelFormat;

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
}
