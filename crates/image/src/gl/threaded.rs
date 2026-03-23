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
use crate::{Crop, Error, Flip, ImageProcessorTrait, MaskRegion, Rotation};
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
    DrawMasks(
        SendablePtr<TensorDyn>,
        SendablePtr<DetectBox>,
        SendablePtr<Segmentation>,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    DrawMasksProto(
        SendablePtr<TensorDyn>,
        SendablePtr<DetectBox>,
        Box<ProtoData>,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    SetInt8Interpolation(
        Int8InterpolationMode,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    DecodeMasksAtlas(
        SendablePtr<DetectBox>,
        Box<ProtoData>,
        usize, // output_width
        usize, // output_height
        tokio::sync::oneshot::Sender<Result<(Vec<u8>, Vec<MaskRegion>), Error>>,
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
}

unsafe impl Send for GLProcessorThreaded {}
unsafe impl Sync for GLProcessorThreaded {}

struct SendablePtr<T: Send> {
    ptr: NonNull<T>,
    len: usize,
}

unsafe impl<T> Send for SendablePtr<T> where T: Send {}

/// Extract a human-readable message from a `catch_unwind` panic payload.
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
            let mut gl_converter = match GLProcessorST::new(kind) {
                Ok(gl) => gl,
                Err(e) => {
                    let _ = create_ctx_send.send(Err(e));
                    return;
                }
            };
            let _ = create_ctx_send.send(Ok(gl_converter.gl_context.transfer_backend));
            let mut poisoned = false;
            while let Some(msg) = recv.blocking_recv() {
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
                        GLProcessorMessage::DrawMasks(.., resp) => {
                            let _ = resp.send(Err(poison_err));
                        }
                        GLProcessorMessage::DrawMasksProto(.., resp) => {
                            let _ = resp.send(Err(poison_err));
                        }
                        GLProcessorMessage::SetColors(_, resp) => {
                            let _ = resp.send(Err(poison_err));
                        }
                        GLProcessorMessage::SetInt8Interpolation(_, resp) => {
                            let _ = resp.send(Err(poison_err));
                        }
                        GLProcessorMessage::DecodeMasksAtlas(.., resp) => {
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
                    GLProcessorMessage::DrawMasks(mut dst, det, seg, resp) => {
                        // SAFETY: This is safe because the draw_masks() function waits for the
                        // resp to be sent before dropping the borrow for dst, detect, and
                        // segmentation
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                            let dst = unsafe { dst.ptr.as_mut() };
                            let det =
                                unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                            let seg =
                                unsafe { std::slice::from_raw_parts(seg.ptr.as_ptr(), seg.len) };
                            gl_converter.draw_masks(dst, det, seg)
                        }));
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during DrawMasks: {}",
                                    panic_message(e.as_ref()),
                                )))
                            }
                        });
                    }
                    GLProcessorMessage::DrawMasksProto(mut dst, det, proto_data, resp) => {
                        // SAFETY: Same safety invariant as DrawMasks — caller
                        // blocks on resp before dropping borrows.
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                            let dst = unsafe { dst.ptr.as_mut() };
                            let det =
                                unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                            gl_converter.draw_masks_proto(dst, det, &proto_data)
                        }));
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during DrawMasksProto: {}",
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
                    GLProcessorMessage::DecodeMasksAtlas(
                        det,
                        proto_data,
                        output_width,
                        output_height,
                        resp,
                    ) => {
                        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                            let det =
                                unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                            gl_converter.decode_masks_atlas(
                                det,
                                &proto_data,
                                output_width,
                                output_height,
                            )
                        }));
                        let _ = resp.send(match result {
                            Ok(res) => res,
                            Err(e) => {
                                poisoned = true;
                                Err(crate::Error::Internal(format!(
                                    "GL thread panicked during DecodeMasksAtlas: {}",
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
        };

        // let handle = tokio::task::spawn(func());
        let handle = std::thread::spawn(func);

        let transfer_backend = match create_ctx_recv.blocking_recv() {
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(Error::Internal(
                    "GL converter error messaging closed without update".to_string(),
                ));
            }
            Ok(Ok(tb)) => tb,
        };

        Ok(Self {
            handle: Some(handle),
            sender: Some(send),
            transfer_backend,
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
            .ok_or(Error::Internal("GL processor is shutting down".to_string()))?
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

    fn draw_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[crate::DetectBox],
        segmentation: &[crate::Segmentation],
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or(Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::DrawMasks(
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
                err_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    fn draw_masks_proto(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        proto_data: &ProtoData,
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or(Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::DrawMasksProto(
                SendablePtr {
                    ptr: NonNull::from(dst),
                    len: 1,
                },
                SendablePtr {
                    ptr: NonNull::new(detect.as_ptr() as *mut DetectBox).unwrap(),
                    len: detect.len(),
                },
                Box::new(proto_data.clone()),
                err_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    fn decode_masks_atlas(
        &mut self,
        detect: &[DetectBox],
        proto_data: ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> crate::Result<(Vec<u8>, Vec<MaskRegion>)> {
        GLProcessorThreaded::decode_masks_atlas(
            self,
            detect,
            proto_data,
            output_width,
            output_height,
        )
    }

    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> Result<(), crate::Error> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or(Error::Internal("GL processor is shutting down".to_string()))?
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
            .ok_or(Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::SetInt8Interpolation(mode, err_send))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    /// Decode all detection masks into a compact atlas via the GL thread.
    ///
    /// Returns `(atlas_pixels, regions)` where `atlas_pixels` is a contiguous
    /// `Vec<u8>` of shape `[atlas_h, output_width]` (compact, bbox-sized strips)
    /// and `regions` describes each detection's location within the atlas.
    pub fn decode_masks_atlas(
        &mut self,
        detect: &[DetectBox],
        proto_data: ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> Result<(Vec<u8>, Vec<MaskRegion>), crate::Error> {
        let (resp_send, resp_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .ok_or(Error::Internal("GL processor is shutting down".to_string()))?
            .blocking_send(GLProcessorMessage::DecodeMasksAtlas(
                SendablePtr {
                    ptr: NonNull::new(detect.as_ptr() as *mut DetectBox).unwrap(),
                    len: detect.len(),
                },
                Box::new(proto_data),
                output_width,
                output_height,
                resp_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        resp_recv.blocking_recv().map_err(|_| {
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

        let channels = format.channels();
        let size = match format.layout() {
            edgefirst_tensor::PixelLayout::SemiPlanar => {
                // NV12: W*H*3/2, NV16: W*H*2
                match format {
                    edgefirst_tensor::PixelFormat::Nv12 => width * height * 3 / 2,
                    edgefirst_tensor::PixelFormat::Nv16 => width * height * 2,
                    _ => width * height * channels,
                }
            }
            edgefirst_tensor::PixelLayout::Packed | edgefirst_tensor::PixelLayout::Planar => {
                width * height * channels
            }
            _ => width * height * channels,
        };
        if size == 0 {
            return Err(Error::OpenGl("Invalid image dimensions".to_string()));
        }

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

        let shape = match format.layout() {
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
        };

        let pbo_tensor =
            edgefirst_tensor::PboTensor::<u8>::from_pbo(buffer_id, size, &shape, None, ops)
                .map_err(|e| Error::OpenGl(format!("PBO tensor creation failed: {e:?}")))?;
        let mut tensor = edgefirst_tensor::Tensor::from_pbo(pbo_tensor);
        tensor
            .set_format(format)
            .map_err(|e| Error::OpenGl(format!("Failed to set format on PBO tensor: {e:?}")))?;
        Ok(tensor)
    }

    /// Returns the active transfer backend.
    pub(crate) fn transfer_backend(&self) -> TransferBackend {
        self.transfer_backend
    }
}

impl Drop for GLProcessorThreaded {
    fn drop(&mut self) {
        drop(self.sender.take());
        let _ = self.handle.take().and_then(|h| h.join().ok());
    }
}
