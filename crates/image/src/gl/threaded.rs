// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_decoder::{DetectBox, ProtoData, Segmentation};
use std::ptr::NonNull;
use std::thread::JoinHandle;
use tokio::sync::mpsc::{Sender, WeakSender};

use super::processor::GLProcessorST;
use super::shaders::check_gl_error;
use super::{EglDisplayKind, Int8InterpolationMode, TransferBackend};
use crate::{
    CPUProcessor, Crop, Error, Flip, ImageProcessorTrait, MaskRegion, Rotation, TensorImage,
    TensorImageRef,
};

#[allow(clippy::type_complexity)]
enum GLProcessorMessage {
    ImageConvert(
        SendablePtr<TensorImage>,
        SendablePtr<TensorImage>,
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
        SendablePtr<TensorImage>,
        SendablePtr<DetectBox>,
        SendablePtr<Segmentation>,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    DrawMasksProto(
        SendablePtr<TensorImage>,
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
    has_bgra: bool,
}

unsafe impl Send for GLProcessorThreaded {}
unsafe impl Sync for GLProcessorThreaded {}

struct SendablePtr<T: Send> {
    ptr: NonNull<T>,
    len: usize,
}

unsafe impl<T> Send for SendablePtr<T> where T: Send {}

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
            let _ = create_ctx_send.send(Ok((
                gl_converter.gl_context.transfer_backend,
                gl_converter.has_bgra,
            )));
            while let Some(msg) = recv.blocking_recv() {
                match msg {
                    GLProcessorMessage::ImageConvert(src, mut dst, rotation, flip, crop, resp) => {
                        // SAFETY: This is safe because the convert() function waits for the resp to
                        // be sent before dropping the borrow for src and dst
                        let src = unsafe { src.ptr.as_ref() };
                        let dst = unsafe { dst.ptr.as_mut() };
                        let res = gl_converter.convert(src, dst, rotation, flip, crop);
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::DrawMasks(mut dst, det, seg, resp) => {
                        // SAFETY: This is safe because the draw_masks() function waits for the
                        // resp to be sent before dropping the borrow for dst, detect, and
                        // segmentation
                        let dst = unsafe { dst.ptr.as_mut() };
                        let det = unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                        let seg = unsafe { std::slice::from_raw_parts(seg.ptr.as_ptr(), seg.len) };
                        let res = gl_converter.draw_masks(dst, det, seg);
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::DrawMasksProto(mut dst, det, proto_data, resp) => {
                        // SAFETY: Same safety invariant as DrawMasks — caller
                        // blocks on resp before dropping borrows.
                        let dst = unsafe { dst.ptr.as_mut() };
                        let det = unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                        let res = gl_converter.draw_masks_proto(dst, det, &proto_data);
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::SetColors(colors, resp) => {
                        let res = gl_converter.set_class_colors(&colors);
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::SetInt8Interpolation(mode, resp) => {
                        gl_converter.set_int8_interpolation_mode(mode);
                        let _ = resp.send(Ok(()));
                    }
                    GLProcessorMessage::DecodeMasksAtlas(
                        det,
                        proto_data,
                        output_width,
                        output_height,
                        resp,
                    ) => {
                        let det = unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                        let res = gl_converter.decode_masks_atlas(
                            det,
                            &proto_data,
                            output_width,
                            output_height,
                        );
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::PboCreate(size, resp) => {
                        let result = unsafe {
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
                        };
                        let _ = resp.send(result);
                    }
                    GLProcessorMessage::PboMap(buffer_id, size, resp) => {
                        let result = unsafe {
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
                        };
                        let _ = resp.send(result);
                    }
                    GLProcessorMessage::PboUnmap(buffer_id, resp) => {
                        let result = unsafe {
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
                        };
                        let _ = resp.send(result);
                    }
                    GLProcessorMessage::PboDelete(buffer_id) => unsafe {
                        gls::gl::DeleteBuffers(1, &buffer_id);
                    },
                }
            }
        };

        // let handle = tokio::task::spawn(func());
        let handle = std::thread::spawn(func);

        let (transfer_backend, has_bgra) = match create_ctx_recv.blocking_recv() {
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
            has_bgra,
        })
    }
}

impl ImageProcessorTrait for GLProcessorThreaded {
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        crop.check_crop(src, dst)?;
        if !GLProcessorST::check_src_format_supported(self.transfer_backend, src) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {} source texture",
                src.fourcc().display()
            )));
        }

        if !GLProcessorST::check_dst_format_supported(self.transfer_backend, dst, self.has_bgra) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {} destination texture",
                dst.fourcc().display()
            )));
        }

        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .unwrap()
            .blocking_send(GLProcessorMessage::ImageConvert(
                SendablePtr {
                    ptr: src.into(),
                    len: 1,
                },
                SendablePtr {
                    ptr: dst.into(),
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

    fn convert_ref(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImageRef<'_>,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        // OpenGL doesn't support PLANAR_RGB output, delegate to CPU
        let mut cpu = CPUProcessor::new();
        cpu.convert_ref(src, dst, rotation, flip, crop)
    }

    fn draw_masks(
        &mut self,
        dst: &mut TensorImage,
        detect: &[crate::DetectBox],
        segmentation: &[crate::Segmentation],
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .unwrap()
            .blocking_send(GLProcessorMessage::DrawMasks(
                SendablePtr {
                    ptr: dst.into(),
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
        dst: &mut TensorImage,
        detect: &[DetectBox],
        proto_data: &ProtoData,
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .unwrap()
            .blocking_send(GLProcessorMessage::DrawMasksProto(
                SendablePtr {
                    ptr: NonNull::new(dst as *mut TensorImage).unwrap(),
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
            .unwrap()
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
            .unwrap()
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
            .unwrap()
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

    /// Create a PBO-backed TensorImage on the GL thread.
    pub fn create_pbo_image(
        &self,
        width: usize,
        height: usize,
        fourcc: four_char_code::FourCharCode,
    ) -> Result<crate::TensorImage, Error> {
        let sender = self
            .sender
            .as_ref()
            .ok_or(Error::OpenGl("GL processor is shutting down".to_string()))?;

        let channels = crate::fourcc_channels(fourcc)?;
        let size = width * height * channels;
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

        let shape = if crate::fourcc_planar(fourcc)? {
            vec![channels, height, width]
        } else {
            vec![height, width, channels]
        };

        let pbo_tensor =
            edgefirst_tensor::PboTensor::<u8>::from_pbo(buffer_id, size, &shape, None, ops)
                .map_err(|e| Error::OpenGl(format!("PBO tensor creation failed: {e:?}")))?;
        let tensor = edgefirst_tensor::Tensor::Pbo(pbo_tensor);
        crate::TensorImage::from_tensor(tensor, fourcc)
            .map_err(|e| Error::OpenGl(format!("Failed to wrap PBO tensor as image: {e:?}")))
    }

    /// Returns the active transfer backend.
    #[allow(dead_code)]
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
