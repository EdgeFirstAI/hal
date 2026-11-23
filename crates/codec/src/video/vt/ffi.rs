// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Declarations of the Core Foundation, CoreMedia, CoreVideo, IOSurface and
//! VideoToolbox functions and constants the backend calls, and [`CfOwned`],
//! an owned Core Foundation reference.
//!
//! Every symbol declared for both macOS and iOS exists from macOS 11.0 and
//! iOS 11.0. Symbols that iOS gained later are declared for macOS only.

#![allow(non_upper_case_globals)]

use std::ffi::c_void;
use std::ptr::NonNull;

pub(crate) type CFTypeRef = *const c_void;
pub(crate) type CFStringRef = *const c_void;
pub(crate) type CFDictionaryRef = *const c_void;
pub(crate) type CFArrayRef = *const c_void;
pub(crate) type CFAllocatorRef = *const c_void;
pub(crate) type CFNumberRef = *const c_void;
pub(crate) type CFBooleanRef = *const c_void;
pub(crate) type CFIndex = isize;
pub(crate) type CFTypeID = usize;
pub(crate) type CFNumberType = CFIndex;
pub(crate) type CFStringEncoding = u32;
pub(crate) type Boolean = u8;
pub(crate) type OSStatus = i32;

pub(crate) type CVReturn = i32;
pub(crate) type CVBufferRef = *mut c_void;
pub(crate) type CVImageBufferRef = CVBufferRef;
pub(crate) type CVPixelBufferRef = CVBufferRef;
pub(crate) type CVPixelBufferPoolRef = *mut c_void;
pub(crate) type CVAttachmentMode = u32;
pub(crate) type CVPixelBufferLockFlags = u64;

pub(crate) type IOSurfaceRef = *mut c_void;

pub(crate) type CMFormatDescriptionRef = *const c_void;
pub(crate) type CMVideoFormatDescriptionRef = CMFormatDescriptionRef;
pub(crate) type CMSampleBufferRef = *mut c_void;
pub(crate) type CMBlockBufferRef = *mut c_void;
pub(crate) type CMBlockBufferFlags = u32;
pub(crate) type CMItemCount = CFIndex;
pub(crate) type CMVideoCodecType = u32;

pub(crate) type VTSessionRef = CFTypeRef;
pub(crate) type VTCompressionSessionRef = *mut c_void;
pub(crate) type VTDecompressionSessionRef = *mut c_void;
pub(crate) type VTEncodeInfoFlags = u32;
pub(crate) type VTDecodeInfoFlags = u32;
pub(crate) type VTDecodeFrameFlags = u32;

pub(crate) const kCFNumberSInt32Type: CFNumberType = 3;
pub(crate) const kCFNumberFloat64Type: CFNumberType = 6;
pub(crate) const kCFStringEncodingUTF8: CFStringEncoding = 0x0800_0100;
pub(crate) const kCVReturnSuccess: CVReturn = 0;
#[cfg(all(test, target_os = "macos"))]
pub(crate) const kCVAttachmentMode_ShouldPropagate: CVAttachmentMode = 1;
pub(crate) const kCVPixelBufferLock_ReadOnly: CVPixelBufferLockFlags = 1;
pub(crate) const kCMVideoCodecType_H264: CMVideoCodecType = u32::from_be_bytes(*b"avc1");
pub(crate) const kCMBlockBufferAssureMemoryNowFlag: CMBlockBufferFlags = 1;
pub(crate) const kCMTimeFlags_Valid: u32 = 1;
pub(crate) const kVTDecodeInfo_FrameDropped: VTDecodeInfoFlags = 1 << 1;
pub(crate) const kVTEncodeInfo_FrameDropped: VTEncodeInfoFlags = 1 << 1;
pub(crate) const kVTInvalidSessionErr: OSStatus = -12903;
#[cfg(test)]
pub(crate) const kVTPropertyNotSupportedErr: OSStatus = -12900;

/// The opaque callback tables `CFDictionaryCreate` and `CFArrayCreate` take.
#[repr(C)]
pub(crate) struct CFCallBacks {
    _opaque: [u8; 0],
}

/// A rational time, `value / timescale` seconds.
#[repr(C, packed(4))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CMTime {
    pub(crate) value: i64,
    pub(crate) timescale: i32,
    pub(crate) flags: u32,
    pub(crate) epoch: i64,
}

/// The timing of one sample in a `CMSampleBuffer`.
#[repr(C, packed(4))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CMSampleTimingInfo {
    pub(crate) duration: CMTime,
    pub(crate) presentation_time_stamp: CMTime,
    pub(crate) decode_time_stamp: CMTime,
}

pub(crate) type VTCompressionOutputCallback = Option<
    unsafe extern "C" fn(
        output_ref_con: *mut c_void,
        source_frame_ref_con: *mut c_void,
        status: OSStatus,
        info_flags: VTEncodeInfoFlags,
        sample_buffer: CMSampleBufferRef,
    ),
>;

pub(crate) type VTDecompressionOutputCallback = Option<
    unsafe extern "C" fn(
        output_ref_con: *mut c_void,
        source_frame_ref_con: *mut c_void,
        status: OSStatus,
        info_flags: VTDecodeInfoFlags,
        image_buffer: CVImageBufferRef,
        presentation_time_stamp: CMTime,
        presentation_duration: CMTime,
    ),
>;

/// The decoder output callback and the pointer passed to it.
#[repr(C, packed(4))]
pub(crate) struct VTDecompressionOutputCallbackRecord {
    pub(crate) callback: VTDecompressionOutputCallback,
    pub(crate) ref_con: *mut c_void,
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    pub(crate) static kCFBooleanTrue: CFBooleanRef;
    pub(crate) static kCFBooleanFalse: CFBooleanRef;
    pub(crate) static kCFTypeDictionaryKeyCallBacks: CFCallBacks;
    pub(crate) static kCFTypeDictionaryValueCallBacks: CFCallBacks;
    pub(crate) static kCFTypeArrayCallBacks: CFCallBacks;

    pub(crate) fn CFRetain(cf: CFTypeRef) -> CFTypeRef;
    pub(crate) fn CFRelease(cf: CFTypeRef);
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn CFGetRetainCount(cf: CFTypeRef) -> CFIndex;
    pub(crate) fn CFEqual(a: CFTypeRef, b: CFTypeRef) -> Boolean;
    pub(crate) fn CFGetTypeID(cf: CFTypeRef) -> CFTypeID;

    pub(crate) fn CFNumberCreate(
        allocator: CFAllocatorRef,
        number_type: CFNumberType,
        value: *const c_void,
    ) -> CFNumberRef;
    pub(crate) fn CFNumberGetValue(
        number: CFNumberRef,
        number_type: CFNumberType,
        value: *mut c_void,
    ) -> Boolean;
    pub(crate) fn CFNumberGetTypeID() -> CFTypeID;
    pub(crate) fn CFBooleanGetValue(boolean: CFBooleanRef) -> Boolean;
    #[cfg(target_os = "macos")]
    pub(crate) fn CFBooleanGetTypeID() -> CFTypeID;

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn CFStringCreateWithBytes(
        allocator: CFAllocatorRef,
        bytes: *const u8,
        len: CFIndex,
        encoding: CFStringEncoding,
        is_external_representation: Boolean,
    ) -> CFStringRef;
    pub(crate) fn CFStringGetCString(
        string: CFStringRef,
        buffer: *mut std::ffi::c_char,
        buffer_size: CFIndex,
        encoding: CFStringEncoding,
    ) -> Boolean;
    pub(crate) fn CFStringGetLength(string: CFStringRef) -> CFIndex;
    pub(crate) fn CFStringGetMaximumSizeForEncoding(
        length: CFIndex,
        encoding: CFStringEncoding,
    ) -> CFIndex;
    pub(crate) fn CFStringGetTypeID() -> CFTypeID;

    pub(crate) fn CFDictionaryCreate(
        allocator: CFAllocatorRef,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        count: CFIndex,
        key_callbacks: *const CFCallBacks,
        value_callbacks: *const CFCallBacks,
    ) -> CFDictionaryRef;
    pub(crate) fn CFDictionaryGetValue(dictionary: CFDictionaryRef, key: CFTypeRef) -> CFTypeRef;

    pub(crate) fn CFArrayCreate(
        allocator: CFAllocatorRef,
        values: *const CFTypeRef,
        count: CFIndex,
        callbacks: *const CFCallBacks,
    ) -> CFArrayRef;
    pub(crate) fn CFArrayGetCount(array: CFArrayRef) -> CFIndex;
    pub(crate) fn CFArrayGetValueAtIndex(array: CFArrayRef, index: CFIndex) -> CFTypeRef;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    pub(crate) static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
    pub(crate) static kCVPixelBufferWidthKey: CFStringRef;
    pub(crate) static kCVPixelBufferHeightKey: CFStringRef;
    pub(crate) static kCVPixelBufferIOSurfacePropertiesKey: CFStringRef;
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) static kCVPixelBufferBytesPerRowAlignmentKey: CFStringRef;

    pub(crate) static kCVImageBufferColorPrimariesKey: CFStringRef;
    pub(crate) static kCVImageBufferTransferFunctionKey: CFStringRef;
    pub(crate) static kCVImageBufferYCbCrMatrixKey: CFStringRef;
    pub(crate) static kCVImageBufferColorPrimaries_ITU_R_709_2: CFStringRef;
    pub(crate) static kCVImageBufferColorPrimaries_SMPTE_C: CFStringRef;
    pub(crate) static kCVImageBufferColorPrimaries_ITU_R_2020: CFStringRef;
    pub(crate) static kCVImageBufferTransferFunction_ITU_R_709_2: CFStringRef;
    pub(crate) static kCVImageBufferTransferFunction_sRGB: CFStringRef;
    pub(crate) static kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ: CFStringRef;
    pub(crate) static kCVImageBufferTransferFunction_ITU_R_2100_HLG: CFStringRef;
    pub(crate) static kCVImageBufferYCbCrMatrix_ITU_R_601_4: CFStringRef;
    pub(crate) static kCVImageBufferYCbCrMatrix_ITU_R_709_2: CFStringRef;
    pub(crate) static kCVImageBufferYCbCrMatrix_ITU_R_2020: CFStringRef;

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn CVBufferSetAttachment(
        buffer: CVBufferRef,
        key: CFStringRef,
        value: CFTypeRef,
        mode: CVAttachmentMode,
    );
    pub(crate) fn CVBufferGetAttachment(
        buffer: CVBufferRef,
        key: CFStringRef,
        mode: *mut CVAttachmentMode,
    ) -> CFTypeRef;

    pub(crate) fn CVPixelBufferCreate(
        allocator: CFAllocatorRef,
        width: usize,
        height: usize,
        pixel_format: u32,
        attributes: CFDictionaryRef,
        out: *mut CVPixelBufferRef,
    ) -> CVReturn;
    pub(crate) fn CVPixelBufferCreateWithIOSurface(
        allocator: CFAllocatorRef,
        surface: IOSurfaceRef,
        attributes: CFDictionaryRef,
        out: *mut CVPixelBufferRef,
    ) -> CVReturn;
    pub(crate) fn CVPixelBufferGetIOSurface(buffer: CVPixelBufferRef) -> IOSurfaceRef;
    pub(crate) fn CVPixelBufferGetPixelFormatType(buffer: CVPixelBufferRef) -> u32;
    pub(crate) fn CVPixelBufferGetWidth(buffer: CVPixelBufferRef) -> usize;
    pub(crate) fn CVPixelBufferGetHeight(buffer: CVPixelBufferRef) -> usize;
    pub(crate) fn CVPixelBufferGetPlaneCount(buffer: CVPixelBufferRef) -> usize;
    pub(crate) fn CVPixelBufferLockBaseAddress(
        buffer: CVPixelBufferRef,
        flags: CVPixelBufferLockFlags,
    ) -> CVReturn;
    pub(crate) fn CVPixelBufferUnlockBaseAddress(
        buffer: CVPixelBufferRef,
        flags: CVPixelBufferLockFlags,
    ) -> CVReturn;
    pub(crate) fn CVPixelBufferGetBaseAddress(buffer: CVPixelBufferRef) -> *mut c_void;
    pub(crate) fn CVPixelBufferGetBytesPerRow(buffer: CVPixelBufferRef) -> usize;
    pub(crate) fn CVPixelBufferGetBaseAddressOfPlane(
        buffer: CVPixelBufferRef,
        plane: usize,
    ) -> *mut c_void;
    pub(crate) fn CVPixelBufferGetBytesPerRowOfPlane(
        buffer: CVPixelBufferRef,
        plane: usize,
    ) -> usize;
    pub(crate) fn CVPixelBufferGetHeightOfPlane(buffer: CVPixelBufferRef, plane: usize) -> usize;

    pub(crate) fn CVPixelBufferPoolCreatePixelBuffer(
        allocator: CFAllocatorRef,
        pool: CVPixelBufferPoolRef,
        out: *mut CVPixelBufferRef,
    ) -> CVReturn;
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn CVPixelBufferPoolCreate(
        allocator: CFAllocatorRef,
        pool_attributes: CFDictionaryRef,
        pixel_buffer_attributes: CFDictionaryRef,
        out: *mut CVPixelBufferPoolRef,
    ) -> CVReturn;
}

#[cfg(target_os = "macos")]
#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    pub(crate) static kCVImageBufferTransferFunction_Linear: CFStringRef;
}

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    pub(crate) fn IOSurfaceGetPixelFormat(surface: IOSurfaceRef) -> u32;
    pub(crate) fn IOSurfaceGetID(surface: IOSurfaceRef) -> u32;
    pub(crate) fn IOSurfaceGetWidth(surface: IOSurfaceRef) -> usize;
    pub(crate) fn IOSurfaceGetHeight(surface: IOSurfaceRef) -> usize;
    pub(crate) fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    pub(crate) fn IOSurfaceGetPlaneCount(surface: IOSurfaceRef) -> usize;
    pub(crate) fn IOSurfaceGetBytesPerRow(surface: IOSurfaceRef) -> usize;
    pub(crate) fn IOSurfaceGetBytesPerRowOfPlane(surface: IOSurfaceRef, plane: usize) -> usize;
    pub(crate) fn IOSurfaceGetBaseAddressOfPlane(
        surface: IOSurfaceRef,
        plane: usize,
    ) -> *mut c_void;
    pub(crate) fn IOSurfaceIncrementUseCount(surface: IOSurfaceRef);
    pub(crate) fn IOSurfaceDecrementUseCount(surface: IOSurfaceRef);
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn IOSurfaceIsInUse(surface: IOSurfaceRef) -> Boolean;
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn IOSurfaceGetUseCount(surface: IOSurfaceRef) -> i32;
}

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    pub(crate) static kCMTimeInvalid: CMTime;
    pub(crate) static kCMSampleAttachmentKey_NotSync: CFStringRef;

    pub(crate) fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        allocator: CFAllocatorRef,
        parameter_set_count: usize,
        parameter_set_pointers: *const *const u8,
        parameter_set_sizes: *const usize,
        nal_unit_header_length: i32,
        out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;
    pub(crate) fn CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        description: CMFormatDescriptionRef,
        index: usize,
        parameter_set: *mut *const u8,
        parameter_set_size: *mut usize,
        parameter_set_count: *mut usize,
        nal_unit_header_length: *mut i32,
    ) -> OSStatus;

    pub(crate) fn CMBlockBufferCreateWithMemoryBlock(
        structure_allocator: CFAllocatorRef,
        memory_block: *mut c_void,
        block_length: usize,
        block_allocator: CFAllocatorRef,
        custom_block_source: *const c_void,
        offset_to_data: usize,
        data_length: usize,
        flags: CMBlockBufferFlags,
        out: *mut CMBlockBufferRef,
    ) -> OSStatus;
    pub(crate) fn CMBlockBufferReplaceDataBytes(
        source: *const c_void,
        destination: CMBlockBufferRef,
        offset_into_destination: usize,
        data_length: usize,
    ) -> OSStatus;
    pub(crate) fn CMBlockBufferGetDataLength(buffer: CMBlockBufferRef) -> usize;
    pub(crate) fn CMBlockBufferCopyDataBytes(
        source: CMBlockBufferRef,
        offset_to_data: usize,
        data_length: usize,
        destination: *mut c_void,
    ) -> OSStatus;

    pub(crate) fn CMSampleBufferCreateReady(
        allocator: CFAllocatorRef,
        data_buffer: CMBlockBufferRef,
        format_description: CMFormatDescriptionRef,
        num_samples: CMItemCount,
        num_sample_timing_entries: CMItemCount,
        sample_timing_array: *const CMSampleTimingInfo,
        num_sample_size_entries: CMItemCount,
        sample_size_array: *const usize,
        out: *mut CMSampleBufferRef,
    ) -> OSStatus;
    pub(crate) fn CMSampleBufferGetDataBuffer(sample: CMSampleBufferRef) -> CMBlockBufferRef;
    pub(crate) fn CMSampleBufferGetFormatDescription(
        sample: CMSampleBufferRef,
    ) -> CMFormatDescriptionRef;
    pub(crate) fn CMSampleBufferGetSampleAttachmentsArray(
        sample: CMSampleBufferRef,
        create_if_necessary: Boolean,
    ) -> CFArrayRef;
}

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    pub(crate) fn VTIsHardwareDecodeSupported(codec_type: CMVideoCodecType) -> Boolean;
}

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    pub(crate) static kVTCompressionPropertyKey_AllowFrameReordering: CFStringRef;
    pub(crate) static kVTCompressionPropertyKey_ProfileLevel: CFStringRef;
    pub(crate) static kVTCompressionPropertyKey_AverageBitRate: CFStringRef;
    pub(crate) static kVTCompressionPropertyKey_DataRateLimits: CFStringRef;
    pub(crate) static kVTCompressionPropertyKey_MaxKeyFrameInterval: CFStringRef;
    pub(crate) static kVTCompressionPropertyKey_ExpectedFrameRate: CFStringRef;
    pub(crate) static kVTCompressionPropertyKey_MaxFrameDelayCount: CFStringRef;
    pub(crate) static kVTCompressionPropertyKey_ColorPrimaries: CFStringRef;
    pub(crate) static kVTCompressionPropertyKey_TransferFunction: CFStringRef;
    pub(crate) static kVTCompressionPropertyKey_YCbCrMatrix: CFStringRef;
    pub(crate) static kVTCompressionPropertyKey_EncoderID: CFStringRef;
    pub(crate) static kVTEncodeFrameOptionKey_ForceKeyFrame: CFStringRef;
    pub(crate) static kVTVideoEncoderList_CodecType: CFStringRef;
    pub(crate) static kVTVideoEncoderList_EncoderID: CFStringRef;
    pub(crate) static kVTVideoEncoderList_DisplayName: CFStringRef;

    pub(crate) static kVTProfileLevel_H264_Baseline_1_3: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Baseline_3_0: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Baseline_3_1: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Baseline_3_2: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Baseline_4_0: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Baseline_4_1: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Baseline_4_2: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Baseline_5_0: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Baseline_5_1: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Baseline_5_2: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Baseline_AutoLevel: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Main_3_0: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Main_3_1: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Main_3_2: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Main_4_0: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Main_4_1: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Main_4_2: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Main_5_0: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Main_5_1: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Main_5_2: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_Main_AutoLevel: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_High_3_0: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_High_3_1: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_High_3_2: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_High_4_0: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_High_4_1: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_High_4_2: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_High_5_0: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_High_5_1: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_High_5_2: CFStringRef;
    pub(crate) static kVTProfileLevel_H264_High_AutoLevel: CFStringRef;

    pub(crate) fn VTCopyVideoEncoderList(
        options: CFDictionaryRef,
        out: *mut CFArrayRef,
    ) -> OSStatus;

    pub(crate) fn VTSessionSetProperty(
        session: VTSessionRef,
        key: CFStringRef,
        value: CFTypeRef,
    ) -> OSStatus;
    pub(crate) fn VTSessionCopyProperty(
        session: VTSessionRef,
        key: CFStringRef,
        allocator: CFAllocatorRef,
        value_out: *mut c_void,
    ) -> OSStatus;

    pub(crate) fn VTCompressionSessionCreate(
        allocator: CFAllocatorRef,
        width: i32,
        height: i32,
        codec_type: CMVideoCodecType,
        encoder_specification: CFDictionaryRef,
        source_image_buffer_attributes: CFDictionaryRef,
        compressed_data_allocator: CFAllocatorRef,
        output_callback: VTCompressionOutputCallback,
        output_ref_con: *mut c_void,
        out: *mut VTCompressionSessionRef,
    ) -> OSStatus;
    pub(crate) fn VTCompressionSessionPrepareToEncodeFrames(
        session: VTCompressionSessionRef,
    ) -> OSStatus;
    pub(crate) fn VTCompressionSessionGetPixelBufferPool(
        session: VTCompressionSessionRef,
    ) -> CVPixelBufferPoolRef;
    pub(crate) fn VTCompressionSessionEncodeFrame(
        session: VTCompressionSessionRef,
        image_buffer: CVImageBufferRef,
        presentation_time_stamp: CMTime,
        duration: CMTime,
        frame_properties: CFDictionaryRef,
        source_frame_ref_con: *mut c_void,
        info_flags_out: *mut VTEncodeInfoFlags,
    ) -> OSStatus;
    pub(crate) fn VTCompressionSessionCompleteFrames(
        session: VTCompressionSessionRef,
        complete_until: CMTime,
    ) -> OSStatus;
    pub(crate) fn VTCompressionSessionInvalidate(session: VTCompressionSessionRef);

    pub(crate) fn VTDecompressionSessionCreate(
        allocator: CFAllocatorRef,
        format_description: CMVideoFormatDescriptionRef,
        decoder_specification: CFDictionaryRef,
        destination_image_buffer_attributes: CFDictionaryRef,
        output_callback: *const VTDecompressionOutputCallbackRecord,
        out: *mut VTDecompressionSessionRef,
    ) -> OSStatus;
    pub(crate) fn VTDecompressionSessionDecodeFrame(
        session: VTDecompressionSessionRef,
        sample_buffer: CMSampleBufferRef,
        flags: VTDecodeFrameFlags,
        source_frame_ref_con: *mut c_void,
        info_flags_out: *mut VTDecodeInfoFlags,
    ) -> OSStatus;
    pub(crate) fn VTDecompressionSessionWaitForAsynchronousFrames(
        session: VTDecompressionSessionRef,
    ) -> OSStatus;
    pub(crate) fn VTDecompressionSessionCanAcceptFormatDescription(
        session: VTDecompressionSessionRef,
        format_description: CMFormatDescriptionRef,
    ) -> Boolean;
    pub(crate) fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);
}

#[cfg(target_os = "macos")]
#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    pub(crate) static kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder: CFStringRef;
    pub(crate) static kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder:
        CFStringRef;
    pub(crate) static kVTVideoEncoderList_IsHardwareAccelerated: CFStringRef;
}

/// An owned reference to a Core Foundation object, released on drop.
#[derive(Debug)]
pub(crate) struct CfOwned(NonNull<c_void>);

// SAFETY: `CFRetain` and `CFRelease` are thread-safe, and `CfOwned` only
// retains, releases and hands out the pointer; code that uses the object
// through the pointer follows that object's own threading rules.
unsafe impl Send for CfOwned {}
// SAFETY: as for `Send`; `&CfOwned` allows only `clone` and `as_ptr`.
unsafe impl Sync for CfOwned {}

impl CfOwned {
    /// Owns a new reference to `ptr`, taken with `CFRetain`, or `None` when
    /// `ptr` is null. Use it for objects a `Get` function returns.
    ///
    /// # Safety
    ///
    /// `ptr` is null or a live Core Foundation object.
    pub(crate) unsafe fn retained(ptr: CFTypeRef) -> Option<Self> {
        let ptr = NonNull::new(ptr.cast_mut())?;
        // SAFETY: the caller guarantees `ptr` is a live CF object.
        unsafe { CFRetain(ptr.as_ptr()) };
        Some(Self(ptr))
    }

    /// Owns the reference a `Create` or `Copy` function returned, or `None`
    /// when it returned null.
    ///
    /// # Safety
    ///
    /// `ptr` is null or a Core Foundation object whose +1 reference the
    /// caller hands over.
    pub(crate) unsafe fn from_create(ptr: CFTypeRef) -> Option<Self> {
        NonNull::new(ptr.cast_mut()).map(Self)
    }

    /// The object, valid while `self` lives.
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

impl Clone for CfOwned {
    fn clone(&self) -> Self {
        // SAFETY: `self` holds a reference, so the object is live.
        unsafe { CFRetain(self.0.as_ptr()) };
        Self(self.0)
    }
}

impl Drop for CfOwned {
    fn drop(&mut self) {
        // SAFETY: `self` owns one reference, released once here.
        unsafe { CFRelease(self.0.as_ptr()) };
    }
}

/// A `CFNumber` holding `value` as a 32-bit integer, or `None` when it cannot
/// be allocated.
pub(crate) fn cf_number_i32(value: i32) -> Option<CfOwned> {
    // SAFETY: `value` outlives the call, and `CFNumberCreate` returns +1.
    unsafe {
        CfOwned::from_create(CFNumberCreate(
            std::ptr::null(),
            kCFNumberSInt32Type,
            (&value as *const i32).cast(),
        ))
    }
}

/// A `CFNumber` holding `value` as a 64-bit float, or `None` when it cannot be
/// allocated.
pub(crate) fn cf_number_f64(value: f64) -> Option<CfOwned> {
    // SAFETY: `value` outlives the call, and `CFNumberCreate` returns +1.
    unsafe {
        CfOwned::from_create(CFNumberCreate(
            std::ptr::null(),
            kCFNumberFloat64Type,
            (&value as *const f64).cast(),
        ))
    }
}

/// `kCFBooleanTrue` or `kCFBooleanFalse`, which live for the whole process.
pub(crate) fn cf_bool(value: bool) -> CFBooleanRef {
    // SAFETY: reads immutable constants Core Foundation defines.
    unsafe {
        if value {
            kCFBooleanTrue
        } else {
            kCFBooleanFalse
        }
    }
}

/// A `CFString` of `s`, or `None` when it cannot be allocated.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn cf_string(s: &str) -> Option<CfOwned> {
    let len = CFIndex::try_from(s.len()).ok()?;
    // SAFETY: `s` is `len` bytes of UTF-8 that outlive the call, and
    // `CFStringCreateWithBytes` returns +1.
    unsafe {
        CfOwned::from_create(CFStringCreateWithBytes(
            std::ptr::null(),
            s.as_ptr(),
            len,
            kCFStringEncodingUTF8,
            0,
        ))
    }
}

/// An immutable `CFArray` of `values`, which it retains, or `None` when it
/// cannot be allocated.
///
/// # Safety
///
/// Every value is a live Core Foundation object.
pub(crate) unsafe fn cf_array(values: &[CFTypeRef]) -> Option<CfOwned> {
    let count = CFIndex::try_from(values.len()).ok()?;
    // SAFETY: `values` holds `count` live CF objects (the caller's
    // guarantee), the callback table is Core Foundation's own, and
    // `CFArrayCreate` returns +1.
    unsafe {
        CfOwned::from_create(CFArrayCreate(
            std::ptr::null(),
            values.as_ptr(),
            count,
            &kCFTypeArrayCallBacks,
        ))
    }
}

/// The text of the `CFString` `s`, or `None` when `s` is null or not a
/// string.
///
/// # Safety
///
/// `s` is null or a live Core Foundation object.
pub(crate) unsafe fn string_of(s: CFTypeRef) -> Option<String> {
    // SAFETY: `s` is a live CF object (the caller's guarantee).
    if s.is_null() || unsafe { CFGetTypeID(s) != CFStringGetTypeID() } {
        return None;
    }
    // SAFETY: `s` is a live `CFString`.
    let size =
        unsafe { CFStringGetMaximumSizeForEncoding(CFStringGetLength(s), kCFStringEncodingUTF8) }
            .checked_add(1)?;
    let mut text = vec![0u8; usize::try_from(size).ok()?];
    // SAFETY: `text` is `size` writable bytes.
    if unsafe { CFStringGetCString(s, text.as_mut_ptr().cast(), size, kCFStringEncodingUTF8) } == 0
    {
        return None;
    }
    let text = std::ffi::CStr::from_bytes_until_nul(&text).ok()?;
    text.to_str().ok().map(str::to_owned)
}

/// An immutable `CFDictionary` of `entries`, which it retains, or `None` when
/// it cannot be allocated.
///
/// # Safety
///
/// Every key and value is a live Core Foundation object.
pub(crate) unsafe fn cf_dictionary(entries: &[(CFTypeRef, CFTypeRef)]) -> Option<CfOwned> {
    let keys: Vec<CFTypeRef> = entries.iter().map(|&(k, _)| k).collect();
    let values: Vec<CFTypeRef> = entries.iter().map(|&(_, v)| v).collect();
    let count = CFIndex::try_from(entries.len()).ok()?;
    // SAFETY: `keys` and `values` hold `count` live CF objects (the caller's
    // guarantee), the callback tables are Core Foundation's own, and
    // `CFDictionaryCreate` returns +1.
    unsafe {
        CfOwned::from_create(CFDictionaryCreate(
            std::ptr::null(),
            keys.as_ptr(),
            values.as_ptr(),
            count,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks,
        ))
    }
}
