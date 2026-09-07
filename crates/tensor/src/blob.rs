// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The tensor blob: a size-prefixed, append-only serialization of a tensor.
//!
//! # Why a blob and not the descriptor
//!
//! [`TensorDesc`](crate::TensorDesc) is a fixed C struct with one handle and no
//! plane table. The blob carries every plane explicitly, in either of two
//! transport modes, and survives being appended to.
//!
//! # Layout
//!
//! **Region order is permanent from first publication.** The format is
//! size-prefixed and append-only: a new field may only be appended, never
//! inserted, and `required_mask` covers the one case appending cannot — an
//! addition that changes how existing fields are *interpreted*, where an
//! unaware consumer must refuse rather than proceed.
//!
//! The fixed header is 64 bytes, laid out widest-first so every scalar is
//! naturally aligned and no reader performs an unaligned access:
//!
//! | Offset | Type | Field |
//! |---|---|---|
//! | 0 | `u64` | `size` — total blob bytes |
//! | 8 | `u64` | `required_mask` |
//! | 16 | `u64` | `planes_bytes` |
//! | 24 | `u32` | `storage_kind` |
//! | 28 | `u32` | `pid` |
//! | 32 | `i32` | `fence_fd` |
//! | 36 | `u32` | `dtype` |
//! | 40 | `i32` | `quant_axis` |
//! | 44 | `u32` | `ndim` |
//! | 48 | `u32` | `plane_count` |
//! | 52 | `u32` | `quant_scales_len` |
//! | 56 | `u32` | `quant_zero_points_len` |
//! | 60 | `u32` | `strings_bytes` |
//! | 64 | `u32` | `strides_len` — `ndim` entries, or 0 for densely packed |
//! | 68 | `u32` | *reserved, must be zero* — keeps the tail 8-aligned |
//!
//! `strides_len` is carried rather than assumed because empty strides are a
//! distinct, meaningful value: `Tensor.msg` defines an empty array as "densely
//! packed C-order". Always writing `ndim` strides would be simpler but would
//! turn a message with empty strides into a blob with explicit ones, so the
//! reference-mode round-trip would no longer be the identity.
//!
//! Because every variable region's length lives in the fixed header, each
//! region's offset is a constant-time arithmetic derivation. This is the one
//! thing CDR cannot do — it has no offset table, so a message's variable
//! fields must be walked sequentially — and it is why the blob is worth having
//! alongside the message rather than simply reusing it.
//!
//! Every integer is little-endian on every host, read and written explicitly.
//! The blob is never `transmute`d from a struct: that would bake in host byte
//! order and padding, and this format crosses hosts.

/// Bytes in the fixed header. See the module docs for the field table.
pub const HEADER_LEN: usize = 72;

/// Bytes in a plane record's scalar block, before its variable payloads.
///
/// Matches the schemas spec's independently-stated `TensorPlane` minimum: a
/// 48-byte scalar block (six 8-byte fields) plus two 4-byte counts.
pub const PLANE_RECORD_LEN: usize = 56;

/// `required_mask` bits this build understands. Zero: no interpretation-changing
/// field has been added yet, so *any* bit set means a producer needs something
/// from us that we cannot provide, and we must refuse.
pub const SUPPORTED_REQUIRED_MASK: u64 = 0;

/// Byte offset of [`BlobHeader::pid`] inside the fixed header, as the table
/// in the module docs assigns it. Named so nothing re-derives the layout by
/// hand; `pid_offset_matches_the_writer` pins it to what
/// [`BlobHeader::write_to`] actually emits.
pub const HEADER_PID_OFFSET: usize = 28;

/// Bytes a Windows D3D11 reference plane carries in
/// [`BlobPlane::handle_bytes`]: three little-endian `u64`s, in order the
/// texture NT handle value, the fence NT handle value and the fence value of
/// the last recorded GPU write.
///
/// Not a `required_mask` bit: a plane's `handle_bytes` length is already part
/// of its record, and a consumer that does not understand a D3D11 reference
/// refuses it on the storage kind rather than on the byte count.
pub const D3D11_HANDLE_BYTES: usize = 24;

/// File-descriptor slot in the blob's accompanying table.
///
/// `std::os::fd` exists only on Unix. Windows still type-checks `export` /
/// `import` so the crate builds; the table is unused there (no dma-buf / SHM).
#[cfg(unix)]
type BlobFd = std::os::fd::RawFd;
#[cfg(not(unix))]
type BlobFd = i32;

/// What went wrong parsing a blob.
///
/// Every variant is a *refusal*, never a best-effort recovery: a blob arrives
/// from another wheel, another process, or a network, so every count and length
/// in its header is attacker-controlled. Recovering from a bad one means acting
/// on a value an attacker chose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlobError {
    /// The buffer is shorter than the structure it claims to contain.
    Truncated {
        /// Bytes the blob says it needs.
        need: usize,
        /// Bytes actually available.
        have: usize,
    },
    /// `required_mask` carries a bit this build does not understand. The
    /// producer needs an interpretation we cannot provide, so proceeding would
    /// silently misread the data.
    UnsupportedRequirement(u64),
    /// A count or length would overflow when multiplied out, or derives an
    /// offset past the end of the blob.
    ForgedLength {
        /// Which header field was implausible.
        field: &'static str,
        /// The value it carried.
        value: u64,
    },
    /// A string region is not valid UTF-8.
    InvalidUtf8(&'static str),
    /// The blob is internally inconsistent in a way no single field explains.
    Malformed(&'static str),
}

impl std::fmt::Display for BlobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { need, have } => {
                write!(f, "blob truncated: needs {need} bytes, buffer has {have}")
            }
            Self::UnsupportedRequirement(m) => write!(
                f,
                "blob requires unsupported features (required_mask {m:#x}); \
                 refusing rather than misreading it"
            ),
            Self::ForgedLength { field, value } => {
                write!(f, "blob field `{field}` has implausible value {value}")
            }
            Self::InvalidUtf8(which) => write!(f, "blob string `{which}` is not valid UTF-8"),
            Self::Malformed(why) => write!(f, "malformed blob: {why}"),
        }
    }
}

impl std::error::Error for BlobError {}

/// The blob's fixed header. See the module docs for the byte layout.
///
/// Field order here mirrors the byte layout so the two can be read side by
/// side, but this struct is never cast to or from bytes — see [`Self::write_to`]
/// and [`parse_header`], which go field by field in explicit little-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlobHeader {
    /// Total bytes of this blob, header included. The size prefix that makes
    /// the format append-only: an older consumer reads the prefix it knows and
    /// skips to `size`.
    pub size: u64,
    /// Bits describing additions that change interpretation rather than adding
    /// information. Any bit a consumer does not recognise means refuse.
    pub required_mask: u64,
    /// Byte length of the planes region.
    pub planes_bytes: u64,
    /// Backing store for the whole tensor; the shared `TensorMemory` code.
    pub storage_kind: u32,
    /// Producer pid, for `pidfd_open`/`pidfd_getfd` and, on Windows, for
    /// `OpenProcess(PROCESS_DUP_HANDLE)`. Zero when all planes are inlined,
    /// where there is no handle to reopen.
    ///
    /// **Trust boundary.** This value and the handle values beside it come
    /// out of the blob, so a hostile one makes an importer probe a pid it
    /// chose and a handle slot in that process. That is the whole of what it
    /// buys: the import asks for `PROCESS_DUP_HANDLE` alone, and hands the
    /// duplicate only to `OpenSharedResource1` / `OpenSharedFence`, which
    /// refuse anything that is not a shareable D3D11 object. Nothing here
    /// makes a blob safe to accept from an untrusted peer -- import blobs
    /// from peers you would already hand a texture to.
    pub pid: u32,
    /// Acquire fence, or `-1` for none. Per-tensor, never per-plane.
    pub fence_fd: i32,
    /// Element type; the shared `DType` code.
    pub dtype: u32,
    /// Per-channel quantization axis; `-1` per-tensor, `-2` none.
    pub quant_axis: i32,
    /// Number of dimensions in the addressing grid.
    pub ndim: u32,
    /// Number of plane records in the planes region.
    pub plane_count: u32,
    /// Number of `f32` quantization scales.
    pub quant_scales_len: u32,
    /// Number of `i32` zero points; zero means symmetric.
    pub quant_zero_points_len: u32,
    /// Byte length of the five-string region.
    pub strings_bytes: u32,
    /// Number of stride entries: `ndim`, or 0 meaning densely packed C-order.
    ///
    /// Carried rather than assumed — an empty stride array is a distinct value
    /// per `Tensor.msg`, not an absent one, and flattening the two would break
    /// the reference-mode round-trip identity.
    pub strides_len: u32,
}

impl BlobHeader {
    /// An all-zero header, for tests and for `..` update syntax.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Serialize into the first [`HEADER_LEN`] bytes of `buf`.
    ///
    /// # Panics
    /// If `buf` is shorter than [`HEADER_LEN`].
    pub fn write_to(&self, buf: &mut [u8]) {
        assert!(buf.len() >= HEADER_LEN, "header buffer too small");
        buf[0..8].copy_from_slice(&self.size.to_le_bytes());
        buf[8..16].copy_from_slice(&self.required_mask.to_le_bytes());
        buf[16..24].copy_from_slice(&self.planes_bytes.to_le_bytes());
        buf[24..28].copy_from_slice(&self.storage_kind.to_le_bytes());
        buf[28..32].copy_from_slice(&self.pid.to_le_bytes());
        buf[32..36].copy_from_slice(&self.fence_fd.to_le_bytes());
        buf[36..40].copy_from_slice(&self.dtype.to_le_bytes());
        buf[40..44].copy_from_slice(&self.quant_axis.to_le_bytes());
        buf[44..48].copy_from_slice(&self.ndim.to_le_bytes());
        buf[48..52].copy_from_slice(&self.plane_count.to_le_bytes());
        buf[52..56].copy_from_slice(&self.quant_scales_len.to_le_bytes());
        buf[56..60].copy_from_slice(&self.quant_zero_points_len.to_le_bytes());
        buf[60..64].copy_from_slice(&self.strings_bytes.to_le_bytes());
        buf[64..68].copy_from_slice(&self.strides_len.to_le_bytes());
        buf[68..72].copy_from_slice(&0u32.to_le_bytes()); // reserved
    }
}

/// Read a little-endian `u64` at `off`. The caller has already bounds-checked.
fn rd_u64(buf: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[off..off + 8]);
    u64::from_le_bytes(b)
}

/// Read a little-endian `u32` at `off`. The caller has already bounds-checked.
fn rd_u32(buf: &[u8], off: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[off..off + 4]);
    u32::from_le_bytes(b)
}

/// Parse and validate a blob's fixed header.
///
/// `buf` is the blob, which need not start at offset 0 of its containing
/// allocation — a `CameraFrame` embeds a `Tensor` at an 8-aligned offset
/// precisely so a consumer can subslice and parse in place.
///
/// Validates only what the header alone can prove: that the buffer holds a
/// header at all, that this build understands every required bit, and that
/// `size` is self-consistent. Region bounds are checked when regions are
/// derived, against this `size`.
pub fn parse_header(buf: &[u8]) -> Result<BlobHeader, BlobError> {
    if buf.len() < HEADER_LEN {
        return Err(BlobError::Truncated {
            need: HEADER_LEN,
            have: buf.len(),
        });
    }
    let required_mask = rd_u64(buf, 8);
    // Checked before anything else is trusted: an unknown required bit means
    // the producer changed what some field *means*, so every value below is
    // suspect, not merely incomplete.
    if required_mask & !SUPPORTED_REQUIRED_MASK != 0 {
        return Err(BlobError::UnsupportedRequirement(required_mask));
    }
    let size = rd_u64(buf, 0);
    if size < HEADER_LEN as u64 {
        return Err(BlobError::ForgedLength {
            field: "size",
            value: size,
        });
    }
    // `size` may legitimately exceed `buf.len()` only if the caller handed us a
    // short read; treat that as truncation rather than trusting the claim.
    if size > buf.len() as u64 {
        return Err(BlobError::Truncated {
            need: size as usize,
            have: buf.len(),
        });
    }
    // A non-zero reserved word means a producer used it for something this
    // build cannot see. Same reasoning as `required_mask`: refuse, do not guess.
    if rd_u32(buf, 68) != 0 {
        return Err(BlobError::Malformed("reserved header word is not zero"));
    }
    Ok(BlobHeader {
        size,
        required_mask,
        planes_bytes: rd_u64(buf, 16),
        storage_kind: rd_u32(buf, 24),
        pid: rd_u32(buf, 28),
        fence_fd: rd_u32(buf, 32) as i32,
        dtype: rd_u32(buf, 36),
        quant_axis: rd_u32(buf, 40) as i32,
        ndim: rd_u32(buf, 44),
        plane_count: rd_u32(buf, 48),
        quant_scales_len: rd_u32(buf, 52),
        quant_zero_points_len: rd_u32(buf, 56),
        strings_bytes: rd_u32(buf, 60),
        strides_len: rd_u32(buf, 64),
    })
}

/// Byte offsets of every variable region, derived from the fixed header.
///
/// All offsets are absolute within the blob and already validated against the
/// header's `size`, so a caller may slice with them directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Regions {
    /// `shape[ndim]: u64`.
    pub shape: usize,
    /// `strides[strides_len]: i64`, in bytes. Zero-width when densely packed.
    pub strides: usize,
    /// `quant_scales[]: f32`.
    pub quant_scales: usize,
    /// `quant_zero_points[]: i32`.
    pub quant_zero_points: usize,
    /// The five length-prefixed UTF-8 strings.
    pub strings: usize,
    /// `plane_count` records.
    pub planes: usize,
    /// One past the last byte the regions occupy.
    pub end: usize,
}

/// Bounds-check a count against an element size, then advance an offset.
///
/// Every step is checked: these counts are attacker-controlled, and a wrapped
/// multiply produces a small-looking length that passes a comparison it should
/// fail. `field` names the header field so a refusal says which value was
/// implausible.
fn advance(
    off: usize,
    count: u32,
    elem: usize,
    limit: usize,
    field: &'static str,
) -> Result<usize, BlobError> {
    let bytes = (count as usize)
        .checked_mul(elem)
        .ok_or(BlobError::ForgedLength {
            field,
            value: count as u64,
        })?;
    let next = off.checked_add(bytes).ok_or(BlobError::ForgedLength {
        field,
        value: count as u64,
    })?;
    if next > limit {
        return Err(BlobError::ForgedLength {
            field,
            value: count as u64,
        });
    }
    Ok(next)
}

/// Derive every variable region's offset from the fixed header.
///
/// This is the blob's reason for existing: because the header carries the
/// length of every region, each offset is constant-time arithmetic rather than
/// a sequential walk. It is also the point where attacker-controlled counts
/// become dangerous, so every derivation is checked against `size` before it
/// is returned.
pub fn region_offsets(h: &BlobHeader) -> Result<Regions, BlobError> {
    let limit = h.size as usize;
    if limit < HEADER_LEN {
        return Err(BlobError::ForgedLength {
            field: "size",
            value: h.size,
        });
    }
    // Strides are all-or-nothing: `ndim` entries, or none. A partial array has
    // no meaning, so treat any other count as forged rather than truncating.
    if h.strides_len != 0 && h.strides_len != h.ndim {
        return Err(BlobError::ForgedLength {
            field: "strides_len",
            value: h.strides_len as u64,
        });
    }
    let shape = HEADER_LEN;
    let strides = advance(shape, h.ndim, 8, limit, "ndim")?;
    let quant_scales = advance(strides, h.strides_len, 8, limit, "strides_len")?;
    let quant_zero_points = advance(
        quant_scales,
        h.quant_scales_len,
        4,
        limit,
        "quant_scales_len",
    )?;
    let strings = advance(
        quant_zero_points,
        h.quant_zero_points_len,
        4,
        limit,
        "quant_zero_points_len",
    )?;
    let planes = advance(strings, h.strings_bytes, 1, limit, "strings_bytes")?;
    // Plane records are read as u64s. The strings region is written padded so
    // this holds; a blob claiming otherwise would make every plane read
    // unaligned, so refuse it rather than reading byte-at-a-time.
    if !planes.is_multiple_of(8) {
        return Err(BlobError::Malformed(
            "planes region is not 8-aligned; strings_bytes must include padding",
        ));
    }
    let planes_bytes = usize::try_from(h.planes_bytes).map_err(|_| BlobError::ForgedLength {
        field: "planes_bytes",
        value: h.planes_bytes,
    })?;
    let end = planes
        .checked_add(planes_bytes)
        .ok_or(BlobError::ForgedLength {
            field: "planes_bytes",
            value: h.planes_bytes,
        })?;
    if end > limit {
        return Err(BlobError::ForgedLength {
            field: "planes_bytes",
            value: h.planes_bytes,
        });
    }
    // Bound the plane count against the bytes actually present rather than
    // against a constant: a forged count is a denial-of-service, and the real
    // ceiling is how many minimum-size records could possibly fit.
    let max_planes = planes_bytes / PLANE_RECORD_LEN;
    if h.plane_count as usize > max_planes {
        return Err(BlobError::ForgedLength {
            field: "plane_count",
            value: h.plane_count as u64,
        });
    }
    Ok(Regions {
        shape,
        strides,
        quant_scales,
        quant_zero_points,
        strings,
        planes,
        end,
    })
}

/// The blob's five strings, borrowed from the buffer.
///
/// Strings rather than enum codes, matching `Tensor.msg` exactly. This
/// permanently avoids the mirrored-enum bug class already seen in this repo,
/// where `TensorMemory.MEM == 3` in Python collided with
/// `HAL_TENSOR_MEMORY_PBO == 3` in C: a string cannot be silently reinterpreted
/// by a consumer that disagrees about numbering.
///
/// `edgefirst-tensor` *carries* these; it never parses `"NV12"`.
/// `edgefirst-image` interprets them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlobStrings<'a> {
    /// Format descriptor (`"NV12"`, `"rgb8"`, `"h264"`); `""` = not an image.
    pub format: &'a str,
    /// Colour primaries.
    pub color_space: &'a str,
    /// Transfer function.
    pub color_transfer: &'a str,
    /// YCbCr matrix; `""` for RGB.
    pub color_encoding: &'a str,
    /// `"full"` / `"limited"` / `""`.
    pub color_range: &'a str,
}

/// Number of strings in the region. Fixed by the format, not by the header.
const STRING_COUNT: usize = 5;

/// Encoded length of a strings region, padded so the planes region that follows
/// starts 8-aligned.
pub fn strings_encoded_len(s: &BlobStrings<'_>) -> usize {
    let raw: usize = STRING_COUNT * 4
        + s.format.len()
        + s.color_space.len()
        + s.color_transfer.len()
        + s.color_encoding.len()
        + s.color_range.len();
    raw.next_multiple_of(8)
}

/// Append the strings region to `out`, padded to an 8-byte boundary.
///
/// The padding is not cosmetic: plane records are read as `u64`s, so the
/// planes region must begin 8-aligned. The quantization regions before this one
/// are 4-byte-element arrays and can leave the offset merely 4-aligned, which
/// makes this region the place alignment is restored.
pub fn write_strings(s: &BlobStrings<'_>, out: &mut Vec<u8>) {
    let start = out.len();
    for part in [
        s.format,
        s.color_space,
        s.color_transfer,
        s.color_encoding,
        s.color_range,
    ] {
        out.extend_from_slice(&(part.len() as u32).to_le_bytes());
        out.extend_from_slice(part.as_bytes());
    }
    while !(out.len() - start).is_multiple_of(8) {
        out.push(0);
    }
}

/// Parse the five strings from a strings region.
///
/// `buf` is exactly the region — `blob[regions.strings..regions.planes]`.
pub fn parse_strings(buf: &[u8]) -> Result<BlobStrings<'_>, BlobError> {
    const NAMES: [&str; STRING_COUNT] = [
        "format",
        "color_space",
        "color_transfer",
        "color_encoding",
        "color_range",
    ];
    let mut out: [&str; STRING_COUNT] = [""; STRING_COUNT];
    let mut off = 0usize;
    for (i, name) in NAMES.iter().enumerate() {
        let name = *name;
        let end = off
            .checked_add(4)
            .ok_or(BlobError::Malformed("string count overflows"))?;
        if end > buf.len() {
            return Err(BlobError::Truncated {
                need: end,
                have: buf.len(),
            });
        }
        let len = rd_u32(buf, off) as usize;
        off = end;
        let stop = off.checked_add(len).ok_or(BlobError::ForgedLength {
            field: name,
            value: len as u64,
        })?;
        if stop > buf.len() {
            return Err(BlobError::ForgedLength {
                field: name,
                value: len as u64,
            });
        }
        // `from_utf8`, never `from_utf8_lossy`: a silent replacement character
        // on a hostile buffer hides the malformed input instead of reporting it.
        out[i] = std::str::from_utf8(&buf[off..stop]).map_err(|_| BlobError::InvalidUtf8(name))?;
        off = stop;
    }
    Ok(BlobStrings {
        format: out[0],
        color_space: out[1],
        color_transfer: out[2],
        color_encoding: out[3],
        color_range: out[4],
    })
}

/// One plane record, borrowed from the buffer.
///
/// Field names and widths mirror `TensorPlane.msg` so the two map 1:1. All
/// scalars are 64-bit, and `handle` is `i64` rather than an fd-sized `i32`:
/// cheap now, and the alternative breaks the round-trip property, since a
/// `u32` `size` would cap a plane at 4 GiB — which camera frames never reach
/// but a tensor carrying model weights plausibly does.
///
/// The two transport modes are distinguished by `handle`:
///
/// * `handle >= 0` — **reference**: the bytes live behind the handle, and
///   `data` is empty.
/// * `handle == -1` — **inline**: the bytes are in `data`, `offset` is ignored,
///   `size` describes `data`, and `modifier` must be 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobPlane<'a> {
    /// **In reference mode, an INDEX into the accompanying fd table** — not a
    /// raw file descriptor. `-1` when inline.
    ///
    /// This is an ABI decision, not an implementation detail. A raw fd is
    /// meaningless in the receiving process: fds travel out of band via
    /// `SCM_RIGHTS` or `pidfd_getfd`, and the receiver is handed different
    /// numbers than the sender used. Wayland, D-Bus and `CameraFrame` all carry
    /// the same (payload, fds) split for the same reason.
    ///
    /// Indices also let planes share a descriptor: NV12's luma and chroma
    /// commonly live in one dma-buf at different offsets, and both planes then
    /// carry index 0 while the table holds a single fd.
    ///
    /// Windows has no fd table: a D3D11 texture is shared by NT handle, which
    /// is a value in the *exporting process's* handle table rather than a
    /// descriptor an out-of-band table can carry. Such a plane carries index
    /// `0` over an empty table — non-negative, so the record is still a
    /// reference and not an inline one — and its handles live in
    /// [`Self::handle_bytes`].
    pub handle: i64,
    /// Byte offset of this plane within the handle; ignored when inline.
    pub offset: u64,
    /// Bytes per line of this plane.
    pub stride: u64,
    /// Plane capacity in bytes.
    pub size: u64,
    /// Valid payload bytes; `used <= size`.
    pub used: u64,
    /// DRM format modifier. 0 = linear. A consumer that does not recognise a
    /// non-zero value must refuse, not read the plane as linear.
    pub modifier: u64,
    /// Opaque handle for backends whose handle is not an integer (a
    /// `cudaIpcMemHandle_t` is 64 bytes), or whose handle does not travel in
    /// the fd table at all.
    ///
    /// Windows D3D11 textures are the one storage kind that fills this in
    /// today: [`D3D11_HANDLE_BYTES`] bytes holding the texture's NT handle
    /// value, the process device fence's NT handle value and the fence value
    /// of the producer's last recorded GPU write, each a little-endian `u64`.
    /// All three are values in the exporting process's handle table, which is
    /// why the header's [`BlobHeader::pid`] is what makes them reachable --
    /// and why that field documents the trust boundary they share.
    pub handle_bytes: &'a [u8],
    /// Inlined plane bytes; non-empty only when `handle == -1`.
    pub data: &'a [u8],
}

impl BlobPlane<'_> {
    /// True when this plane carries its bytes rather than a reference to them.
    pub fn is_inline(&self) -> bool {
        self.handle < 0
    }

    /// Encoded length of this record, including the padding that keeps the
    /// following record 8-aligned.
    pub fn encoded_len(&self) -> usize {
        (PLANE_RECORD_LEN + self.handle_bytes.len() + self.data.len()).next_multiple_of(8)
    }
}

/// Append one plane record to `out`, padded so the next record starts 8-aligned.
pub fn write_plane(p: &BlobPlane<'_>, out: &mut Vec<u8>) {
    let start = out.len();
    out.extend_from_slice(&p.handle.to_le_bytes());
    out.extend_from_slice(&p.offset.to_le_bytes());
    out.extend_from_slice(&p.stride.to_le_bytes());
    out.extend_from_slice(&p.size.to_le_bytes());
    out.extend_from_slice(&p.used.to_le_bytes());
    out.extend_from_slice(&p.modifier.to_le_bytes());
    out.extend_from_slice(&(p.handle_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&(p.data.len() as u32).to_le_bytes());
    out.extend_from_slice(p.handle_bytes);
    out.extend_from_slice(p.data);
    while !(out.len() - start).is_multiple_of(8) {
        out.push(0);
    }
}

/// Read a little-endian `i64` at `off`. The caller has already bounds-checked.
fn rd_i64(buf: &[u8], off: usize) -> i64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[off..off + 8]);
    i64::from_le_bytes(b)
}

/// Parse `count` plane records from a planes region.
///
/// `buf` is exactly the region — `blob[regions.planes..regions.end]`. `count`
/// has already been bounded by [`region_offsets`] against the bytes present,
/// but every per-record length is re-checked here because those live inside
/// the region rather than in the header.
pub fn parse_planes(buf: &[u8], count: u32) -> Result<Vec<BlobPlane<'_>>, BlobError> {
    let mut planes = Vec::with_capacity(count as usize);
    let mut off = 0usize;
    for _ in 0..count {
        let head = off
            .checked_add(PLANE_RECORD_LEN)
            .ok_or(BlobError::Malformed("plane record offset overflows"))?;
        if head > buf.len() {
            return Err(BlobError::Truncated {
                need: head,
                have: buf.len(),
            });
        }
        let hb_len = rd_u32(buf, off + 48) as usize;
        let data_len = rd_u32(buf, off + 52) as usize;
        let hb_end = head.checked_add(hb_len).ok_or(BlobError::ForgedLength {
            field: "handle_bytes_len",
            value: hb_len as u64,
        })?;
        let data_end = hb_end
            .checked_add(data_len)
            .ok_or(BlobError::ForgedLength {
                field: "data_len",
                value: data_len as u64,
            })?;
        if hb_end > buf.len() {
            return Err(BlobError::ForgedLength {
                field: "handle_bytes_len",
                value: hb_len as u64,
            });
        }
        if data_end > buf.len() {
            return Err(BlobError::ForgedLength {
                field: "data_len",
                value: data_len as u64,
            });
        }
        planes.push(BlobPlane {
            handle: rd_i64(buf, off),
            offset: rd_u64(buf, off + 8),
            stride: rd_u64(buf, off + 16),
            size: rd_u64(buf, off + 24),
            used: rd_u64(buf, off + 32),
            modifier: rd_u64(buf, off + 40),
            handle_bytes: &buf[head..hb_end],
            data: &buf[hb_end..data_end],
        });
        off = data_end.next_multiple_of(8);
    }
    Ok(planes)
}

/// How a plane's bytes cross a boundary.
///
/// Both modes are first-class; the choice is the application's, not the HAL's.
/// The HAL's job is to make it *expressible*, which is why `handle == -1` plus
/// carried bytes is part of the format rather than a convention layered on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    /// Carry handles. Requires the same host and an IPC-capable storage kind.
    /// Export borrows: the handles are valid only while the source tensor lives.
    Reference,
    /// Carry the bytes. Always available, and *required* for `mem` and `pbo`
    /// (which have no shareable handle) and for any storage kind travelling
    /// over a network, where an fd and a pid are meaningless.
    Inline,
}

/// A parsed, validated view over a blob.
///
/// Borrows the buffer; every accessor is a constant-time slice, not a walk.
#[derive(Debug, Clone, Copy)]
pub struct BlobView<'a> {
    buf: &'a [u8],
    header: BlobHeader,
    regions: Regions,
}

impl<'a> BlobView<'a> {
    /// Parse and validate a blob.
    ///
    /// `buf` need not start at offset 0 of its allocation — a `CameraFrame`
    /// embeds a `Tensor` 8-aligned so a consumer can subslice and parse in
    /// place, so this takes a slice and never requires ownership.
    pub fn parse(buf: &'a [u8]) -> Result<Self, BlobError> {
        let header = parse_header(buf)?;
        let regions = region_offsets(&header)?;
        Ok(Self {
            buf,
            header,
            regions,
        })
    }

    /// The validated fixed header.
    pub fn header(&self) -> BlobHeader {
        self.header
    }

    /// The addressing grid.
    pub fn shape(&self) -> Vec<u64> {
        (0..self.header.ndim as usize)
            .map(|i| rd_u64(self.buf, self.regions.shape + i * 8))
            .collect()
    }

    /// Strides in bytes; empty means densely packed C-order.
    pub fn strides(&self) -> Vec<i64> {
        (0..self.header.strides_len as usize)
            .map(|i| rd_i64(self.buf, self.regions.strides + i * 8))
            .collect()
    }

    /// The five carried strings.
    pub fn strings(&self) -> BlobStrings<'a> {
        parse_strings(&self.buf[self.regions.strings..self.regions.planes]).unwrap_or_default()
    }

    /// The plane records.
    pub fn planes(&self) -> Result<Vec<BlobPlane<'a>>, BlobError> {
        parse_planes(
            &self.buf[self.regions.planes..self.regions.end],
            self.header.plane_count,
        )
    }
}

/// Serialize a tensor to a blob in the requested transport mode.
///
/// Returns the blob **and its fd table**. The two travel together and are
/// meaningless apart: a plane's `handle` is an index into the table, because a
/// raw fd number does not survive a process boundary.
///
/// **Export borrows and performs no syscalls.** In [`TransportMode::Reference`]
/// the fds in the table are the source tensor's own, valid only while it lives;
/// `dup` happens on import, never here. That is what makes export cheap and
/// removes any keepalive protocol between producer and consumer.
///
/// On Windows the table is empty and each plane carries
/// [`D3D11_HANDLE_BYTES`] instead: the texture's NT handle value, the process
/// device fence's, and the last recorded GPU write. Those are borrowed on the
/// same terms -- they are handle-table entries of *this* process, which the
/// header's [`pid`](BlobHeader::pid) is what lets an importer reach.
pub fn export(t: &crate::TensorDyn, mode: TransportMode) -> crate::Result<(Vec<u8>, Vec<BlobFd>)> {
    let fmt = t.format();
    // The addressing grid, not the allocation: for a subsampled format these
    // differ, and the blob carries the grid because extent comes from planes.
    let shape = export_addressing_shape(t, fmt)?;
    let esz = t.dtype().size() as i64;
    let strides = export_c_strides(t, fmt, &shape, esz);

    let quant = t.quantization();
    let (quant_axis, scales, zeros): (i32, &[f32], &[i32]) = match quant {
        None => (-2, &[], &[]),
        Some(q) => (
            q.axis().map(|a| a as i32).unwrap_or(-1),
            q.scale(),
            q.zero_point().unwrap_or(&[]),
        ),
    };

    let colorimetry = t.colorimetry().unwrap_or_default();
    let strings = BlobStrings {
        format: fmt.map(|f| f.as_str()).unwrap_or(""),
        color_space: colorimetry.space.map(|v| v.as_str()).unwrap_or(""),
        color_transfer: colorimetry.transfer.map(|v| v.as_str()).unwrap_or(""),
        color_encoding: colorimetry.encoding.map(|v| v.as_str()).unwrap_or(""),
        color_range: colorimetry.range.map(|v| v.as_str()).unwrap_or(""),
    };

    // Plane geometry: from the format's table when this is an image, otherwise
    // a single plane spanning the whole allocation.
    let geoms = export_plane_geoms(t, fmt, esz)?;

    // Bytes, only when inlining. `pin_host` is the one read here; reference
    // mode performs no syscalls at all.
    let pin = match mode {
        TransportMode::Inline => Some(t.pin_host(crate::CpuAccess::Read)?),
        TransportMode::Reference => None,
    };
    let all_bytes: &[u8] = match &pin {
        Some(p) => unsafe { std::slice::from_raw_parts(p.as_mut_ptr(), p.len()) },
        None => &[],
    };

    // The fd table. Planes sharing one descriptor (NV12 luma + chroma in a
    // single dma-buf) must share one entry, or the receiver would dup the same
    // buffer twice and the offsets would no longer refer to one allocation.
    let (handle, handle_bytes, fds): (i64, Vec<u8>, Vec<BlobFd>) = match mode {
        TransportMode::Inline => (-1, Vec::new(), Vec::new()),
        TransportMode::Reference => {
            let raw = reference_handle(t)?;
            #[cfg(target_os = "windows")]
            {
                // The table stays empty: an NT handle is not a file
                // descriptor and does not travel out of band, so the handle
                // values ride in the record and the importer duplicates them
                // out of the pid the header carries.
                (0, reference_handle_bytes(t, raw), Vec::new())
            }
            #[cfg(not(target_os = "windows"))]
            {
                let raw = i32::try_from(raw).map_err(|_| {
                    crate::Error::InvalidOperation("handle is not representable as an fd".into())
                })?;
                // `handle` is an index into `fds`, never the fd itself.
                (0, Vec::new(), vec![raw])
            }
        }
    };

    let mut planes_buf = Vec::new();
    write_export_planes(
        &geoms,
        mode,
        handle,
        &handle_bytes,
        all_bytes,
        &mut planes_buf,
    )?;

    let strings_bytes = strings_encoded_len(&strings);
    let header = BlobHeader {
        size: 0, // patched below
        required_mask: 0,
        planes_bytes: planes_buf.len() as u64,
        storage_kind: t.memory().code(),
        // No handle to reopen once the bytes travel, so no pid to reopen it with.
        pid: match mode {
            TransportMode::Inline => 0,
            TransportMode::Reference => std::process::id(),
        },
        // Reserved in this build, and meaningless inline in any case: inlining
        // means the producer already read the bytes, so it already waited.
        fence_fd: -1,
        dtype: t.dtype().code(),
        quant_axis,
        ndim: shape.len() as u32,
        plane_count: geoms.len() as u32,
        quant_scales_len: scales.len() as u32,
        quant_zero_points_len: zeros.len() as u32,
        strings_bytes: strings_bytes as u32,
        strides_len: strides.len() as u32,
    };

    let mut out = vec![0u8; HEADER_LEN];
    for d in &shape {
        out.extend_from_slice(&d.to_le_bytes());
    }
    for st in &strides {
        out.extend_from_slice(&st.to_le_bytes());
    }
    for sc in scales {
        out.extend_from_slice(&sc.to_le_bytes());
    }
    for z in zeros {
        out.extend_from_slice(&z.to_le_bytes());
    }
    write_strings(&strings, &mut out);
    out.extend_from_slice(&planes_buf);

    let mut header = header;
    header.size = out.len() as u64;
    header.write_to(&mut out);
    Ok((out, fds))
}

fn export_addressing_shape(
    t: &crate::TensorDyn,
    fmt: Option<crate::PixelFormat>,
) -> crate::Result<Vec<u64>> {
    match fmt {
        Some(f) => {
            let (w, h) = image_dims(t, f)?;
            Ok(f.addressing_shape(w, h)
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!("no addressing shape for {f:?}"))
                })?
                .iter()
                .map(|d| *d as u64)
                .collect())
        }
        None => Ok(t.shape().iter().map(|d| *d as u64).collect()),
    }
}

fn export_c_strides(
    t: &crate::TensorDyn,
    fmt: Option<crate::PixelFormat>,
    shape: &[u64],
    esz: i64,
) -> Vec<i64> {
    let mut acc = esz;
    let mut v = vec![0i64; shape.len()];
    for i in (0..shape.len()).rev() {
        v[i] = acc;
        acc *= shape[i] as i64;
    }
    if let (Some(rs), true) = (t.row_stride(), shape.len() >= 2) {
        let row_dim = match fmt.map(|f| f.layout()) {
            Some(crate::PixelLayout::Planar) if shape.len() >= 3 => 1,
            _ => 0,
        };
        v[row_dim] = rs as i64;
    }
    v
}

fn export_plane_geoms(
    t: &crate::TensorDyn,
    fmt: Option<crate::PixelFormat>,
    esz: i64,
) -> crate::Result<Vec<crate::PlaneGeometry>> {
    match fmt {
        Some(f) => {
            let (w, h) = image_dims(t, f)?;
            let rs = t
                .effective_row_stride()
                .unwrap_or(w * f.channels() * esz as usize);
            f.plane_table(w, h, rs)
                .ok_or_else(|| crate::Error::InvalidArgument(format!("no plane table for {f:?}")))
        }
        None => Ok(vec![crate::PlaneGeometry {
            offset: 0,
            stride: t.capacity_bytes() as u64,
            size: t.capacity_bytes() as u64,
        }]),
    }
}

fn write_export_planes(
    geoms: &[crate::PlaneGeometry],
    mode: TransportMode,
    handle: i64,
    handle_bytes: &[u8],
    all_bytes: &[u8],
    planes_buf: &mut Vec<u8>,
) -> crate::Result<()> {
    for g in geoms {
        let data: &[u8] = match mode {
            TransportMode::Inline => {
                let start = g.offset as usize;
                let end = start
                    .checked_add(g.size as usize)
                    .ok_or(crate::Error::InvalidSize(g.size as usize))?;
                all_bytes
                    .get(start..end)
                    .ok_or(crate::Error::InvalidSize(end))?
            }
            TransportMode::Reference => &[],
        };
        write_plane(
            &BlobPlane {
                handle,
                // Ignored when inline, and zeroed so the record cannot be
                // misread as pointing into a handle it does not have.
                offset: if mode == TransportMode::Inline {
                    0
                } else {
                    g.offset
                },
                stride: g.stride,
                // Inline: `size` describes `data`, per the schema's own rule.
                size: if mode == TransportMode::Inline {
                    data.len() as u64
                } else {
                    g.size
                },
                used: if mode == TransportMode::Inline {
                    data.len() as u64
                } else {
                    g.size
                },
                modifier: 0,
                // Every plane carries the same bytes, for the same reason
                // every plane carries the same fd index: they name one
                // allocation, and the plane table already says where each
                // one starts inside it.
                handle_bytes,
                data,
            },
            planes_buf,
        );
    }
    Ok(())
}

/// Recover an image's logical width and height from its allocation shape.
///
/// The tensor stores its allocation geometry; the blob needs the logical
/// dimensions to ask the format for a grid and a plane table.
fn image_dims(t: &crate::TensorDyn, f: crate::PixelFormat) -> crate::Result<(usize, usize)> {
    let shape = t.shape();
    let bad = || crate::Error::InvalidShape(format!("shape {shape:?} is not an image for {f:?}"));
    match f.layout() {
        crate::PixelLayout::Packed => {
            if shape.len() < 2 {
                return Err(bad());
            }
            Ok((shape[1], shape[0]))
        }
        crate::PixelLayout::Planar => {
            if shape.len() < 3 {
                return Err(bad());
            }
            Ok((shape[2], shape[1]))
        }
        crate::PixelLayout::SemiPlanar => {
            if shape.len() < 2 {
                return Err(bad());
            }
            // The stored shape is the combined-plane allocation; invert it to
            // recover the luma height rather than assuming h*2/3, which is
            // wrong for odd heights and for NV16/NV24.
            let total = shape[0];
            let h = (0..=total)
                .find(|&h| f.combined_plane_height(h) == Some(total))
                .ok_or_else(bad)?;
            Ok((shape[1], h))
        }
    }
}

/// The handle a reference-mode export borrows, or an error when the storage
/// kind has none to share.
fn reference_handle(t: &crate::TensorDyn) -> crate::Result<i64> {
    match t.memory() {
        #[cfg(target_os = "linux")]
        crate::TensorMemory::DmaBuf => {
            use std::os::fd::AsRawFd;
            // `dmabuf()` is fallible rather than optional on Linux, so this
            // maps the error instead of an absent value.
            t.dmabuf().map(|fd| fd.as_raw_fd() as i64).map_err(|e| {
                crate::Error::InvalidOperation(format!("dma-buf tensor has no fd: {e}"))
            })
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        crate::TensorMemory::DmaBuf => t
            .iosurface_id()
            .map(|id| id as i64)
            .ok_or_else(|| crate::Error::InvalidOperation("IOSurface tensor has no id".into())),
        // The tensor's own NT handle value, not a duplicate: export borrows,
        // so the value is valid exactly as long as the source tensor is, and
        // a duplicate made here would have nowhere to be closed. The importer
        // duplicates instead -- out of this process, by pid.
        #[cfg(target_os = "windows")]
        crate::TensorMemory::DmaBuf => {
            t.d3d11_shared_handle_value()
                .map(|h| h as i64)
                .ok_or_else(|| {
                    crate::Error::InvalidOperation(
                        "D3D11 texture tensor has no shared handle".into(),
                    )
                })
        }
        // `mem` and `pbo` have no shareable handle at all. Refusing beats
        // silently inlining: the caller asked for a reference and would
        // otherwise receive a copy without being told.
        other => Err(crate::Error::InvalidOperation(format!(
            "{other:?} has no shareable handle; export it with TransportMode::Inline"
        ))),
    }
}

/// The [`D3D11_HANDLE_BYTES`] a Windows reference plane carries: the
/// texture's NT handle value (`raw`), the process device fence's, and the
/// fence value of the last recorded GPU write.
///
/// A device with no fence -- or one whose fence is private to this copy
/// rather than the process-shared one, whose values another copy's recorded
/// write is not comparable with -- contributes a zero handle **and a zero
/// value**, so the import builds no completion. That is the same answer a
/// tensor with no recorded write gives: nothing to wait on.
///
/// The value has to go with the handle. `import` refuses a value with no
/// fence beside it -- a completion nobody can wait on -- so exporting the
/// recorded value on its own would make a degraded copy's blobs unimportable
/// rather than completion-less.
#[cfg(target_os = "windows")]
fn reference_handle_bytes(t: &crate::TensorDyn, raw: i64) -> Vec<u8> {
    let fence = crate::d3d11::device()
        .map(|d| d.exported_fence_handle_value())
        .unwrap_or(0);
    let (fence, value) = reference_completion(fence, t.gpu_write_value());
    let mut out = Vec::with_capacity(D3D11_HANDLE_BYTES);
    out.extend_from_slice(&(raw as u64).to_le_bytes());
    out.extend_from_slice(&fence.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
    out
}

/// The `(fence handle, fence value)` pair a reference plane carries, given
/// what the device is willing to export and what the tensor recorded.
///
/// One rule, in one place, because the importer enforces its other half: a
/// value with no fence beside it is refused outright (a completion nobody can
/// wait on), so a copy that exports no handle -- no fence at all, or a private
/// one whose values no other copy can compare with -- must drop the value with
/// it. A handle with a zero value is fine and means "nothing recorded yet".
#[cfg(target_os = "windows")]
fn reference_completion(fence: usize, recorded: u64) -> (u64, u64) {
    if fence == 0 {
        (0, 0)
    } else {
        (fence as u64, recorded)
    }
}

/// Reconstruct a tensor from a blob.
///
/// **Import dups every fd it retains**, so the result is fully independent and
/// the source tensor may die immediately. In inline mode the result is a new
/// allocation holding a copy of the carried bytes.
///
/// `fds` is the table that travelled beside the blob; a reference-mode plane's
/// `handle` indexes into it. Pass an empty slice for an inline blob, and on
/// Windows for a reference-mode one too: a D3D11 texture is named by the NT
/// handle values in the plane's [`BlobPlane::handle_bytes`], and this
/// duplicates them out of the exporting process (the header's
/// [`pid`](BlobHeader::pid)) with `OpenProcess(PROCESS_DUP_HANDLE)` when that
/// is not this process. The duplication is what makes the import independent
/// there, exactly as `dup` does on Unix. Both the pid and the handle values
/// are blob fields, so a hostile blob can steer that probe -- see
/// [`BlobHeader::pid`] for what it can and cannot reach.
///
/// `blob` is untrusted: it may come from another wheel, another process, or a
/// network. Every length has already been bounded by [`BlobView::parse`]; this
/// function adds the *semantic* checks that parsing alone cannot make, and
/// every index is bounded against `fds` before use.
pub fn import(blob: &[u8], fds: &[BlobFd]) -> crate::Result<crate::TensorDyn> {
    #[cfg(not(unix))]
    let _ = fds;
    let v = BlobView::parse(blob).map_err(|e| crate::Error::InvalidArgument(e.to_string()))?;
    let h = v.header();
    let planes = v
        .planes()
        .map_err(|e| crate::Error::InvalidArgument(e.to_string()))?;

    // All planes inline, or none. A frame mixing transport modes has no
    // coherent meaning: one storage_kind, pid and fence_fd cover every plane,
    // so half of them cannot be somewhere else.
    let inline = planes.first().map(BlobPlane::is_inline).unwrap_or(true);
    validate_import_planes(&planes, inline)?;

    let dtype = crate::protocol::dtype_to_dtype(h.dtype)
        .ok_or_else(|| crate::Error::NotImplemented(format!("blob dtype code {}", h.dtype)))?;
    let strings = v.strings();
    let format = if strings.format.is_empty() {
        None
    } else {
        Some(
            crate::PixelFormat::from_str_code(strings.format).ok_or_else(|| {
                crate::Error::NotImplemented(format!("blob format {:?}", strings.format))
            })?,
        )
    };

    if !inline {
        #[cfg(target_os = "windows")]
        {
            return import_referenced_d3d11_blob(&v, &h, &planes, format, &strings, dtype);
        }
        #[cfg(not(any(unix, target_os = "windows")))]
        {
            return Err(crate::Error::NotImplemented(
                "reference-mode blob import is not available on this platform".into(),
            ));
        }
        #[cfg(unix)]
        {
            return import_referenced_blob(&v, &h, &planes, fds, format, &strings, dtype);
        }
    }

    // Allocate the ALLOCATION geometry, not the grid. The grid is what the blob
    // carries; the buffer that backs it is larger for a subsampled format, and
    // sizing from the grid would undersize NV12 by a third.
    let grid: Vec<usize> = v.shape().iter().map(|d| *d as usize).collect();
    let alloc_shape = alloc_shape_from_grid(format, &grid)?;

    let mut t = crate::TensorDyn::new(&alloc_shape, dtype, Some(crate::TensorMemory::Mem), None)?;
    if let Some(f) = format {
        t.set_format(f)?;
    }

    // Copy each plane back to where the format says it belongs. The exporter
    // zeroes `offset` on an inline plane (it names a position inside a handle
    // that is not being carried), so the layout is rebuilt from the plane
    // table rather than trusted from the wire.
    blit_inline_planes(&mut t, &planes, format, &grid, dtype)?;

    t.set_colorimetry(colorimetry_from(&strings));
    if h.quant_axis != -2 {
        let scales: Vec<f32> = (0..h.quant_scales_len as usize)
            .map(|i| {
                let mut b = [0u8; 4];
                let off = v.regions.quant_scales + i * 4;
                b.copy_from_slice(&blob[off..off + 4]);
                f32::from_le_bytes(b)
            })
            .collect();
        let zeros: Vec<i32> = (0..h.quant_zero_points_len as usize)
            .map(|i| {
                let off = v.regions.quant_zero_points + i * 4;
                rd_u32(blob, off) as i32
            })
            .collect();
        if let Some(q) = quantization_from(h.quant_axis, &scales, &zeros) {
            t.set_quantization(q)?;
        }
    }
    Ok(t)
}

fn validate_import_planes(planes: &[BlobPlane<'_>], inline: bool) -> crate::Result<()> {
    if planes.iter().any(|p| p.is_inline() != inline) {
        return Err(crate::Error::InvalidArgument(
            "blob mixes inline and referenced planes; a tensor has one transport mode".into(),
        ));
    }
    for p in planes {
        if p.is_inline() {
            // The schemas validator's own rules, enforced on the way in too.
            if p.size as usize != p.data.len() {
                return Err(crate::Error::InvalidArgument(format!(
                    "inline plane declares size {} but carries {} bytes",
                    p.size,
                    p.data.len()
                )));
            }
            if p.modifier != 0 || !p.handle_bytes.is_empty() {
                return Err(crate::Error::InvalidArgument(
                    "inline plane must have modifier 0 and no handle_bytes".into(),
                ));
            }
        } else if !p.data.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "referenced plane must not also carry inline bytes".into(),
            ));
        }
    }
    Ok(())
}

fn alloc_shape_from_grid(
    format: Option<crate::PixelFormat>,
    grid: &[usize],
) -> crate::Result<Vec<usize>> {
    match format {
        Some(f) => {
            let (w, h_px) = dims_from_grid(f, grid)?;
            f.allocation_shape(w, h_px)
                .ok_or_else(|| crate::Error::InvalidShape(format!("no allocation shape for {f:?}")))
        }
        None => Ok(grid.to_vec()),
    }
}

#[cfg(unix)]
fn import_referenced_blob(
    v: &BlobView<'_>,
    h: &BlobHeader,
    planes: &[BlobPlane<'_>],
    fds: &[BlobFd],
    format: Option<crate::PixelFormat>,
    strings: &BlobStrings<'_>,
    dtype: crate::DType,
) -> crate::Result<crate::TensorDyn> {
    // Reference mode. Every plane's `handle` is an index into `fds`, and
    // the index is bounded before use: it arrived from an untrusted blob,
    // and an out-of-range read here would hand an arbitrary descriptor to
    // `dup`.
    let first = planes
        .first()
        .ok_or_else(|| crate::Error::InvalidArgument("referenced blob carries no planes".into()))?;
    let idx = usize::try_from(first.handle).map_err(|_| {
        crate::Error::InvalidArgument(format!("plane handle {} is not an index", first.handle))
    })?;
    let raw = *fds.get(idx).ok_or_else(|| {
        crate::Error::InvalidArgument(format!(
            "plane handle indexes fd {idx} but only {} were provided",
            fds.len()
        ))
    })?;
    for p in planes {
        let i = usize::try_from(p.handle).unwrap_or(usize::MAX);
        if i >= fds.len() {
            return Err(crate::Error::InvalidArgument(format!(
                "plane handle indexes fd {i} but only {} were provided",
                fds.len()
            )));
        }
    }

    let grid: Vec<usize> = v.shape().iter().map(|d| *d as usize).collect();
    let alloc_shape = alloc_shape_from_grid(format, &grid)?;
    let mut t = open_referenced_tensor(h.storage_kind, raw, &alloc_shape, dtype)?;
    if let Some(f) = format {
        t.set_format(f)?;
    }
    if first.stride > 0 {
        // The producer's pitch, which the shape alone cannot express.
        let _ = t.set_row_stride(first.stride as usize);
    }
    t.set_colorimetry(colorimetry_from(strings));
    Ok(t)
}

#[cfg(unix)]
fn open_referenced_tensor(
    storage_kind: u32,
    raw: BlobFd,
    alloc_shape: &[usize],
    dtype: crate::DType,
) -> crate::Result<crate::TensorDyn> {
    // The table entry is a PLATFORM handle, not universally a file
    // descriptor: on Linux it is a dma-buf fd, on macOS an IOSurface ID.
    // Dispatching on `storage_kind` rather than assuming an fd is required
    // -- treating an IOSurface ID as an fd yields EBADF, which is how this
    // gap was found.
    match crate::TensorMemory::from_code(storage_kind) {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        Some(crate::TensorMemory::DmaBuf) | Some(crate::TensorMemory::IoSurface) => {
            let id = u32::try_from(raw).map_err(|_| {
                crate::Error::InvalidArgument(format!("{raw} is not an IOSurface id"))
            })?;
            crate::TensorDyn::from_iosurface_id(id, alloc_shape, dtype, None)
        }
        #[cfg(all(unix, not(any(target_os = "macos", target_os = "ios"))))]
        Some(crate::TensorMemory::DmaBuf) | Some(crate::TensorMemory::Shm) => {
            use std::os::fd::{BorrowedFd, OwnedFd};
            // Import dups: the result is independent and the producer may
            // die immediately, which is what removes the keepalive protocol.
            // SAFETY: `raw` is live for this call -- the caller owns it and
            // has not yet closed its own copy. Borrow, never adopt.
            let borrowed = unsafe { BorrowedFd::borrow_raw(raw) };
            let owned: OwnedFd = borrowed
                .try_clone_to_owned()
                .map_err(crate::Error::IoError)?;
            crate::TensorDyn::from_fd(owned, alloc_shape, dtype, None)
        }
        other => Err(crate::Error::NotImplemented(format!(
            "reference-mode import for {other:?} on this platform"
        ))),
    }
}

/// The Windows sibling of [`import_referenced_blob`]: a D3D11 texture named by
/// the NT handle values in the first plane's
/// [`handle_bytes`](BlobPlane::handle_bytes), reachable through the header's
/// pid.
///
/// No fd table is consulted -- there is none on Windows -- so the plane
/// `handle` is only ever read as "this record is a reference", which
/// [`BlobPlane::is_inline`] has already decided.
#[cfg(target_os = "windows")]
fn import_referenced_d3d11_blob(
    v: &BlobView<'_>,
    h: &BlobHeader,
    planes: &[BlobPlane<'_>],
    format: Option<crate::PixelFormat>,
    strings: &BlobStrings<'_>,
    dtype: crate::DType,
) -> crate::Result<crate::TensorDyn> {
    let first = planes
        .first()
        .ok_or_else(|| crate::Error::InvalidArgument("referenced blob carries no planes".into()))?;
    // Every plane must name the same texture, exactly as every Unix plane
    // must index the same fd: the planes describe one allocation and the
    // plane table already says where each starts inside it, so a blob whose
    // planes disagree about which object that is has no coherent reading.
    // Checked over all of them, not just the one the open uses, because the
    // disagreement is the corruption.
    for p in planes {
        if p.handle != 0 || p.handle_bytes != first.handle_bytes {
            return Err(crate::Error::InvalidArgument(
                "D3D11 reference blob planes must all carry handle 0 and identical \
                 handle bytes; they name one texture"
                    .into(),
            ));
        }
    }
    // The blob's own grid, not an allocation shape derived from it: this is
    // the shape hint the geometry check wants, and a semi-planar texture's
    // image width can only come from a shape (the texture is as wide as the
    // driver's row pitch).
    let grid: Vec<usize> = v.shape().iter().map(|d| *d as usize).collect();
    let mut t = open_referenced_texture(h.storage_kind, first, h.pid, &grid, dtype, format)?;
    // The producer's `stride` is deliberately not restored. A texture import
    // has a pitch of its own -- `from_d3d11_shared_handle` records the one
    // this device's staging copy reports -- and the producer's is a fact
    // about the producer's driver, not about the texture as this process
    // sees it. The descriptor path makes the same exclusion for the same
    // reason (`restore_imported_row_stride` is `HOST | DMABUF` only).
    t.set_colorimetry(colorimetry_from(strings));
    Ok(t)
}

/// Open the D3D11 texture a reference-mode plane names, duplicating both NT
/// handles out of the exporting process when that is not this one.
///
/// The handle values are the exporting process's, so in the same process (a
/// second module of one wheel, or the export-then-import round trip a test
/// makes) they are usable as they stand -- one handle table. Across processes
/// they are duplicated through a `PROCESS_DUP_HANDLE` handle on the exporter,
/// which is what the header's pid is carried for.
///
/// The tensor lands on the format's *allocation* shape, exactly as the Unix
/// arm does: [`export`] writes the addressing shape for every image, so the
/// producer's own `shape()` is not recoverable from the blob and picking the
/// addressing spelling here would make one blob import at two different
/// shapes depending on the platform.
#[cfg(target_os = "windows")]
fn open_referenced_texture(
    storage_kind: u32,
    first: &BlobPlane<'_>,
    pid: u32,
    grid: &[usize],
    dtype: crate::DType,
    format: Option<crate::PixelFormat>,
) -> crate::Result<crate::TensorDyn> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_DUP_HANDLE};

    if storage_kind != crate::TensorMemory::DmaBuf.code() {
        return Err(crate::Error::InvalidArgument(format!(
            "storage kind {storage_kind} is not referenceable on Windows"
        )));
    }
    if first.handle_bytes.len() != D3D11_HANDLE_BYTES {
        return Err(crate::Error::InvalidArgument(format!(
            "D3D11 reference plane needs {D3D11_HANDLE_BYTES} handle bytes, not {}",
            first.handle_bytes.len()
        )));
    }
    let format = format.ok_or_else(|| {
        crate::Error::InvalidArgument("D3D11 reference blob without a pixel format".into())
    })?;
    let word = |i: usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&first.handle_bytes[i * 8..(i + 1) * 8]);
        u64::from_le_bytes(b)
    };
    let handle_at = |i: usize| HANDLE(word(i) as usize as *mut std::ffi::c_void);
    let (mut tex, mut fence, value) = (handle_at(0), handle_at(1), word(2));
    if tex.is_invalid() {
        return Err(crate::Error::InvalidArgument(
            "D3D11 reference plane carries no texture handle".into(),
        ));
    }
    // The same refusal the descriptor path makes: a fence value with no fence
    // to read it on names a completion nobody can wait on, and importing
    // anyway would alias a texture the producer's device may still be writing.
    // The other way round is not an error -- a zero value means nothing was
    // recorded, so a fence handle beside it is simply unused.
    if fence.is_invalid() && value != 0 {
        return Err(crate::Error::InvalidArgument(format!(
            "D3D11 reference plane names fence value {value} with no fence handle; \
             the completion it names cannot be waited on"
        )));
    }

    // Duplicates this import owns, closed by `OwnedHandle` on every path --
    // including a panic between the duplication and the constructor. Empty in
    // the same process, where the values are already this process's own.
    let mut duplicates: Vec<OwnedHandle> = Vec::new();
    if std::process::id() != pid {
        // SAFETY: documented call. `pid` came out of the blob, so it may name
        // a process this token cannot reach or one that no longer exists;
        // both fail rather than returning a handle.
        let src = unsafe { OpenProcess(PROCESS_DUP_HANDLE, false, pid) }
            .map_err(|e| open_process_error(pid, &e))?;
        // SAFETY: `OpenProcess` returned a live handle this scope owns; the
        // wrapper is what closes it.
        let src = unsafe { OwnedHandle::from_raw_handle(src.0) };
        for (h, which) in [(&mut tex, "texture"), (&mut fence, "fence")] {
            if h.is_invalid() {
                continue;
            }
            let dup = duplicate_out_of(&src, *h, pid, which)?;
            *h = HANDLE(dup.as_raw_handle());
            duplicates.push(dup);
        }
    }

    // SAFETY: `tex` is a shared NT texture handle valid in this process --
    // the exporter's own value in the same process, a duplicate otherwise.
    // The helper opens its own texture and drops it on return.
    let geometry = unsafe { crate::protocol::d3d11_geometry_checked(format, tex.0, grid) };
    // A fence value of 0 is "nothing recorded", so there is nothing to wait on
    // even when the device published a fence handle.
    let completion = (value != 0).then_some((fence.0, value));
    let opened = match geometry {
        // SAFETY: both handles are valid in this process, as above, and the
        // constructor duplicates what it keeps rather than adopting them.
        Ok((width, height)) => unsafe {
            crate::TensorDyn::from_d3d11_shared_handle(
                tex.0,
                width,
                height,
                format,
                dtype,
                crate::CpuAccess::ReadWrite,
                completion,
                None,
            )
        },
        Err(e) => Err(e),
    };
    // The constructor duplicated what it keeps, so whatever was duplicated
    // here is a temporary; this is where it closes.
    drop(duplicates);
    opened
}

/// The `ErrorKind` a Win32 failure maps to, by the code the call itself
/// reported rather than a re-read of the thread's last error that any cleanup
/// call in between can overwrite.
///
/// Spec section 8 names `EACCES` for the cross-process import, and both halves
/// of it can be denied: the `OpenProcess` that reaches the exporting process,
/// and the `DuplicateHandle` that takes a handle out of it.
/// `PermissionDenied` is what the C ABI turns into that errno.
#[cfg(target_os = "windows")]
fn win32_error_kind(e: &windows::core::Error) -> std::io::ErrorKind {
    use std::io::ErrorKind;
    use windows::core::HRESULT;
    use windows::Win32::Foundation::{ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER};

    let code = e.code();
    if code == HRESULT::from_win32(ERROR_ACCESS_DENIED.0) {
        ErrorKind::PermissionDenied
    } else if code == HRESULT::from_win32(ERROR_INVALID_PARAMETER.0) {
        // What Windows answers for a pid no process holds: the blob outlived
        // its producer, or never named a real one. A handle value the named
        // process does not have answers `ERROR_INVALID_HANDLE` instead, which
        // falls to `Other` below -- the message carries the code either way.
        ErrorKind::NotFound
    } else {
        ErrorKind::Other
    }
}

/// Map an `OpenProcess` failure onto the crate error.
#[cfg(target_os = "windows")]
fn open_process_error(pid: u32, e: &windows::core::Error) -> crate::Error {
    use std::io::ErrorKind;

    let kind = win32_error_kind(e);
    let what = match kind {
        ErrorKind::PermissionDenied => {
            format!("cannot open exporting process {pid} to duplicate its handles: access denied")
        }
        ErrorKind::NotFound => format!("exporting process {pid} is gone"),
        _ => format!(
            "cannot open exporting process {pid} to duplicate its handles: {} \
             (HRESULT 0x{:08X})",
            e.message(),
            e.code().0 as u32
        ),
    };
    crate::Error::IoError(std::io::Error::new(kind, what))
}

/// One `DuplicateHandle` out of `src`, as an owned handle. `which` names the
/// handle in the error: a blob carries two, and they fail for different
/// reasons.
#[cfg(target_os = "windows")]
fn duplicate_out_of(
    src: &std::os::windows::io::OwnedHandle,
    handle: windows::Win32::Foundation::HANDLE,
    pid: u32,
    which: &str,
) -> crate::Result<std::os::windows::io::OwnedHandle> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use windows::Win32::Foundation::{DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE};
    use windows::Win32::System::Threading::GetCurrentProcess;

    let mut out = HANDLE(std::ptr::null_mut());
    // SAFETY: `src` is a live process handle opened for PROCESS_DUP_HANDLE,
    // `handle` is a value in that process, and `out` is a valid local.
    unsafe {
        DuplicateHandle(
            HANDLE(src.as_raw_handle()),
            handle,
            GetCurrentProcess(),
            &mut out,
            0,
            false,
            DUPLICATE_SAME_ACCESS,
        )
    }
    .map_err(|e| {
        // The same mapping `open_process_error` uses: this is the other half
        // of one operation, and a denied duplication is as much the `EACCES`
        // case spec section 8 names as a denied `OpenProcess` is.
        crate::Error::IoError(std::io::Error::new(
            win32_error_kind(&e),
            format!(
                "cannot duplicate the {which} handle out of exporting process {pid}: {} \
                 (HRESULT 0x{:08X})",
                e.message(),
                e.code().0 as u32
            ),
        ))
    })?;
    // SAFETY: `DuplicateHandle` succeeded, so `out` is a live handle this
    // process owns.
    Ok(unsafe { OwnedHandle::from_raw_handle(out.0) })
}

fn blit_inline_planes(
    t: &mut crate::TensorDyn,
    planes: &[BlobPlane<'_>],
    format: Option<crate::PixelFormat>,
    grid: &[usize],
    dtype: crate::DType,
) -> crate::Result<()> {
    let pin = t.pin_host(crate::CpuAccess::Write)?;
    let dst = unsafe { std::slice::from_raw_parts_mut(pin.as_mut_ptr(), pin.len()) };
    let geoms: Vec<crate::PlaneGeometry> = match format {
        Some(f) => {
            let (w, h_px) = dims_from_grid(f, grid)?;
            let rs = t
                .effective_row_stride()
                .unwrap_or(w * f.channels() * dtype.size());
            f.plane_table(w, h_px, rs)
                .ok_or_else(|| crate::Error::InvalidShape(format!("no plane table for {f:?}")))?
        }
        None => vec![crate::PlaneGeometry {
            offset: 0,
            stride: dst.len() as u64,
            size: dst.len() as u64,
        }],
    };
    if geoms.len() != planes.len() {
        return Err(crate::Error::InvalidArgument(format!(
            "blob carries {} planes but {:?} has {}",
            planes.len(),
            format,
            geoms.len()
        )));
    }
    for (g, p) in geoms.iter().zip(planes) {
        let start = g.offset as usize;
        let end = start
            .checked_add(p.data.len())
            .ok_or(crate::Error::InvalidSize(p.data.len()))?;
        let slot = dst
            .get_mut(start..end)
            .ok_or(crate::Error::InvalidSize(end))?;
        slot.copy_from_slice(p.data);
    }
    Ok(())
}

/// Rebuild `Colorimetry` from the carried strings. An unrecognised value is
/// dropped to `None` rather than refused: colorimetry is descriptive metadata,
/// and a consumer that does not know one axis can still use the pixels.
fn colorimetry_from(s: &BlobStrings<'_>) -> Option<crate::Colorimetry> {
    let c = crate::Colorimetry {
        space: crate::ColorSpace::from_str_code(s.color_space),
        transfer: crate::ColorTransfer::from_str_code(s.color_transfer),
        encoding: crate::ColorEncoding::from_str_code(s.color_encoding),
        range: crate::ColorRange::from_str_code(s.color_range),
    };
    if c == crate::Colorimetry::default() {
        None
    } else {
        Some(c)
    }
}

/// Rebuild a `Quantization` from the wire representation.
fn quantization_from(axis: i32, scales: &[f32], zeros: &[i32]) -> Option<crate::Quantization> {
    let first = *scales.first()?;
    match (axis, zeros.first()) {
        (-1, None) => Some(crate::Quantization::per_tensor_symmetric(first)),
        (-1, Some(z)) => Some(crate::Quantization::per_tensor(first, *z)),
        (a, _) if a >= 0 => {
            crate::Quantization::per_channel_symmetric(scales.to_vec(), a as usize).ok()
        }
        _ => None,
    }
}

/// Recover logical width and height from an addressing grid.
///
/// The inverse of [`PixelFormat::addressing_shape`](crate::PixelFormat::addressing_shape).
fn dims_from_grid(f: crate::PixelFormat, grid: &[usize]) -> crate::Result<(usize, usize)> {
    let bad = || crate::Error::InvalidShape(format!("grid {grid:?} is not an image for {f:?}"));
    match f.layout() {
        crate::PixelLayout::Planar => {
            if grid.len() < 3 {
                return Err(bad());
            }
            Ok((grid[2], grid[1]))
        }
        // Packed multi-channel is [h, w, c]; single-channel packed and
        // semi-planar are both [h, w].
        _ => {
            if grid.len() < 2 {
                return Err(bad());
            }
            Ok((grid[1], grid[0]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The half of the completion contract the exporter owns: a copy that
    /// exports no fence handle must export no value either, because `import`
    /// refuses that pair outright -- a degraded copy's blobs would be
    /// unimportable rather than merely completion-less.
    #[cfg(target_os = "windows")]
    #[test]
    fn a_reference_plane_drops_the_value_with_the_fence() {
        assert_eq!(super::reference_completion(0, 42), (0, 0));
        assert_eq!(super::reference_completion(0, 0), (0, 0));
        // A handle with nothing recorded is a legitimate pair: the importer
        // reads a zero value as "nothing to wait on".
        assert_eq!(super::reference_completion(0x1234, 0), (0x1234, 0));
        assert_eq!(super::reference_completion(0x1234, 42), (0x1234, 42));
    }

    #[test]
    fn every_region_offset_is_constant_time_arithmetic() {
        // The property that distinguishes the blob from CDR: no sequential walk.
        let h = BlobHeader {
            size: 4096,
            ndim: 3,
            strides_len: 3,
            quant_scales_len: 4,
            quant_zero_points_len: 4,
            strings_bytes: 24,
            plane_count: 2,
            planes_bytes: 128,
            ..BlobHeader::empty()
        };
        let r = region_offsets(&h).unwrap();
        assert_eq!(r.shape, HEADER_LEN);
        assert_eq!(r.strides, HEADER_LEN + 3 * 8);
        assert_eq!(r.quant_scales, HEADER_LEN + 3 * 8 + 3 * 8);
        assert_eq!(r.quant_zero_points, r.quant_scales + 4 * 4);
        assert_eq!(r.strings, r.quant_zero_points + 4 * 4);
        assert_eq!(r.planes, r.strings + 24);
        assert_eq!(r.end, r.planes + 128);
    }

    #[test]
    fn empty_strides_leave_the_region_zero_width() {
        let h = BlobHeader {
            size: 4096,
            ndim: 2,
            strides_len: 0,
            ..BlobHeader::empty()
        };
        let r = region_offsets(&h).unwrap();
        assert_eq!(r.strides, HEADER_LEN + 2 * 8);
        assert_eq!(
            r.quant_scales, r.strides,
            "densely-packed strides occupy no bytes"
        );
    }

    #[test]
    fn strides_must_be_ndim_entries_or_none() {
        // A partial stride array has no meaning; it is a forged length.
        let h = BlobHeader {
            size: 4096,
            ndim: 3,
            strides_len: 2,
            ..BlobHeader::empty()
        };
        assert!(matches!(
            region_offsets(&h),
            Err(BlobError::ForgedLength {
                field: "strides_len",
                ..
            })
        ));
    }

    #[test]
    fn a_forged_ndim_cannot_derive_an_offset_past_the_buffer() {
        let h = BlobHeader {
            size: 72,
            ndim: u32::MAX,
            strides_len: u32::MAX,
            ..BlobHeader::empty()
        };
        assert!(
            region_offsets(&h).is_err(),
            "ndim * 8 must be checked, not wrapped"
        );
    }

    #[test]
    fn regions_that_overrun_size_are_refused() {
        // Every derived offset is validated against `size`, which parse_header
        // has already bounded by the real buffer length.
        let h = BlobHeader {
            size: 80,
            ndim: 2,
            strides_len: 2,
            plane_count: 1,
            planes_bytes: 4096,
            ..BlobHeader::empty()
        };
        assert!(region_offsets(&h).is_err());
    }

    #[test]
    fn plane_count_is_bounded_by_the_planes_region_not_by_a_constant() {
        // A forged plane_count is a denial-of-service: it must be refused
        // against the bytes actually present, before any allocation.
        let h = BlobHeader {
            size: 4096,
            plane_count: u32::MAX,
            planes_bytes: 56,
            ..BlobHeader::empty()
        };
        assert!(matches!(
            region_offsets(&h),
            Err(BlobError::ForgedLength {
                field: "plane_count",
                ..
            })
        ));
    }

    /// Fill a tensor's bytes with a recognisable ramp so a content comparison
    /// can actually fail. An all-zero buffer round-trips through almost any bug.
    ///
    /// Bracketed with the coherency pair, which is free on `Mem` and one ioctl
    /// on a dma-buf but load-bearing for a D3D11 texture: the pin is a host
    /// shadow buffer there, so nothing written through it reaches the texture
    /// until the upload.
    fn fill_ramp(t: &TensorDyn) {
        let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin");
        t.sync_for_cpu(CpuAccess::ReadWrite).expect("sync for cpu");
        // SAFETY: the pin holds `pin.len()` writable bytes live for its own
        // lifetime, and this test is their only holder -- no map guard is
        // open and no device work is in flight across the two syncs.
        let s = unsafe { std::slice::from_raw_parts_mut(pin.as_mut_ptr(), pin.len()) };
        for (i, b) in s.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        t.sync_for_device(CpuAccess::ReadWrite)
            .expect("sync for device");
    }

    fn bytes_of(t: &TensorDyn) -> Vec<u8> {
        let pin = t.pin_host(CpuAccess::Read).expect("pin");
        t.sync_for_cpu(CpuAccess::Read).expect("sync for cpu");
        // SAFETY: as in `fill_ramp`; the pin holds `pin.len()` readable bytes
        // live and `sync_for_cpu` has just refreshed them.
        let bytes = unsafe { std::slice::from_raw_parts(pin.as_mut_ptr(), pin.len()) }.to_vec();
        t.sync_for_device(CpuAccess::Read).expect("sync for device");
        bytes
    }

    /// A reference-capable tensor, or `None` on a host that has no shareable
    /// backing (no dma-heap, or one this user cannot open).
    fn reference_capable(w: usize, h: usize) -> Option<TensorDyn> {
        use std::io::Write;
        match Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        ) {
            Ok(t) => Some(TensorDyn::from(t)),
            Err(e) => {
                let _ = writeln!(
                    std::io::stderr(),
                    "SKIP: no shareable backing on this host ({e:?}); \
                     reference transport not exercised"
                );
                None
            }
        }
    }

    #[test]
    fn a_reference_export_puts_an_index_in_the_handle_not_a_raw_fd() {
        let Some(src) = reference_capable(64, 48) else {
            return;
        };
        let (blob, fds) = export(&src, TransportMode::Reference).unwrap();
        let v = BlobView::parse(&blob).unwrap();
        let planes = v.planes().unwrap();
        for p in &planes {
            assert!(!p.is_inline());
        }
        if cfg!(target_os = "windows") {
            // A D3D11 texture is shared by NT handle, a value in this
            // process's handle table rather than a descriptor an out-of-band
            // table can carry, so the table is empty and every plane carries
            // the same handle bytes.
            assert!(fds.is_empty(), "an NT handle does not travel out of band");
            assert!(planes
                .iter()
                .all(|p| p.handle_bytes.len() == D3D11_HANDLE_BYTES));
        } else {
            assert!(!fds.is_empty(), "reference mode must carry an fd table");
            for p in &planes {
                assert!(
                    (p.handle as usize) < fds.len(),
                    "handle {} must index the {}-entry fd table, not be a raw fd",
                    p.handle,
                    fds.len()
                );
            }
            // NV12's two planes share one allocation, so they share one table
            // entry -- duplicating it would make the receiver dup the same
            // buffer twice and the offsets would stop referring to one buffer.
            assert_eq!(fds.len(), 1, "planes sharing a buffer share a table entry");
        }
        assert!(planes.iter().all(|p| p.handle == 0));
    }

    #[test]
    fn a_reference_round_trip_preserves_content_and_the_source_may_die() {
        let Some(src) = reference_capable(64, 48) else {
            return;
        };
        fill_ramp(&src);
        let expected = bytes_of(&src);
        let (blob, fds) = export(&src, TransportMode::Reference).unwrap();
        let src_id = src.buffer_identity().id();
        let got = import(&blob, &fds).unwrap();

        // The property that makes reference mode worth having: this is the
        // SAME buffer, not a copy that happens to hold equal bytes. `dup`
        // preserves the underlying object, so derived identity survives the
        // round trip -- which is also what lets the GL import cache hit on a
        // tensor that arrived from another process.
        //
        // Windows shares the same buffer but cannot say so through the
        // identity: every texture tensor keys on its own `ID3D11Texture2D*`,
        // and `OpenSharedResource1` mints a fresh object per open, so an
        // import never compares equal to what it was exported from
        // (`IdentityKind::D3d11Texture` says why a handle value would be the
        // unsound alternative). The byte comparison below is what carries the
        // property there.
        if !cfg!(target_os = "windows") {
            assert_eq!(
                got.buffer_identity().id(),
                src_id,
                "reference mode must share the buffer, not copy it"
            );
        }

        // Import dups, so dropping the producer must not invalidate the import.
        drop(src);
        assert_eq!(
            bytes_of(&got),
            expected,
            "the imported tensor sees the bytes"
        );
        assert_eq!(got.format(), Some(PixelFormat::Nv12));
    }

    #[test]
    fn an_fd_index_past_the_table_is_refused() {
        // The index arrives from an untrusted blob. An unbounded read here
        // would hand an arbitrary descriptor to dup.
        let Some(src) = reference_capable(64, 48) else {
            return;
        };
        let (blob, fds) = export(&src, TransportMode::Reference).unwrap();
        if cfg!(target_os = "windows") {
            // There is no fd table to index there. What arrives untrusted
            // instead is the pid and the two handle values, and neither may
            // be taken on faith: one names a process whose handle table this
            // import reaches into, the other is opened as a texture.
            let err = import(&patch_handle_word(&blob, 0, 0), &[])
                .expect_err("a null texture handle must be refused, not opened");
            assert!(
                matches!(err, crate::Error::InvalidArgument(_)),
                "expected InvalidArgument, got {err:?}"
            );

            // A fence value with no fence handle to read it on: the same
            // refusal the descriptor path makes for a completion nobody can
            // wait on. Both words are patched, since an exported tensor with
            // no recorded write carries value 0, where a missing fence handle
            // is simply unused.
            let named = patch_handle_word(&patch_handle_word(&blob, 1, 0), 2, 7);
            let err = import(&named, &[])
                .expect_err("a fence value with no fence handle must be refused");
            let msg = format!("{err}");
            assert!(
                matches!(err, crate::Error::InvalidArgument(_))
                    && msg.contains("cannot be waited on"),
                "expected the unwaitable-completion refusal, got {err:?}"
            );

            // A pid that does not exist: the import must fail to open it
            // rather than falling back on this process's handle table.
            let mut tampered = blob.clone();
            tampered[HEADER_PID_OFFSET..HEADER_PID_OFFSET + 4]
                .copy_from_slice(&u32::MAX.to_le_bytes());
            let err =
                import(&tampered, &[]).expect_err("a pid that does not exist must be refused");
            match &err {
                crate::Error::IoError(e) => assert_eq!(
                    e.kind(),
                    std::io::ErrorKind::NotFound,
                    "a pid no process holds is reported as gone, got {e:?}"
                ),
                other => panic!("expected IoError, got {other:?}"),
            }
            return;
        }
        let r = region_offsets(&BlobView::parse(&blob).unwrap().header()).unwrap();
        assert!(
            import(&blob, &[]).is_err(),
            "an empty table cannot satisfy an index"
        );
        let mut tampered = blob.clone();
        tampered[r.planes..r.planes + 8].copy_from_slice(&99i64.to_le_bytes());
        assert!(
            import(&tampered, &fds).is_err(),
            "an out-of-range index must be refused, not read"
        );
    }

    /// A copy of `blob` with one of the three handle words rewritten in
    /// *every* plane, so the tampering under test is the refusal that fires
    /// rather than the planes-disagree one.
    ///
    /// Every reference plane record is the same length -- the scalar block
    /// plus [`D3D11_HANDLE_BYTES`], padded to 8 -- which is what makes the
    /// stride computable without re-parsing.
    ///
    /// Compiled everywhere, because the Windows branch that calls it is a
    /// `cfg!` arm rather than a `#[cfg]` one.
    fn patch_handle_word(blob: &[u8], word: usize, value: u64) -> Vec<u8> {
        let v = BlobView::parse(blob).unwrap();
        let r = region_offsets(&v.header()).unwrap();
        let stride = (PLANE_RECORD_LEN + D3D11_HANDLE_BYTES).next_multiple_of(8);
        let mut out = blob.to_vec();
        for i in 0..v.header().plane_count as usize {
            let at = r.planes + i * stride + PLANE_RECORD_LEN + word * 8;
            out[at..at + 8].copy_from_slice(&value.to_le_bytes());
        }
        out
    }

    /// A handle value that is neither null nor a shared texture. The
    /// constructor must refuse it: the blob is untrusted input, and the one
    /// thing this import may not do is hand an arbitrary handle to the driver
    /// and carry on.
    #[test]
    #[cfg(target_os = "windows")]
    fn a_garbage_texture_handle_is_refused_by_the_constructor() {
        let Some(src) = reference_capable(64, 48) else {
            return;
        };
        let (blob, _) = export(&src, TransportMode::Reference).unwrap();
        let err = import(&patch_handle_word(&blob, 0, 0x1234), &[])
            .expect_err("a handle that is not a shared texture must be refused");
        let msg = format!("{err}");
        assert!(
            msg.contains("OpenSharedResource"),
            "the refusal must come from opening the handle, got: {msg}"
        );
    }

    /// The planes of one reference blob name one texture. A blob whose planes
    /// disagree is refused rather than read through the first of them.
    #[test]
    #[cfg(target_os = "windows")]
    fn planes_that_disagree_about_the_texture_are_refused() {
        let Some(src) = reference_capable(64, 48) else {
            return;
        };
        let (blob, _) = export(&src, TransportMode::Reference).unwrap();
        let v = BlobView::parse(&blob).unwrap();
        let r = region_offsets(&v.header()).unwrap();
        assert_eq!(v.planes().unwrap().len(), 2, "NV12 exports two planes");
        // The second record starts one padded record after the first.
        let second = r.planes + (PLANE_RECORD_LEN + D3D11_HANDLE_BYTES).next_multiple_of(8);
        let mut tampered = blob.clone();
        tampered[second + PLANE_RECORD_LEN..second + PLANE_RECORD_LEN + 8]
            .copy_from_slice(&0xBEEFu64.to_le_bytes());
        let err = import(&tampered, &[]).expect_err("planes naming two textures must be refused");
        let msg = format!("{err}");
        assert!(
            matches!(err, crate::Error::InvalidArgument(_)) && msg.contains("one texture"),
            "expected the one-texture refusal, got {err:?}"
        );
    }

    /// The pid offset the tamper test writes at is the one the writer emits.
    #[test]
    fn pid_offset_matches_the_writer() {
        let h = BlobHeader {
            size: HEADER_LEN as u64,
            pid: 0x0BAD_F00D,
            ..BlobHeader::empty()
        };
        let mut buf = vec![0u8; HEADER_LEN];
        h.write_to(&mut buf);
        assert_eq!(rd_u32(&buf, HEADER_PID_OFFSET), h.pid);
    }

    #[test]
    fn inline_round_trip_preserves_content_and_every_metadata_field() {
        let src = nv12_mem(64, 48).with_colorimetry(crate::Colorimetry::jfif());
        fill_ramp(&src);
        let blob = export(&src, TransportMode::Inline).unwrap().0;
        let got = import(&blob, &[]).unwrap();

        assert_eq!(got.dtype(), src.dtype());
        assert_eq!(got.format(), src.format());
        assert_eq!(got.colorimetry(), src.colorimetry());
        assert_eq!(got.shape(), src.shape(), "allocation geometry is rebuilt");
        assert_eq!(bytes_of(&got), bytes_of(&src), "every byte survived");
    }

    #[test]
    fn inline_import_is_a_new_allocation_and_the_source_may_die() {
        let src = nv12_mem(64, 48);
        fill_ramp(&src);
        let expected = bytes_of(&src);
        let blob = export(&src, TransportMode::Inline).unwrap().0;
        drop(src);
        let got = import(&blob, &[]).unwrap();
        assert_eq!(bytes_of(&got), expected);
    }

    #[test]
    fn a_bare_tensor_round_trips_including_byte_strides() {
        let src =
            TensorDyn::from(Tensor::<f32>::new(&[3, 8], Some(TensorMemory::Mem), None).unwrap());
        fill_ramp(&src);
        let blob = export(&src, TransportMode::Inline).unwrap().0;
        let got = import(&blob, &[]).unwrap();
        assert_eq!(got.shape(), &[3, 8]);
        assert_eq!(got.dtype(), crate::DType::F32);
        assert_eq!(got.format(), None);
        assert_eq!(bytes_of(&got), bytes_of(&src));
    }

    /// Assemble a minimal valid blob around a hand-written plane set.
    ///
    /// Tampering an exported blob is not good enough for the mode-mix test:
    /// flipping a plane's `handle` also leaves its bytes attached, which trips
    /// the separate "a referenced plane must not carry bytes" check, so the
    /// test would pass with the mode-mix guard removed entirely. Building the
    /// plane set directly isolates the one property under test.
    fn build_blob(planes: &[BlobPlane<'_>]) -> Vec<u8> {
        let strings = BlobStrings::default();
        let mut planes_buf = Vec::new();
        for p in planes {
            write_plane(p, &mut planes_buf);
        }
        let mut out = vec![0u8; HEADER_LEN];
        out.extend_from_slice(&1u64.to_le_bytes()); // shape [1]
        write_strings(&strings, &mut out);
        out.extend_from_slice(&planes_buf);
        let h = BlobHeader {
            size: out.len() as u64,
            planes_bytes: planes_buf.len() as u64,
            dtype: crate::DType::U8.code(),
            quant_axis: -2,
            ndim: 1,
            plane_count: planes.len() as u32,
            strings_bytes: strings_encoded_len(&strings) as u32,
            ..BlobHeader::empty()
        };
        h.write_to(&mut out);
        out
    }

    #[test]
    fn no_corrupted_header_value_can_panic_the_parser() {
        // The blanket property behind every individual forged-field test: a
        // panic on hostile input is a denial-of-service, so the parser must
        // return Err for anything it cannot handle. Walks every 4-byte word of
        // the header through several hostile values rather than checking the
        // handful of fields someone thought to name.
        let good = export(&nv12_mem(64, 48), TransportMode::Inline).unwrap().0;
        for word in 0..HEADER_LEN / 4 {
            for pattern in [u32::MAX, u32::MAX / 2, 1, 0x8000_0000] {
                let mut bad = good.clone();
                bad[word * 4..word * 4 + 4].copy_from_slice(&pattern.to_le_bytes());
                // Either it parses (the value happened to be benign) or it
                // errors. It must never panic, and `import` must not either.
                if let Ok(v) = BlobView::parse(&bad) {
                    let _ = v.shape();
                    let _ = v.strides();
                    let _ = v.strings();
                    let _ = v.planes();
                }
                let _ = import(&bad, &[]);
            }
        }
    }

    #[test]
    fn no_corrupted_plane_record_can_panic_the_parser() {
        let good = export(&nv12_mem(64, 48), TransportMode::Inline).unwrap().0;
        let r = region_offsets(&BlobView::parse(&good).unwrap().header()).unwrap();
        for word in 0..PLANE_RECORD_LEN / 4 {
            for pattern in [u32::MAX, u32::MAX / 2, 0x8000_0000] {
                let mut bad = good.clone();
                let at = r.planes + word * 4;
                bad[at..at + 4].copy_from_slice(&pattern.to_le_bytes());
                if let Ok(v) = BlobView::parse(&bad) {
                    let _ = v.planes();
                }
                let _ = import(&bad, &[]);
            }
        }
    }

    #[test]
    fn a_mixed_inline_and_referenced_plane_set_is_refused() {
        // The schemas validator's rule: all planes inline, or none. A frame
        // mixing modes has no coherent meaning, since one storage_kind, pid
        // and fence_fd cover every plane.
        //
        // Both planes here are individually well-formed -- the referenced one
        // carries no bytes, the inline one's size matches its data -- so the
        // ONLY thing wrong is that they disagree about transport mode.
        let payload = [1u8, 2, 3, 4];
        let referenced = BlobPlane {
            handle: 7,
            offset: 0,
            stride: 4,
            size: 4,
            used: 4,
            modifier: 0,
            handle_bytes: &[],
            data: &[],
        };
        let inlined = BlobPlane {
            handle: -1,
            offset: 0,
            stride: 4,
            size: payload.len() as u64,
            used: payload.len() as u64,
            modifier: 0,
            handle_bytes: &[],
            data: &payload,
        };
        let blob = build_blob(&[referenced, inlined]);
        let err =
            import(&blob, &[]).expect_err("a plane set mixing transport modes must be refused");
        assert!(
            err.to_string().contains("one transport mode"),
            "must fail on the mode mix, not incidentally: {err}"
        );
    }

    #[test]
    fn a_uniformly_referenced_plane_set_is_not_rejected_as_mixed() {
        // The control for the test above: two referenced planes must get past
        // the mode check (and then fail later, on unimplemented reference
        // import). Without this, a guard that rejected everything would look
        // correct.
        let referenced = BlobPlane {
            handle: 7,
            offset: 0,
            stride: 4,
            size: 4,
            used: 4,
            modifier: 0,
            handle_bytes: &[],
            data: &[],
        };
        let blob = build_blob(&[referenced, referenced]);
        let err = import(&blob, &[]).expect_err("reference-mode import is not implemented yet");
        assert!(
            !err.to_string().contains("one transport mode"),
            "a uniform plane set must clear the mode check: {err}"
        );
    }

    #[test]
    fn an_inline_plane_whose_size_disagrees_with_its_data_is_refused() {
        let blob = export(&nv12_mem(64, 48), TransportMode::Inline).unwrap().0;
        let mut tampered = blob.clone();
        let r = region_offsets(&BlobView::parse(&blob).unwrap().header()).unwrap();
        tampered[r.planes + 24..r.planes + 32].copy_from_slice(&9999u64.to_le_bytes());
        assert!(import(&tampered, &[]).is_err());
    }

    #[test]
    fn quantization_survives_the_round_trip() {
        let mut src =
            TensorDyn::from(Tensor::<i8>::new(&[3, 8], Some(TensorMemory::Mem), None).unwrap());
        src.set_quantization(crate::Quantization::per_tensor(0.5, -7))
            .unwrap();
        let blob = export(&src, TransportMode::Inline).unwrap().0;
        let got = import(&blob, &[]).unwrap();
        let q = got.quantization().expect("quantization survived");
        assert_eq!(q.scale(), &[0.5]);
        assert_eq!(q.zero_point(), Some(&[-7i32][..]));
    }

    use crate::{CpuAccess, PixelFormat, Tensor, TensorDyn, TensorMemory};

    fn nv12_mem(w: usize, h: usize) -> TensorDyn {
        TensorDyn::from(
            Tensor::<u8>::image(
                w,
                h,
                PixelFormat::Nv12,
                Some(TensorMemory::Mem),
                CpuAccess::ReadWrite,
            )
            .expect("alloc"),
        )
    }

    #[test]
    fn export_carries_the_addressing_grid_not_the_allocation() {
        let blob = export(&nv12_mem(640, 480), TransportMode::Inline)
            .unwrap()
            .0;
        let v = BlobView::parse(&blob).unwrap();
        // The grid, matching the schemas golden's shape for a 640x480 NV12.
        assert_eq!(v.shape(), &[480, 640]);
        assert_eq!(v.header().dtype, crate::DType::U8.code());
        assert_eq!(v.strings().format, "NV12");
    }

    #[test]
    fn export_emits_one_plane_per_plane_table_entry() {
        let blob = export(&nv12_mem(640, 480), TransportMode::Inline)
            .unwrap()
            .0;
        let v = BlobView::parse(&blob).unwrap();
        let planes = v.planes().unwrap();
        assert_eq!(planes.len(), 2, "NV12 is two planes");
        // Inline: `size` describes `data`, per the schemas validator's rule.
        assert_eq!(planes[0].size as usize, planes[0].data.len());
        assert_eq!(planes[1].size as usize, planes[1].data.len());
        assert_eq!(planes[0].size, 307_200);
        assert_eq!(planes[1].size, 153_600);
    }

    #[test]
    fn inline_export_clears_pid_and_fence_fd() {
        // Both are meaningless once the bytes travel: there is no handle to
        // reopen and nothing to wait on, because the producer already read it.
        let blob = export(&nv12_mem(64, 48), TransportMode::Inline).unwrap().0;
        let h = BlobView::parse(&blob).unwrap().header();
        assert_eq!(h.pid, 0, "no handle to reopen");
        assert_eq!(h.fence_fd, -1, "inlining means the producer already waited");
    }

    #[test]
    fn inline_planes_carry_no_modifier_and_no_handle_bytes() {
        // The schemas validator's rule for an inline plane.
        let blob = export(&nv12_mem(64, 48), TransportMode::Inline).unwrap().0;
        for p in BlobView::parse(&blob).unwrap().planes().unwrap() {
            assert!(p.is_inline());
            assert_eq!(p.modifier, 0);
            assert!(p.handle_bytes.is_empty());
            assert_eq!(p.size as usize, p.data.len());
        }
    }

    #[test]
    fn reference_export_of_a_mem_tensor_is_refused() {
        // `mem` has no shareable handle at all, so reference mode is not
        // expressible for it. Refusing beats silently inlining: the caller
        // asked for a reference and would otherwise get a copy without knowing.
        assert!(export(&nv12_mem(64, 48), TransportMode::Reference).is_err());
    }

    #[test]
    fn a_blob_parses_correctly_from_a_non_zero_offset() {
        // CameraFrame embeds a Tensor 8-aligned precisely so a consumer can
        // subslice and parse in place, without owning the containing buffer.
        let blob = export(&nv12_mem(64, 48), TransportMode::Inline).unwrap().0;
        let mut framed = vec![0xAAu8; 16];
        framed.extend_from_slice(&blob);
        let direct = BlobView::parse(&blob).unwrap();
        let framed_view = BlobView::parse(&framed[16..]).unwrap();
        assert_eq!(framed_view.header(), direct.header());
        assert_eq!(framed_view.shape(), direct.shape());
        assert_eq!(framed_view.strings(), direct.strings());
    }

    #[test]
    fn colorimetry_axes_are_carried_as_strings() {
        let t = nv12_mem(64, 48).with_colorimetry(crate::Colorimetry::jfif());
        let blob = export(&t, TransportMode::Inline).unwrap().0;
        let v = BlobView::parse(&blob).unwrap();
        let s = v.strings();
        assert_eq!(s.color_space, "srgb");
        assert_eq!(s.color_transfer, "srgb");
        assert_eq!(s.color_encoding, "bt601");
        assert_eq!(s.color_range, "full");
    }

    #[test]
    fn a_bare_tensor_carries_no_format_and_its_shape_is_its_grid() {
        let t =
            TensorDyn::from(Tensor::<f32>::new(&[3, 8], Some(TensorMemory::Mem), None).unwrap());
        let blob = export(&t, TransportMode::Inline).unwrap().0;
        let v = BlobView::parse(&blob).unwrap();
        assert_eq!(v.strings().format, "", "not an image");
        assert_eq!(v.shape(), &[3, 8]);
        assert_eq!(v.strides(), &[32i64, 4], "byte strides for f32");
    }

    fn ref_plane() -> BlobPlane<'static> {
        BlobPlane {
            handle: 7,
            offset: 0,
            stride: 640,
            size: 307_200,
            used: 307_200,
            modifier: 0,
            handle_bytes: &[],
            data: &[],
        }
    }

    #[test]
    fn a_plane_record_round_trips() {
        let mut out = Vec::new();
        write_plane(&ref_plane(), &mut out);
        let got = parse_planes(&out, 1).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], ref_plane());
    }

    #[test]
    fn a_plane_scalar_block_is_56_bytes() {
        // Matches the schemas spec's independently-stated TensorPlane minimum:
        // a 48-byte scalar block plus two 4-byte counts.
        let mut out = Vec::new();
        write_plane(&ref_plane(), &mut out);
        assert_eq!(out.len(), PLANE_RECORD_LEN);
        assert_eq!(PLANE_RECORD_LEN, 56);
    }

    #[test]
    fn every_plane_record_starts_8_aligned_whatever_the_payload() {
        // Records are read as u64s, so a payload of any length must be padded
        // before the next record begins.
        for n in 0..17usize {
            let payload: Vec<u8> = (0..n as u8).collect();
            let mut out = Vec::new();
            write_plane(
                &BlobPlane {
                    handle: -1,
                    size: n as u64,
                    data: &payload,
                    ..ref_plane()
                },
                &mut out,
            );
            write_plane(&ref_plane(), &mut out);
            assert_eq!(
                out.len() % 8,
                0,
                "payload of {n} bytes left the region unaligned"
            );
            let got = parse_planes(&out, 2).unwrap();
            assert_eq!(got[0].data, &payload[..], "payload {n} survived padding");
            assert_eq!(
                got[1],
                ref_plane(),
                "the second record parsed from an aligned offset"
            );
        }
    }

    #[test]
    fn plane_payloads_round_trip_independently() {
        // handle_bytes and data are separate regions; a length applied to the
        // wrong one would be invisible if either were empty.
        let hb = [0xDEu8, 0xAD, 0xBE, 0xEF];
        let data = [1u8, 2, 3];
        let mut out = Vec::new();
        write_plane(
            &BlobPlane {
                handle_bytes: &hb,
                data: &data,
                ..ref_plane()
            },
            &mut out,
        );
        let got = parse_planes(&out, 1).unwrap();
        assert_eq!(got[0].handle_bytes, &hb[..]);
        assert_eq!(got[0].data, &data[..]);
    }

    #[test]
    fn a_forged_plane_payload_length_is_refused() {
        let mut out = vec![0u8; PLANE_RECORD_LEN];
        out[48..52].copy_from_slice(&u32::MAX.to_le_bytes()); // handle_bytes_len
        assert!(matches!(
            parse_planes(&out, 1),
            Err(BlobError::ForgedLength { .. })
        ));
    }

    #[test]
    fn a_plane_count_larger_than_the_records_present_is_refused() {
        let mut out = Vec::new();
        write_plane(&ref_plane(), &mut out);
        assert!(
            parse_planes(&out, 2).is_err(),
            "claims two records, holds one"
        );
    }

    fn sample_strings() -> BlobStrings<'static> {
        BlobStrings {
            format: "NV12",
            color_space: "bt709",
            color_transfer: "bt709",
            color_encoding: "bt709",
            color_range: "limited",
        }
    }

    #[test]
    fn the_five_strings_round_trip_in_message_order() {
        let mut out = Vec::new();
        write_strings(&sample_strings(), &mut out);
        let got = parse_strings(&out).unwrap();
        assert_eq!(got, sample_strings());
    }

    #[test]
    fn string_order_is_positional_and_matches_the_message() {
        // Order is Tensor.msg's: format, then the four colour axes. Five
        // distinct values prove nothing is transposed -- with equal values a
        // swap would be invisible.
        let s = BlobStrings {
            format: "a",
            color_space: "bb",
            color_transfer: "ccc",
            color_encoding: "dddd",
            color_range: "eeeee",
        };
        let mut out = Vec::new();
        write_strings(&s, &mut out);
        assert_eq!(parse_strings(&out).unwrap(), s);
    }

    #[test]
    fn empty_strings_are_legal_and_distinct_from_absent() {
        // "" means "not an image" / "unspecified" -- a real value, not a null.
        let s = BlobStrings::default();
        let mut out = Vec::new();
        write_strings(&s, &mut out);
        assert_eq!(parse_strings(&out).unwrap(), s);
        assert!(
            !out.is_empty(),
            "five empty strings still occupy their counts"
        );
    }

    #[test]
    fn the_strings_region_is_padded_so_planes_start_8_aligned() {
        // Plane records are read as u64s; an unaligned record would fault or
        // force byte-at-a-time reads on every consumer.
        for f in ["", "a", "ab", "abc", "NV12", "rgb8_planar"] {
            let mut out = Vec::new();
            write_strings(
                &BlobStrings {
                    format: f,
                    ..Default::default()
                },
                &mut out,
            );
            assert_eq!(out.len() % 8, 0, "region for format {f:?} is not 8-aligned");
        }
    }

    #[test]
    fn a_non_utf8_string_is_refused() {
        // Strings come from a hostile buffer. A lossy substitution would hide
        // the malformed input rather than reporting it.
        let mut out = Vec::new();
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&[0xff, 0xfe]);
        for _ in 0..4 {
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        while out.len() % 8 != 0 {
            out.push(0);
        }
        assert!(matches!(
            parse_strings(&out),
            Err(BlobError::InvalidUtf8(_))
        ));
    }

    #[test]
    fn a_forged_string_length_cannot_read_past_the_region() {
        let mut out = Vec::new();
        out.extend_from_slice(&u32::MAX.to_le_bytes());
        out.extend_from_slice(b"NV12");
        while out.len() % 8 != 0 {
            out.push(0);
        }
        assert!(matches!(
            parse_strings(&out),
            Err(BlobError::ForgedLength { .. })
        ));
    }

    #[test]
    fn a_truncated_strings_region_is_refused() {
        assert!(parse_strings(&[0u8; 4]).is_err(), "needs five counts");
    }

    #[test]
    fn an_overrunning_region_is_attributed_to_its_own_field() {
        // Offsets increase monotonically, so the final `end <= size` check
        // already makes over-long regions *safe*. The per-step check exists for
        // a different reason: it names the field that was actually implausible.
        // Without it every overrun is reported as `planes_bytes`, and a caller
        // debugging a forged blob is sent to the wrong field.
        let h = BlobHeader {
            size: 4096,
            ndim: 2,
            strides_len: 2,
            strings_bytes: u32::MAX,
            ..BlobHeader::empty()
        };
        assert!(
            matches!(
                region_offsets(&h),
                Err(BlobError::ForgedLength {
                    field: "strings_bytes",
                    ..
                })
            ),
            "the overrun must name strings_bytes, not a later field"
        );
    }

    #[test]
    fn header_is_72_bytes_and_every_scalar_is_naturally_aligned() {
        assert_eq!(HEADER_LEN, 72);
        assert_eq!(HEADER_LEN % 8, 0, "the variable tail starts 8-aligned");
        for off in [0usize, 8, 16] {
            assert_eq!(off % 8, 0, "u64 field at {off} is not 8-aligned");
        }
        for off in [24usize, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64] {
            assert_eq!(off % 4, 0, "u32 field at {off} is not 4-aligned");
        }
    }

    #[test]
    fn a_header_round_trips_through_bytes() {
        let h = BlobHeader {
            size: 72,
            required_mask: 0,
            planes_bytes: 0,
            storage_kind: 2,
            pid: 4242,
            fence_fd: -1,
            dtype: 1,
            quant_axis: -2,
            ndim: 2,
            plane_count: 0,
            quant_scales_len: 0,
            quant_zero_points_len: 0,
            strings_bytes: 0,
            strides_len: 2,
        };
        let mut buf = vec![0u8; HEADER_LEN];
        h.write_to(&mut buf);
        assert_eq!(parse_header(&buf).unwrap(), h);
    }

    #[test]
    fn the_header_is_little_endian_on_every_host() {
        let mut buf = vec![0u8; HEADER_LEN];
        BlobHeader {
            size: 0x0102_0304_0506_0708,
            ..BlobHeader::empty()
        }
        .write_to(&mut buf);
        assert_eq!(
            &buf[0..8],
            &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
            "size must be little-endian regardless of host byte order"
        );
    }

    #[test]
    fn a_truncated_buffer_is_refused_not_read() {
        assert!(matches!(
            parse_header(&[0u8; 71]),
            Err(BlobError::Truncated { .. })
        ));
    }

    #[test]
    fn an_unknown_required_bit_is_refused() {
        let mut buf = vec![0u8; HEADER_LEN];
        BlobHeader {
            size: 72,
            required_mask: 1 << 63,
            ..BlobHeader::empty()
        }
        .write_to(&mut buf);
        assert!(matches!(
            parse_header(&buf),
            Err(BlobError::UnsupportedRequirement(_))
        ));
    }

    #[test]
    fn a_size_smaller_than_the_header_is_refused() {
        let mut buf = vec![0u8; HEADER_LEN];
        BlobHeader {
            size: 8,
            ..BlobHeader::empty()
        }
        .write_to(&mut buf);
        assert!(
            parse_header(&buf).is_err(),
            "size must cover at least the header"
        );
    }
}
