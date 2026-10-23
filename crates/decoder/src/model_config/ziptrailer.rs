// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Files appended to a model as a ZIP trailer: TFLite associated files and
//! the `edgefirst.json` / `labels.txt` that EdgeFirst tooling appends to HEF,
//! DVM and TensorRT engines.

use std::io::{Read, Seek, SeekFrom};

use super::ModelConfigError;

/// End of central directory signature, `PK\x05\x06`.
const EOCD_MAGIC: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
/// ZIP64 end of central directory locator signature.
const LOCATOR_MAGIC: [u8; 4] = [0x50, 0x4b, 0x06, 0x07];
/// ZIP64 end of central directory record signature.
const EOCD64_MAGIC: [u8; 4] = [0x50, 0x4b, 0x06, 0x06];
/// Central directory file header signature.
const CDFH_MAGIC: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
/// Fixed EOCD record size.
const EOCD_LEN: usize = 22;
/// Fixed EOCD record plus the largest comment.
const EOCD_SEARCH: u64 = EOCD_LEN as u64 + 65535;
/// ZIP64 locator size, and the ZIP64 record size without extensible data.
const LOCATOR_LEN: u64 = 20;
const EOCD64_LEN: u64 = 56;
/// Largest entry read as text.
const MAX_TEXT_ENTRY: u64 = 1 << 20;

/// The text entries of a ZIP trailer and where the model bytes end.
pub(crate) struct Trailer {
    /// Bytes before the first ZIP entry: the bare model.
    pub payload_len: u64,
    /// `(name, text)` for every UTF-8 entry of at most 1 MiB.
    pub entries: Vec<(String, String)>,
}

fn u16_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from(u16::from_le_bytes(
        b.get(at..at + 2)?.try_into().ok()?,
    )))
}

fn u32_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from(u32::from_le_bytes(
        b.get(at..at + 4)?.try_into().ok()?,
    )))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// Where the archive starts in a `len`-byte file whose last `tail.len()`
/// bytes are `tail`: found from an EOCD record that ends exactly at the end
/// of the file and a central directory that starts with its signature, so
/// nothing else of the file is read. `None` when there is no such archive.
fn archive_start<R: Read + Seek>(r: &mut R, len: u64, tail: &[u8]) -> Option<u64> {
    let tail_start = len - tail.len() as u64;
    let at = (0..=tail.len().checked_sub(EOCD_LEN)?).rev().find(|&i| {
        tail[i..i + 4] == EOCD_MAGIC
            && u16_at(tail, i + 20).map(|c| i as u64 + EOCD_LEN as u64 + c)
                == Some(tail.len() as u64)
    })?;
    if u16_at(tail, at + 4)? != 0 || u16_at(tail, at + 6)? != 0 {
        return None;
    }
    let eocd = tail_start + at as u64;
    let (cd_size, cd_offset) = (u32_at(tail, at + 12)?, u32_at(tail, at + 16)?);
    let (cd_end, cd_size, cd_offset) =
        if cd_offset == u64::from(u32::MAX) || cd_size == u64::from(u32::MAX) {
            // ZIP64: the locator sits just before the EOCD, the record before it.
            let locator = eocd.checked_sub(LOCATOR_LEN)?;
            let record = locator.checked_sub(EOCD64_LEN)?;
            let mut b = [0u8; (EOCD64_LEN + LOCATOR_LEN) as usize];
            r.seek(SeekFrom::Start(record)).ok()?;
            r.read_exact(&mut b).ok()?;
            let (rec, loc) = b.split_at(EOCD64_LEN as usize);
            if loc[..4] != LOCATOR_MAGIC || rec[..4] != EOCD64_MAGIC {
                return None;
            }
            (record, u64_at(rec, 40)?, u64_at(rec, 48)?)
        } else {
            (eocd, cd_size, cd_offset)
        };
    let cd_start = cd_end.checked_sub(cd_size)?;
    let start = cd_start.checked_sub(cd_offset)?;
    let mut magic = [0u8; 4];
    r.seek(SeekFrom::Start(cd_start)).ok()?;
    r.read_exact(&mut magic).ok()?;
    (magic == CDFH_MAGIC).then_some(start)
}

/// The bytes of `inner` from `start` on, as a reader of their own.
struct Window<R> {
    inner: R,
    start: u64,
    len: u64,
    pos: u64,
}

impl<R: Read + Seek> Read for Window<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.len.saturating_sub(self.pos);
        let n = buf.len().min(usize::try_from(left).unwrap_or(usize::MAX));
        self.inner.seek(SeekFrom::Start(self.start + self.pos))?;
        let n = self.inner.read(&mut buf[..n])?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl<R: Read + Seek> Seek for Window<R> {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let pos = match to {
            SeekFrom::Start(n) => Some(n),
            SeekFrom::End(d) => self.len.checked_add_signed(d),
            SeekFrom::Current(d) => self.pos.checked_add_signed(d),
        };
        self.pos = pos.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek before the archive")
        })?;
        Ok(self.pos)
    }
}

/// The ZIP trailer of `r` (`len` bytes long), or `None` when it has none.
/// A malformed trailer is treated as none: model bytes may contain the
/// signature by chance. Only the last 64 KiB, the central directory and the
/// text entries are read.
pub(crate) fn read_trailer<R: Read + Seek>(
    r: &mut R,
    len: u64,
) -> Result<Option<Trailer>, ModelConfigError> {
    let io = |e: std::io::Error| ModelConfigError::Container(format!("ZIP trailer: {e}"));
    let tail_len = len.min(EOCD_SEARCH);
    if tail_len < EOCD_LEN as u64 {
        return Ok(None);
    }
    r.seek(SeekFrom::Start(len - tail_len)).map_err(io)?;
    let mut tail = vec![0u8; tail_len as usize];
    r.read_exact(&mut tail).map_err(io)?;
    let Some(start) = archive_start(r, len, &tail) else {
        return Ok(None);
    };
    let window = Window {
        inner: &mut *r,
        start,
        len: len - start,
        pos: 0,
    };
    let config = zip::read::Config {
        archive_offset: zip::read::ArchiveOffset::Known(0),
    };
    let Ok(mut archive) = zip::ZipArchive::with_config(config, window) else {
        log::debug!("model file ends with a ZIP directory that does not read as an archive");
        return Ok(None);
    };
    let mut first = u64::MAX;
    for i in 0..archive.len() {
        if let Ok(file) = archive.by_index_raw(i) {
            first = first.min(file.header_start());
        }
    }
    if first == u64::MAX {
        return Ok(None);
    }
    let mut entries = Vec::new();
    for i in 0..archive.len() {
        let Ok(mut file) = archive.by_index(i) else {
            continue;
        };
        if file.is_dir() || file.size() > MAX_TEXT_ENTRY || file.compressed_size() > MAX_TEXT_ENTRY
        {
            log::debug!("ZIP entry {} skipped", file.name());
            continue;
        }
        let name = file.name().to_owned();
        // The declared size is not trusted: inflation stops past the limit.
        let mut bytes = Vec::new();
        let read = (&mut file).take(MAX_TEXT_ENTRY + 1).read_to_end(&mut bytes);
        if read.is_err() || bytes.len() as u64 > MAX_TEXT_ENTRY {
            continue;
        }
        if let Ok(text) = String::from_utf8(bytes) {
            entries.push((name, text));
        }
    }
    Ok(Some(Trailer {
        payload_len: start + first,
        entries,
    }))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    /// `payload` followed by a ZIP holding `files`, with offsets absolute
    /// from the start of the combined bytes.
    pub(crate) fn with_trailer(payload: &[u8], files: &[(&str, &str)]) -> Vec<u8> {
        let mut cursor = Cursor::new(payload.to_vec());
        cursor.set_position(payload.len() as u64);
        let mut zip = zip::ZipWriter::new(cursor);
        for (name, text) in files {
            zip.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(text.as_bytes()).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    fn read(bytes: &[u8]) -> Option<Trailer> {
        read_trailer(&mut Cursor::new(bytes), bytes.len() as u64).unwrap()
    }

    #[test]
    fn zip_trailer_entries_after_payload() {
        let payload = vec![0xabu8; 1000];
        let bytes = with_trailer(
            &payload,
            &[("edgefirst.json", "{}"), ("labels.txt", "a\nb\n")],
        );
        let t = read(&bytes).expect("trailer");
        assert_eq!(t.payload_len, 1000);
        assert_eq!(t.entries.len(), 2);
        assert!(t.entries.contains(&("labels.txt".into(), "a\nb\n".into())));
    }

    /// A standalone ZIP concatenated onto a model (`cat model zip`) has
    /// offsets relative to itself.
    #[test]
    fn concatenated_zip_reports_its_start() {
        let zip = with_trailer(&[], &[("labels.txt", "a\n")]);
        let mut bytes = vec![0x5au8; 777];
        bytes.extend_from_slice(&zip);
        let t = read(&bytes).expect("trailer");
        assert_eq!(t.payload_len, 777);
        assert_eq!(t.entries, [("labels.txt".to_string(), "a\n".to_string())]);
    }

    /// An entry whose stream inflates past its declared size is dropped
    /// without inflating it all.
    #[test]
    fn oversized_entries_are_skipped() {
        let big = "x".repeat((MAX_TEXT_ENTRY + 10) as usize);
        let mut cursor = Cursor::new(vec![0u8; 16]);
        cursor.set_position(16);
        let mut zip = zip::ZipWriter::new(cursor);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("big.txt", opts).unwrap();
        zip.write_all(big.as_bytes()).unwrap();
        zip.start_file("small.txt", opts).unwrap();
        zip.write_all(b"ok").unwrap();
        let bytes = zip.finish().unwrap().into_inner();
        let t = read(&bytes).expect("trailer");
        assert_eq!(t.entries, [("small.txt".to_string(), "ok".to_string())]);
    }

    #[test]
    fn zip_trailer_with_out_of_bounds_cd_offset_is_ignored() {
        let mut bytes = vec![0xaau8; 64];
        bytes.extend_from_slice(&EOCD_MAGIC);
        bytes.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0, 46, 0, 0, 0]);
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&[0, 0]);
        assert!(read(&bytes).is_none());
    }

    /// Reads through it, counting the bytes.
    struct Counting<R> {
        inner: R,
        read: u64,
    }

    impl<R: Read> Read for Counting<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(buf)?;
            self.read += n as u64;
            Ok(n)
        }
    }

    impl<R: Seek> Seek for Counting<R> {
        fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(to)
        }
    }

    fn bytes_read(bytes: Vec<u8>) -> (bool, u64) {
        let len = bytes.len() as u64;
        let mut r = Counting {
            inner: Cursor::new(bytes),
            read: 0,
        };
        let found = read_trailer(&mut r, len).unwrap().is_some();
        (found, r.read)
    }

    /// Neither a concatenated archive nor a chance signature makes the
    /// reader scan the model bytes.
    #[test]
    fn trailer_detection_reads_only_the_tail() {
        const MODEL: usize = 8 << 20;
        let zip = with_trailer(&[], &[("labels.txt", "a\n")]);
        let mut concatenated = vec![0x5au8; MODEL];
        concatenated.extend_from_slice(&zip);
        let (found, read) = bytes_read(concatenated);
        assert!(found);
        assert!(read < 200 << 10, "read {read} bytes");

        let mut fake = vec![0x5au8; MODEL];
        fake[MODEL - 100..MODEL - 96].copy_from_slice(&EOCD_MAGIC);
        let (found, read) = bytes_read(fake);
        assert!(!found);
        assert!(read <= EOCD_SEARCH + 64, "read {read} bytes");
    }

    /// An entry that cannot be opened still marks where the archive starts.
    #[test]
    fn payload_ends_at_the_first_entry_even_if_unreadable() {
        let mut cursor = Cursor::new(vec![0u8; 100]);
        cursor.set_position(100);
        let mut zip = zip::ZipWriter::new(cursor);
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file("first.bin", stored).unwrap();
        zip.write_all(b"x").unwrap();
        zip.start_file("labels.txt", stored).unwrap();
        zip.write_all(b"a\n").unwrap();
        let mut bytes = zip.finish().unwrap().into_inner();
        // Mark the first entry's method unsupported in its local and central headers.
        bytes[100 + 8] = 77;
        let cd = bytes.windows(4).position(|w| w == CDFH_MAGIC).unwrap();
        bytes[cd + 10] = 77;
        let t = read(&bytes).expect("trailer");
        assert_eq!(t.payload_len, 100);
        assert_eq!(t.entries, [("labels.txt".to_string(), "a\n".to_string())]);
    }

    #[test]
    fn no_trailer_no_metadata() {
        assert!(read(&[0u8; 4096]).is_none());
        assert!(read(&[1, 2, 3]).is_none());
        assert!(read(&[]).is_none());
    }
}
