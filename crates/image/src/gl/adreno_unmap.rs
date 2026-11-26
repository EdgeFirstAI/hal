// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Mapping bookkeeping for the Adreno EGL stack on Linux.
//!
//! On aarch64 Linux with Qualcomm's Adreno driver, [`install`] points the
//! `mmap`/`mmap64`/`munmap` GOT slots of `msm_gbm.so` and the `munmap` slot
//! of each loaded `libeglSubDriver*.so` at the hooks below. The hooks record
//! the ranges `msm_gbm.so` maps. A subdriver `munmap` of such a range unmaps
//! it and marks it released for the calling thread; the next `msm_gbm.so`
//! `munmap` of that range on the same thread consumes the mark and returns 0
//! without unmapping. An `msm_gbm.so` `mmap` that returns the range on that
//! thread clears the mark. Every other call goes straight to libc.

use std::collections::HashSet;
use std::ffi::{c_int, c_void, CStr};
use std::sync::{Mutex, Once};

/// `msm_gbm.so` mapping state, guarded by [`MAPS`].
#[derive(Default)]
struct GbmMaps {
    /// The `(address, length)` of every live mapping `msm_gbm.so` made.
    held: HashSet<(usize, usize)>,
    /// Ranges a subdriver unmapped while `msm_gbm.so` held them, keyed with
    /// the calling thread's id.
    released: HashSet<((usize, usize), libc::pid_t)>,
}

static MAPS: Mutex<Option<GbmMaps>> = Mutex::new(None);

/// Runs `f` with [`MAPS`] locked.
fn with_maps<R>(f: impl FnOnce(&mut GbmMaps) -> R) -> R {
    let mut guard = MAPS.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(GbmMaps::default))
}

unsafe extern "C" fn gbm_mmap(
    addr: *mut c_void,
    len: usize,
    prot: c_int,
    flags: c_int,
    fd: c_int,
    offset: libc::off_t,
) -> *mut c_void {
    // SAFETY: forwards the caller's arguments unchanged.
    let mapped = unsafe { libc::mmap(addr, len, prot, flags, fd, offset) };
    if mapped != libc::MAP_FAILED {
        let range = (mapped as usize, len);
        // SAFETY: `gettid` has no preconditions.
        let tid = unsafe { libc::gettid() };
        with_maps(|maps| {
            maps.released.remove(&(range, tid));
            maps.held.insert(range);
        });
    }
    mapped
}

unsafe extern "C" fn gbm_munmap(addr: *mut c_void, len: usize) -> c_int {
    let range = (addr as usize, len);
    // SAFETY: `gettid` has no preconditions.
    let tid = unsafe { libc::gettid() };
    with_maps(|maps| {
        if maps.released.remove(&(range, tid)) {
            return 0;
        }
        maps.held.remove(&range);
        // SAFETY: forwards the caller's arguments unchanged.
        unsafe { libc::munmap(addr, len) }
    })
}

unsafe extern "C" fn subdriver_munmap(addr: *mut c_void, len: usize) -> c_int {
    let range = (addr as usize, len);
    // SAFETY: `gettid` has no preconditions.
    let tid = unsafe { libc::gettid() };
    with_maps(|maps| {
        // SAFETY: forwards the caller's arguments unchanged.
        let result = unsafe { libc::munmap(addr, len) };
        if result == 0 && maps.held.remove(&range) {
            maps.released.insert((range, tid));
        }
        result
    })
}

/// Whether `path` names Qualcomm's `msm_gbm.so`.
fn is_msm_gbm(path: &str) -> bool {
    file_name(path) == "msm_gbm.so"
}

/// Whether `path` names a Qualcomm `libeglSubDriver*.so`.
fn is_egl_subdriver(path: &str) -> bool {
    let name = file_name(path);
    name.starts_with("libeglSubDriver") && name.contains(".so")
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// `Ok` when [`install`] patches for a context with this `GL_VENDOR` and
/// `GL_RENDERER` and these loaded object paths: a Qualcomm vendor, an Adreno
/// renderer, and both `msm_gbm.so` and a `libeglSubDriver*.so` loaded.
/// Otherwise `Err` with the first condition that does not hold.
fn patch_decision(vendor: &str, renderer: &str, objects: &[&str]) -> Result<(), &'static str> {
    if !vendor.to_ascii_lowercase().contains("qualcomm") {
        return Err("GL_VENDOR is not Qualcomm");
    }
    if !renderer.to_ascii_lowercase().contains("adreno") {
        return Err("GL_RENDERER is not Adreno");
    }
    if !objects.iter().any(|o| is_msm_gbm(o)) {
        return Err("msm_gbm.so is not loaded");
    }
    if !objects.iter().any(|o| is_egl_subdriver(o)) {
        return Err("no libeglSubDriver*.so is loaded");
    }
    Ok(())
}

/// Patches the GOT slots described in the module documentation, once per
/// process, on aarch64 when [`patch_decision`] accepts `vendor`, `renderer`
/// and the loaded objects. Logs at debug what it patched, or why it patched
/// nothing.
pub(in crate::opengl_headless) fn install(vendor: &str, renderer: &str) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if !cfg!(target_arch = "aarch64") {
            log::debug!("Adreno mapping bookkeeping: not installed (not aarch64)");
            return;
        }
        let mut objects = Vec::new();
        for_each_object(|name, _| objects.push(name.to_owned()));
        let names: Vec<&str> = objects.iter().map(String::as_str).collect();
        if let Err(why) = patch_decision(vendor, renderer, &names) {
            log::debug!("Adreno mapping bookkeeping: not installed ({why})");
            return;
        }
        let mut gbm = 0usize;
        let mut subdriver = 0usize;
        for_each_object(|name, info| {
            if is_msm_gbm(name) {
                gbm += patch_slots(info, "mmap", gbm_mmap as *const () as usize);
                gbm += patch_slots(info, "mmap64", gbm_mmap as *const () as usize);
                gbm += patch_slots(info, "munmap", gbm_munmap as *const () as usize);
            } else if is_egl_subdriver(name) {
                subdriver += patch_slots(info, "munmap", subdriver_munmap as *const () as usize);
            }
        });
        log::debug!(
            "Adreno mapping bookkeeping: {gbm} msm_gbm.so slot(s), {subdriver} subdriver slot(s)"
        );
    });
}

/// Calls `f` with the path and `dl_phdr_info` of each loaded object.
fn for_each_object(mut f: impl FnMut(&str, &libc::dl_phdr_info)) {
    unsafe extern "C" fn visit(
        info: *mut libc::dl_phdr_info,
        _size: usize,
        data: *mut c_void,
    ) -> c_int {
        // SAFETY: `data` is the `&mut dyn FnMut` passed below and `info` is
        // valid for the duration of the callback.
        let (f, info) = unsafe {
            (
                &mut *(data as *mut &mut dyn FnMut(&str, &libc::dl_phdr_info)),
                &*info,
            )
        };
        if !info.dlpi_name.is_null() {
            // SAFETY: `dlpi_name` is a NUL-terminated string owned by ld.so.
            let name = unsafe { CStr::from_ptr(info.dlpi_name) };
            if let Ok(name) = name.to_str() {
                f(name, info);
            }
        }
        0
    }
    let mut f: &mut dyn FnMut(&str, &libc::dl_phdr_info) = &mut f;
    // SAFETY: `visit` only reads `info` and calls `f`.
    unsafe {
        libc::dl_iterate_phdr(Some(visit), &mut f as *mut _ as *mut c_void);
    }
}

const PT_DYNAMIC: u32 = 2;
const PT_GNU_RELRO: u32 = 0x6474_e552;
const DT_NULL: i64 = 0;
const DT_PLTRELSZ: i64 = 2;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_RELA: i64 = 7;
const DT_RELASZ: i64 = 8;
const DT_JMPREL: i64 = 23;
const R_AARCH64_GLOB_DAT: u32 = 1025;
const R_AARCH64_JUMP_SLOT: u32 = 1026;

#[repr(C)]
struct Elf64Dyn {
    tag: i64,
    val: u64,
}

#[repr(C)]
struct Elf64Rela {
    offset: u64,
    info: u64,
    addend: i64,
}

#[repr(C)]
struct Elf64Sym {
    name: u32,
    info: u8,
    other: u8,
    shndx: u16,
    value: u64,
    size: u64,
}

/// Points every `R_AARCH64_JUMP_SLOT` and `R_AARCH64_GLOB_DAT` slot of
/// `symbol` in the object `info` describes at `target`. Returns the number of
/// slots patched.
fn patch_slots(info: &libc::dl_phdr_info, symbol: &str, target: usize) -> usize {
    if !cfg!(target_arch = "aarch64") {
        return 0;
    }
    let base = info.dlpi_addr as usize;
    // SAFETY: ld.so guarantees `dlpi_phdr` points at `dlpi_phnum` headers.
    let phdrs = unsafe { std::slice::from_raw_parts(info.dlpi_phdr, info.dlpi_phnum as usize) };
    let Some(dynamic) = phdrs.iter().find(|p| p.p_type == PT_DYNAMIC) else {
        return 0;
    };
    let relro = phdrs.iter().find(|p| p.p_type == PT_GNU_RELRO).map(|p| {
        (
            base + p.p_vaddr as usize,
            base + (p.p_vaddr + p.p_memsz) as usize,
        )
    });
    let abs = |v: u64| {
        let v = v as usize;
        if v < base {
            base + v
        } else {
            v
        }
    };
    let (mut strtab, mut symtab) = (0usize, 0usize);
    let mut tables = [(0usize, 0usize); 2];
    let mut entry = (base + dynamic.p_vaddr as usize) as *const Elf64Dyn;
    loop {
        // SAFETY: the dynamic section is a `DT_NULL`-terminated array.
        let d = unsafe { &*entry };
        match d.tag {
            DT_NULL => break,
            DT_STRTAB => strtab = abs(d.val),
            DT_SYMTAB => symtab = abs(d.val),
            DT_JMPREL => tables[0].0 = abs(d.val),
            DT_PLTRELSZ => tables[0].1 = d.val as usize,
            DT_RELA => tables[1].0 = abs(d.val),
            DT_RELASZ => tables[1].1 = d.val as usize,
            _ => {}
        }
        // SAFETY: the loop stops at `DT_NULL`, which ends the array.
        entry = unsafe { entry.add(1) };
    }
    if strtab == 0 || symtab == 0 {
        return 0;
    }
    let mut patched = 0;
    for (table, size) in tables {
        if table == 0 {
            continue;
        }
        let count = size / std::mem::size_of::<Elf64Rela>();
        // SAFETY: the table holds `count` relocations.
        let relas = unsafe { std::slice::from_raw_parts(table as *const Elf64Rela, count) };
        for rela in relas {
            let kind = (rela.info & 0xffff_ffff) as u32;
            if kind != R_AARCH64_JUMP_SLOT && kind != R_AARCH64_GLOB_DAT {
                continue;
            }
            let index = (rela.info >> 32) as usize;
            // SAFETY: `index` names an entry of the object's symbol table, and
            // `st_name` an offset into its string table.
            let name = unsafe {
                let sym = &*((symtab as *const Elf64Sym).add(index));
                CStr::from_ptr((strtab + sym.name as usize) as *const libc::c_char)
            };
            if name.to_bytes() != symbol.as_bytes() {
                continue;
            }
            let slot = base + rela.offset as usize;
            // SAFETY: `slot` is a GOT entry of a loaded object.
            if unsafe { write_slot(slot, target, relro) } {
                patched += 1;
            }
        }
    }
    patched
}

/// Writes `target` into the GOT entry at `slot`, making its page writable for
/// the write and read-only again when the slot lies in `relro`. Returns
/// whether the write happened.
///
/// # Safety
/// `slot` must be the address of a pointer-sized GOT entry.
unsafe fn write_slot(slot: usize, target: usize, relro: Option<(usize, usize)>) -> bool {
    // SAFETY: `sysconf` has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let start = slot & !(page - 1);
    let end = (slot + std::mem::size_of::<usize>() + page - 1) & !(page - 1);
    // SAFETY: the pages hold the object's GOT, which stays mapped for the
    // object's lifetime.
    unsafe {
        if libc::mprotect(
            start as *mut c_void,
            end - start,
            libc::PROT_READ | libc::PROT_WRITE,
        ) != 0
        {
            return false;
        }
        std::ptr::write_volatile(slot as *mut usize, target);
        if relro.is_some_and(|(lo, hi)| slot >= lo && slot < hi) {
            libc::mprotect(start as *mut c_void, end - start, libc::PROT_READ);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{gbm_mmap, gbm_munmap, patch_decision, subdriver_munmap};
    use std::ffi::c_void;

    const LEN: usize = 16384;

    fn gbm_map() -> *mut c_void {
        // SAFETY: an anonymous private mapping.
        let p = unsafe {
            gbm_mmap(
                std::ptr::null_mut(),
                LEN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(p, libc::MAP_FAILED);
        p
    }

    fn mapped(p: *mut c_void) -> bool {
        let mut pages = [0u8; LEN / 4096];
        // SAFETY: `pages` holds one byte per page of the range.
        unsafe { libc::mincore(p, LEN, pages.as_mut_ptr()) == 0 }
    }

    /// After a subdriver unmaps a gbm mapping, gbm's unmap of the same range
    /// leaves a mapping made there in between in place.
    #[test]
    fn gbm_unmap_after_a_subdriver_unmap_leaves_a_new_mapping_alone() {
        let p = gbm_map();
        // SAFETY: `p` is the mapping made above.
        assert_eq!(unsafe { subdriver_munmap(p, LEN) }, 0);
        assert!(!mapped(p), "the subdriver unmap goes through");
        // SAFETY: a fixed anonymous mapping over the range just unmapped.
        let q = unsafe {
            libc::mmap(
                p,
                LEN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
        };
        let occupied = mapped(p);
        // SAFETY: the range gbm mapped.
        assert_eq!(unsafe { gbm_munmap(p, LEN) }, 0);
        assert_eq!(
            mapped(p),
            occupied,
            "gbm's second unmap leaves whatever was mapped there since in place"
        );
        if q == p {
            // SAFETY: `q` is the mapping made above.
            unsafe { libc::munmap(q, LEN) };
        }
    }

    /// A gbm mapping the subdriver never unmapped is unmapped by gbm.
    #[test]
    fn gbm_unmaps_its_own_mapping() {
        let p = gbm_map();
        // SAFETY: `p` is the mapping made above.
        assert_eq!(unsafe { gbm_munmap(p, LEN) }, 0);
        assert!(!mapped(p));
    }

    /// A subdriver unmap of a range gbm did not map goes through and leaves
    /// no mark.
    #[test]
    fn subdriver_unmaps_of_other_ranges_go_through() {
        // SAFETY: an anonymous private mapping.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                LEN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(p, libc::MAP_FAILED);
        // SAFETY: `p` is the mapping made above.
        assert_eq!(unsafe { subdriver_munmap(p, LEN) }, 0);
        assert!(!mapped(p));
        // SAFETY: `gettid` has no preconditions.
        let tid = unsafe { libc::gettid() };
        assert!(super::with_maps(|maps| !maps
            .released
            .contains(&((p as usize, LEN), tid))));
    }

    /// A released mark belongs to the releasing thread, and gbm's unmap on
    /// that thread consumes it.
    #[test]
    fn released_marks_belong_to_the_releasing_thread() {
        let p = gbm_map();
        // SAFETY: `p` is the mapping made above.
        assert_eq!(unsafe { subdriver_munmap(p, LEN) }, 0);
        let addr = p as usize;
        std::thread::spawn(move || {
            // SAFETY: `gettid` has no preconditions.
            let tid = unsafe { libc::gettid() };
            assert!(super::with_maps(|maps| !maps
                .released
                .contains(&((addr, LEN), tid))));
        })
        .join()
        .unwrap();
        // SAFETY: the range gbm mapped.
        assert_eq!(unsafe { gbm_munmap(p, LEN) }, 0);
        // SAFETY: `gettid` has no preconditions.
        let tid = unsafe { libc::gettid() };
        assert!(super::with_maps(|maps| !maps
            .released
            .contains(&((addr, LEN), tid))));
    }

    const GBM: &str = "/usr/lib/aarch64-linux-gnu/gbm/msm_gbm.so";
    const SUBDRIVER: &str = "/lib/aarch64-linux-gnu/libeglSubDriverWayland.so.1";
    const OTHER: &str = "/lib/aarch64-linux-gnu/libc.so.6";

    /// Qualcomm's Adreno driver with both libraries loaded is patched.
    #[test]
    fn qualcomm_adreno_with_both_libraries_is_patched() {
        assert_eq!(
            patch_decision("Qualcomm", "Adreno (TM) 663", &[OTHER, GBM, SUBDRIVER]),
            Ok(())
        );
    }

    /// Mesa freedreno is not patched.
    #[test]
    fn freedreno_is_not_patched() {
        let mesa_gbm = "/usr/lib/aarch64-linux-gnu/gbm/dri_gbm.so";
        assert!(patch_decision("freedreno", "FD663", &[OTHER, mesa_gbm]).is_err());
        assert!(patch_decision("freedreno", "Adreno (TM) 663", &[OTHER, GBM, SUBDRIVER]).is_err());
    }

    /// Qualcomm with either library missing is not patched.
    #[test]
    fn qualcomm_with_a_library_missing_is_not_patched() {
        assert!(patch_decision("Qualcomm", "Adreno (TM) 663", &[OTHER, GBM]).is_err());
        assert!(patch_decision("Qualcomm", "Adreno (TM) 663", &[OTHER, SUBDRIVER]).is_err());
        assert!(patch_decision("Qualcomm", "Adreno (TM) 663", &[OTHER]).is_err());
    }

    /// A renderer other than Adreno is not patched.
    #[test]
    fn a_non_adreno_renderer_is_not_patched() {
        assert!(patch_decision("Qualcomm", "Mali-G310", &[OTHER, GBM, SUBDRIVER]).is_err());
        assert!(patch_decision("ARM", "Mali-G310", &[OTHER, GBM, SUBDRIVER]).is_err());
    }
}
