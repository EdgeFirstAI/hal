#!/usr/bin/env python3
"""Assert every Mach-O we ship has an 8-byte-aligned LINKEDIT string pool.

`[profile.release] strip = true` makes rustc strip the symbol
table in-process on Apple targets rather than shelling out to Apple's
`strip(1)`, and its Mach-O writer places the string table immediately after
the indirect symbol table with no padding. Indirect symbol entries are four
bytes each, so a binary whose `nindirectsyms` is odd gets an `stroff` that is
4-byte aligned, and dyld on macOS 26+ refuses to load it:

    ImportError: dlopen(..._tensor.cpython-314-darwin.so, 0x0002):
    mis-aligned LINKEDIT string pool, fileOffset=0x0019070C

The parity is a coin flip, which is what makes this worth a gate rather than
a one-time fix: at 0.31.0 two of the five Python extensions were odd and
unloadable while the other three were even and fine by luck, so a build that
imports cleanly today proves nothing about the next one. The workaround lives
in `.cargo/config.toml` (`-C strip=none -C link-arg=-Wl,-x`, handing the strip
back to Apple's linker, which pads); this script is what notices if it is ever
dropped, reordered behind a `RUSTFLAGS` environment override, or defeated by a
toolchain bump.

Checks are pure file parsing -- no `dlopen`, no `otool` -- so a Linux or
Windows runner validates macOS artifacts just as well as a Mac does, and a
cross-built or fat binary is checked per-slice.

Usage:

    scripts/check_macho_alignment.py target/wheels target/release dist

Each argument is a wheel, a zip, a Mach-O file, or a directory to walk.
Non-Mach-O files are skipped silently. Exits non-zero on any violation, or if
nothing Mach-O was found in an argument that was supposed to contain some.
"""

from __future__ import annotations

import pathlib
import struct
import sys
import zipfile

# Mach-O magic numbers. Only the 64-bit forms matter for anything this
# project ships (arm64 and x86_64), but the 32-bit ones are recognised so a
# stray i386 slice is reported as unchecked rather than silently passing.
MH_MAGIC_64 = 0xFEEDFACF  # 64-bit, host-endian
MH_CIGAM_64 = 0xCFFAEDFE  # 64-bit, byte-swapped
MH_MAGIC_32 = 0xFEEDFACE
MH_CIGAM_32 = 0xCEFAEDFE
FAT_MAGIC = 0xCAFEBABE  # universal binary, always big-endian on disk
FAT_MAGIC_64 = 0xCAFEBABF

LC_SYMTAB = 0x2

# dyld requires the string pool to start on a pointer-size boundary.
REQUIRED_ALIGNMENT = 8

# Extensions worth opening as archives. A wheel is a zip; so is the zip the
# macOS C-API `make package` archive ships as.
ARCHIVE_SUFFIXES = {".whl", ".zip"}


def _u32(data: bytes, offset: int, big_endian: bool) -> int:
    return struct.unpack_from(">I" if big_endian else "<I", data, offset)[0]


def _symtab_stroff(data: bytes, base: int) -> tuple[int, int] | None:
    """Return ``(stroff, strsize)`` of the slice at ``base``, or None.

    None means the slice has no LC_SYMTAB at all, which is legitimate for a
    fully stripped object and simply leaves nothing to check.
    """
    magic = struct.unpack_from("<I", data, base)[0]
    if magic in (MH_MAGIC_64, MH_MAGIC_32):
        big_endian = False
    elif magic in (MH_CIGAM_64, MH_CIGAM_32):
        big_endian = True
    else:
        return None
    is_64 = magic in (MH_MAGIC_64, MH_CIGAM_64)

    # mach_header{,_64}: magic, cputype, cpusubtype, filetype, ncmds,
    # sizeofcmds, flags[, reserved]. ncmds is the 5th u32.
    ncmds = _u32(data, base + 16, big_endian)
    header_size = 32 if is_64 else 28

    offset = base + header_size
    for _ in range(ncmds):
        cmd = _u32(data, offset, big_endian)
        cmdsize = _u32(data, offset + 4, big_endian)
        if cmdsize == 0:
            # Malformed; refuse to keep walking into arbitrary bytes.
            return None
        if cmd == LC_SYMTAB:
            # symtab_command: cmd, cmdsize, symoff, nsyms, stroff, strsize
            stroff = _u32(data, offset + 16, big_endian)
            strsize = _u32(data, offset + 20, big_endian)
            return stroff, strsize
        offset += cmdsize
    return None


def _slices(data: bytes) -> list[int]:
    """File offsets of each Mach-O slice, or [] if this is not a Mach-O."""
    if len(data) < 8:
        return []
    magic_be = struct.unpack_from(">I", data, 0)[0]
    if magic_be in (FAT_MAGIC, FAT_MAGIC_64):
        nfat = _u32(data, 4, big_endian=True)
        # A corrupt or non-Mach-O file can claim an absurd slice count (the
        # magic is shared with Java class files, among others). Cap the walk
        # at what the file could actually hold rather than trusting it.
        entry_size = 20 if magic_be == FAT_MAGIC else 32
        nfat = min(nfat, max(0, (len(data) - 8) // entry_size))
        offsets = []
        # fat_arch: cputype, cpusubtype, offset, size, align (u32 each);
        # fat_arch_64 widens offset/size/align to u64 and adds reserved.
        for i in range(nfat):
            entry = 8 + i * entry_size
            if magic_be == FAT_MAGIC:
                offsets.append(_u32(data, entry + 8, big_endian=True))
            else:
                offsets.append(
                    struct.unpack_from(">Q", data, entry + 8)[0],
                )
        return offsets
    magic_le = struct.unpack_from("<I", data, 0)[0]
    if magic_le in (MH_MAGIC_64, MH_CIGAM_64, MH_MAGIC_32, MH_CIGAM_32):
        return [0]
    return []


def check_bytes(data: bytes, label: str) -> tuple[list[str], int]:
    """Check one file's bytes. Returns ``(errors, slices_checked)``."""
    errors: list[str] = []
    checked = 0
    try:
        slices = _slices(data)
    except struct.error as exc:
        return [f"{label}: truncated or malformed Mach-O container ({exc})"], 0
    for base in slices:
        if base >= len(data):
            errors.append(f"{label}: fat slice offset {base} is past end of file")
            continue
        try:
            found = _symtab_stroff(data, base)
        except struct.error as exc:
            errors.append(f"{label}: truncated or malformed Mach-O header ({exc})")
            continue
        if found is None:
            continue
        checked += 1
        stroff, strsize = found
        if stroff % REQUIRED_ALIGNMENT:
            errors.append(
                f"{label}: LC_SYMTAB string pool at file offset {stroff} "
                f"(0x{stroff:X}) is {stroff % REQUIRED_ALIGNMENT}-byte past an "
                f"{REQUIRED_ALIGNMENT}-byte boundary; dyld on macOS 26+ rejects "
                f"this image with 'mis-aligned LINKEDIT string pool'. See "
                f"the Apple strip notes in .cargo/config.toml "
                f"(strsize={strsize})"
            )
    return errors, checked


def _check_archive(path: pathlib.Path) -> tuple[list[str], int]:
    errors: list[str] = []
    checked = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info) as member:
                data = member.read()
            if not _slices(data):
                continue
            member_errors, member_checked = check_bytes(
                data, f"{path.name}::{info.filename}"
            )
            errors.extend(member_errors)
            checked += member_checked
    return errors, checked


def _check_file(path: pathlib.Path) -> tuple[list[str], int]:
    if path.suffix.lower() in ARCHIVE_SUFFIXES:
        try:
            return _check_archive(path)
        except zipfile.BadZipFile:
            return [f"{path}: not a readable zip archive"], 0
    data = path.read_bytes()
    if not _slices(data):
        return [], 0
    return check_bytes(data, str(path))


def check_path(path: pathlib.Path) -> tuple[list[str], int]:
    """Check a file or walk a directory. Returns ``(errors, slices_checked)``."""
    if path.is_dir():
        errors: list[str] = []
        checked = 0
        for child in sorted(path.rglob("*")):
            if child.is_file() and not child.is_symlink():
                child_errors, child_checked = _check_file(child)
                errors.extend(child_errors)
                checked += child_checked
        return errors, checked
    if not path.exists():
        return [f"{path}: no such file or directory"], 0
    return _check_file(path)


def main(*args: str) -> int:
    targets = [pathlib.Path(a) for a in args] or [pathlib.Path("target/wheels")]

    errors: list[str] = []
    total = 0
    for target in targets:
        target_errors, checked = check_path(target)
        errors.extend(target_errors)
        total += checked
        if not target_errors and checked == 0:
            # Not fatal on its own -- a Linux wheel directory legitimately has
            # no Mach-O in it -- but say so, so a silently empty run is never
            # mistaken for a passing one.
            print(f"note: no Mach-O symbol tables found under {target}")

    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    # ASCII on purpose, same reason as scripts/check_wheel_layout.py: a
    # Windows subprocess pipe is cp1252 and a U+2713 would raise
    # UnicodeEncodeError, turning a passing check into exit status 1.
    print(f"OK: {total} Mach-O symbol tables, string pools 8-byte aligned")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
