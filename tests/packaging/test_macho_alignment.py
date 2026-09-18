"""Shipped Mach-O images must have an 8-byte-aligned LINKEDIT string pool.

`[profile.release] strip = true` makes rustc strip Apple binaries with its own
in-process Mach-O writer, which places the string table immediately after the
indirect symbol table with no padding. Indirect symbol entries are four bytes,
so a binary whose ``nindirectsyms`` is odd ends up with a 4-byte-aligned
``stroff``, and dyld on macOS 26+ refuses to load it with "mis-aligned LINKEDIT
string pool". `.cargo/config.toml` works around it by handing the strip back to
Apple's linker (``-C strip=none -C link-arg=-Wl,-x``), which pads correctly.

These cases synthesise Mach-O headers rather than compiling anything, so the
checker's logic is gated on every platform's CI lane, not only macOS -- the
whole point of a file-parsing gate is that the runner does not have to be able
to load the image. The final case additionally scans real build output when it
happens to be present.

The checker is imported and called in-process. Driving it only as a subprocess
would leave it invisible to coverage, which instruments this interpreter and
not its children. One case still goes through the command line, because the
exit status the Makefile branches on is its own contract.
"""

from __future__ import annotations

import importlib.util
import pathlib
import struct
import subprocess
import sys
import zipfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_macho_alignment.py"

MH_MAGIC_64 = 0xFEEDFACF
MH_CIGAM_64 = 0xCFFAEDFE
FAT_MAGIC = 0xCAFEBABE
FAT_MAGIC_64 = 0xCAFEBABF
LC_SYMTAB = 0x2
LC_UUID = 0x1B
CPU_TYPE_ARM64 = 0x0100000C
MH_DYLIB = 0x6


def _load():
    spec = importlib.util.spec_from_file_location("check_macho_alignment", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load()


def _macho64(stroff: int, *, big_endian: bool = False, cmd: int = LC_SYMTAB) -> bytes:
    """A minimal 64-bit Mach-O carrying one load command.

    Only the fields the checker reads have to be truthful: the magic, ncmds,
    and the symtab_command itself. Everything else is plausible filler -- this
    is a parser fixture, not a loadable image.
    """
    endian = ">" if big_endian else "<"
    magic = MH_CIGAM_64 if big_endian else MH_MAGIC_64
    command = struct.pack(
        endian + "IIIIII",
        cmd,
        24,  # cmdsize
        0x1000,  # symoff
        4,  # nsyms
        stroff,
        16,  # strsize
    )
    header = struct.pack(
        endian + "IiiIIIII",
        magic,
        CPU_TYPE_ARM64,
        0,  # cpusubtype
        MH_DYLIB,
        1,  # ncmds
        len(command),
        0,  # flags
        0,  # reserved
    )
    # The magic is written in the byte order the loader reads it in, so a
    # big-endian fixture has to carry the swapped magic literally.
    if big_endian:
        header = struct.pack("<I", magic) + header[4:]
    body = header + command
    # Pad out past the offsets named above so nothing reads off the end.
    return body + b"\0" * (0x2000 - len(body))


def _macho64_no_commands() -> bytes:
    header = struct.pack(
        "<IiiIIIII", MH_MAGIC_64, CPU_TYPE_ARM64, 0, MH_DYLIB, 0, 0, 0, 0
    )
    return header + b"\0" * (0x1000 - len(header))


def _fat(slices: list[bytes], *, magic: int = FAT_MAGIC) -> bytes:
    entry_size = 20 if magic == FAT_MAGIC else 32
    header_size = 8 + len(slices) * entry_size
    first = (header_size + 0x3FFF) & ~0x3FFF

    offsets, cursor = [], first
    for payload in slices:
        offsets.append(cursor)
        cursor += len(payload)

    blob = struct.pack(">II", magic, len(slices))
    for offset, payload in zip(offsets, slices):
        if magic == FAT_MAGIC:
            blob += struct.pack(">iiIII", CPU_TYPE_ARM64, 0, offset, len(payload), 14)
        else:
            blob += struct.pack(
                ">iiQQII", CPU_TYPE_ARM64, 0, offset, len(payload), 14, 0
            )
    return blob.ljust(first, b"\0") + b"".join(slices)


def _run_cli(*paths: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), *(str(p) for p in paths)],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )


@pytest.mark.parametrize("stroff", [0x1004, 0x100C, 0x1F04])
def test_misaligned_string_pool_is_rejected(stroff):
    """The failure this gate exists for. Must be red, or the gate is theatre."""
    errors, checked = checker.check_bytes(_macho64(stroff), "libbad.dylib")

    assert checked == 1
    assert len(errors) == 1
    assert "mis-aligned LINKEDIT string pool" in errors[0]
    assert str(stroff) in errors[0]


@pytest.mark.parametrize("stroff", [0x1000, 0x1008, 0x2000])
def test_aligned_string_pool_is_accepted(stroff):
    errors, checked = checker.check_bytes(_macho64(stroff), "libgood.dylib")

    assert errors == []
    assert checked == 1


def test_byte_swapped_image_is_parsed():
    """A big-endian header must be read, not skipped as unrecognised."""
    errors, checked = checker.check_bytes(
        _macho64(0x100C, big_endian=True), "libswapped.dylib"
    )

    assert checked == 1
    assert "mis-aligned LINKEDIT string pool" in errors[0]


def test_image_without_symtab_has_nothing_to_check():
    """A fully stripped object is legitimate and must not be a violation."""
    assert checker.check_bytes(_macho64_no_commands(), "libnosym.dylib") == ([], 0)
    assert checker.check_bytes(_macho64(0x100C, cmd=LC_UUID), "libuuid.dylib") == (
        [],
        0,
    )


def test_zero_length_load_command_stops_the_walk():
    """A cmdsize of zero would otherwise walk the header forever."""
    blob = bytearray(_macho64(0x100C))
    struct.pack_into("<I", blob, 32 + 4, 0)  # load_command.cmdsize

    assert checker.check_bytes(bytes(blob), "libloop.dylib") == ([], 0)


def test_wheel_member_is_checked(tmp_path):
    """A wheel is a zip; the extension inside it is what dyld will open."""
    wheel = tmp_path / "edgefirst_tensor-0.0.0-cp314-cp314-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("edgefirst/tensor/", "")
        zf.writestr("edgefirst/tensor/__init__.py", "")
        zf.writestr("edgefirst/tensor/_tensor.cpython-314-darwin.so", _macho64(0x100C))

    errors, checked = checker.check_path(wheel)

    assert checked == 1
    assert len(errors) == 1
    assert "_tensor.cpython-314-darwin.so" in errors[0]


def test_unreadable_archive_is_reported(tmp_path):
    wheel = tmp_path / "broken.whl"
    wheel.write_bytes(b"not a zip at all")

    errors, checked = checker.check_path(wheel)

    assert checked == 0
    assert "not a readable zip archive" in errors[0]


@pytest.mark.parametrize("magic", [FAT_MAGIC, FAT_MAGIC_64])
def test_fat_binary_slices_are_checked_individually(magic):
    """A universal binary hides a bad slice behind a good one."""
    blob = _fat([_macho64(0x1000), _macho64(0x100C)], magic=magic)

    errors, checked = checker.check_bytes(blob, "libfat.dylib")

    assert checked == 2
    assert len(errors) == 1
    assert "mis-aligned LINKEDIT string pool" in errors[0]


def test_fat_slice_offset_past_end_is_reported():
    """A truncated universal binary must name the problem, not index off it."""
    blob = bytearray(_fat([_macho64(0x1000)]))
    struct.pack_into(">I", blob, 8 + 8, 0xFFFF0000)  # fat_arch.offset

    errors, checked = checker.check_bytes(bytes(blob), "libshort.dylib")

    assert checked == 0
    assert "past end of file" in errors[0]


def test_non_macho_files_are_ignored(tmp_path):
    """An ELF or a text file must not be mistaken for a violation."""
    (tmp_path / "libelf.so").write_bytes(b"\x7fELF" + b"\0" * 1024)
    (tmp_path / "README.txt").write_text("not a binary")

    assert checker.check_path(tmp_path) == ([], 0)


def test_directory_walk_reaches_nested_artifacts(tmp_path):
    nested = tmp_path / "release" / "deps"
    nested.mkdir(parents=True)
    (nested / "libgood.dylib").write_bytes(_macho64(0x1000))
    (nested / "libbad.dylib").write_bytes(_macho64(0x100C))

    errors, checked = checker.check_path(tmp_path)

    assert checked == 2
    assert len(errors) == 1
    assert "libbad.dylib" in errors[0]


def test_truncated_macho_is_reported_not_raised(tmp_path):
    """A partial write in a build tree must fail loudly, not traceback."""
    # Mach-O magic and a cputype, then nothing -- the header runs off the end.
    truncated = tmp_path / "libtrunc.dylib"
    truncated.write_bytes(struct.pack("<Ii", MH_MAGIC_64, CPU_TYPE_ARM64))

    errors, checked = checker.check_path(truncated)

    assert checked == 0
    assert "truncated or malformed" in errors[0]


def test_missing_path_is_an_error(tmp_path):
    errors, checked = checker.check_path(tmp_path / "absent")

    assert checked == 0
    assert "no such file or directory" in errors[0]


def test_main_reports_a_violation(tmp_path, capsys):
    (tmp_path / "libbad.dylib").write_bytes(_macho64(0x100C))

    assert checker.main(str(tmp_path)) == 1
    assert "ERROR:" in capsys.readouterr().out


def test_main_says_so_when_it_found_nothing(tmp_path, capsys):
    """A silently empty run must never read as a passing one."""
    (tmp_path / "notes.txt").write_text("no binaries here")

    assert checker.main(str(tmp_path)) == 0

    out = capsys.readouterr().out
    assert "no Mach-O symbol tables found" in out
    assert "OK: 0 Mach-O symbol tables" in out


def test_main_defaults_to_the_wheel_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert checker.main() == 1
    # Separator-agnostic: the default target is a relative path.
    assert "wheels" in capsys.readouterr().out


def test_command_line_exit_status(tmp_path):
    """`make` branches on the exit status, so the CLI plumbing is a contract."""
    good, bad = tmp_path / "libgood.dylib", tmp_path / "libbad.dylib"
    good.write_bytes(_macho64(0x1000))
    bad.write_bytes(_macho64(0x100C))

    passing = _run_cli(good)
    assert passing.returncode == 0, passing.stdout + passing.stderr
    assert "OK:" in passing.stdout

    failing = _run_cli(bad)
    assert failing.returncode == 1, failing.stdout + failing.stderr
    assert "mis-aligned LINKEDIT string pool" in failing.stdout
    assert "Traceback" not in failing.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS build output only")
@pytest.mark.skipif(
    not list((REPO_ROOT / "target" / "release").glob("*.dylib")),
    reason="no release dylibs built; run `cargo build --release`",
)
def test_real_release_artifacts_are_aligned():
    errors, checked = checker.check_path(REPO_ROOT / "target" / "release")

    assert errors == []
    assert checked > 0
