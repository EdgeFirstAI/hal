"""Issue #200: shipped Mach-O images must have an 8-byte-aligned string pool.

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
"""

import pathlib
import struct
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_macho_alignment.py"

MH_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC = 0xCAFEBABE
LC_SYMTAB = 0x2
CPU_TYPE_ARM64 = 0x0100000C
MH_DYLIB = 0x6


def _macho64(stroff: int) -> bytes:
    """A minimal little-endian 64-bit Mach-O carrying one LC_SYMTAB.

    Only the fields the checker reads have to be truthful: the magic, ncmds,
    and the symtab_command itself. Everything else is plausible filler -- this
    is a parser fixture, not a loadable image.
    """
    symtab = struct.pack(
        "<IIIIII",
        LC_SYMTAB,
        24,  # cmdsize
        0x1000,  # symoff
        4,  # nsyms
        stroff,
        16,  # strsize
    )
    header = struct.pack(
        "<IiiIIIII",
        MH_MAGIC_64,
        CPU_TYPE_ARM64,
        0,  # cpusubtype
        MH_DYLIB,
        1,  # ncmds
        len(symtab),
        0,  # flags
        0,  # reserved
    )
    body = header + symtab
    # Pad out past the offsets named above so nothing reads off the end.
    return body + b"\0" * (0x2000 - len(body))


def _run(*paths: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), *(str(p) for p in paths)],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )


@pytest.mark.parametrize("stroff", [0x1004, 0x100C, 0x1F04])
def test_misaligned_string_pool_is_rejected(tmp_path, stroff):
    """The failure this gate exists for. Must be red, or the gate is theatre."""
    bad = tmp_path / "libbad.dylib"
    bad.write_bytes(_macho64(stroff))

    result = _run(bad)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "mis-aligned LINKEDIT string pool" in result.stdout
    assert str(stroff) in result.stdout


@pytest.mark.parametrize("stroff", [0x1000, 0x1008, 0x2000])
def test_aligned_string_pool_is_accepted(tmp_path, stroff):
    good = tmp_path / "libgood.dylib"
    good.write_bytes(_macho64(stroff))

    result = _run(good)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK:" in result.stdout


def test_wheel_member_is_checked(tmp_path):
    """A wheel is a zip; the extension inside it is what dyld will open."""
    import zipfile

    wheel = tmp_path / "edgefirst_tensor-0.0.0-cp314-cp314-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("edgefirst/tensor/__init__.py", "")
        zf.writestr("edgefirst/tensor/_tensor.cpython-314-darwin.so", _macho64(0x100C))

    result = _run(wheel)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "_tensor.cpython-314-darwin.so" in result.stdout


def test_fat_binary_slices_are_checked_individually(tmp_path):
    """A universal binary hides a bad slice behind a good one."""
    good, bad = _macho64(0x1000), _macho64(0x100C)
    header_size = 8 + 2 * 20
    good_off = (header_size + 0x3FFF) & ~0x3FFF
    bad_off = good_off + len(good)

    blob = struct.pack(">II", FAT_MAGIC, 2)
    for offset, slice_bytes in ((good_off, good), (bad_off, bad)):
        blob += struct.pack(">iiIII", CPU_TYPE_ARM64, 0, offset, len(slice_bytes), 14)
    blob = blob.ljust(good_off, b"\0") + good + bad

    fat = tmp_path / "libfat.dylib"
    fat.write_bytes(blob)

    result = _run(fat)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "mis-aligned LINKEDIT string pool" in result.stdout


def test_non_macho_files_are_ignored(tmp_path):
    """An ELF or a text file must not be mistaken for a violation."""
    (tmp_path / "libelf.so").write_bytes(b"\x7fELF" + b"\0" * 1024)
    (tmp_path / "README.txt").write_text("not a binary")

    result = _run(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no Mach-O symbol tables found" in result.stdout


def test_truncated_macho_is_reported_not_raised(tmp_path):
    """A partial write in a build tree must fail loudly, not traceback."""
    # Mach-O magic and a cputype, then nothing -- the header runs off the end.
    (tmp_path / "libtrunc.dylib").write_bytes(
        struct.pack("<Ii", MH_MAGIC_64, CPU_TYPE_ARM64)
    )

    result = _run(tmp_path / "libtrunc.dylib")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "truncated or malformed" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS build output only")
@pytest.mark.skipif(
    not list((REPO_ROOT / "target" / "release").glob("*.dylib")),
    reason="no release dylibs built; run `cargo build --release`",
)
def test_real_release_artifacts_are_aligned():
    result = _run(REPO_ROOT / "target" / "release")
    assert result.returncode == 0, result.stdout + result.stderr
