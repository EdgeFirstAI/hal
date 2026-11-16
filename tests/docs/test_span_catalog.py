"""The tracing catalog in ARCHITECTURE.md must describe the spans that exist.

Checks that every catalogued span name is emitted, every quoted field value is recorded,
every emitted span is catalogued, and the ``*_memory`` row lists exactly the
``TensorMemory`` variants. A catalog token with a ``<placeholder>`` or a ``*`` glob is a
pattern and is skipped.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOT_ARCH = REPO / "ARCHITECTURE.md"

# Spans emitted only from a `#[cfg(test)]` block.
TEST_ONLY_SPANS = frozenset({"hal.test_span"})

# Matches a span macro and captures its name, on the macro's line or the next.
SPAN_MACRO = re.compile(r'(?:trace_span|info_span|debug_span)!\(\s*"([a-zA-Z0-9_.]+)"')
# Leading segments of a span name.
SPAN_SEGMENTS = (
    "codec",
    "tensor",
    "image",
    "decoder",
    "tracker",
    "kernels",
    "python",
    "hal",
)
SPAN_NAME = re.compile(r"^(?:" + "|".join(SPAN_SEGMENTS) + r")(?:\.[a-z0-9_]+)+$")


def _rust_sources() -> list[pathlib.Path]:
    return sorted((REPO / "crates").rglob("*.rs"))


@pytest.fixture(scope="module")
def rust_text() -> str:
    """Every Rust source concatenated -- the corpus all checks search."""
    return "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in _rust_sources()
    )


def _span_bodies(text: str) -> list[str]:
    """The argument list of every span macro, by balanced-paren scan."""
    bodies = []
    for m in re.finditer(r"(?:trace_span|info_span|debug_span)!\(", text):
        depth, i = 0, m.end() - 1
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    bodies.append(text[m.end() : i])
                    break
            i += 1
    return bodies


def _recorded_field_strings() -> set[str]:
    """Every string a span macro can record as a field value.

    Includes the string literals of a `let` that binds a field passed bare
    (`info_span!("image.gl_init", platform, ...)`).
    """
    values: set[str] = set()
    for path in _rust_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        bodies = _span_bodies(text)
        if not bodies:
            continue
        bare: set[str] = set()
        for body in bodies:
            values |= set(re.findall(r'"([^"\\]*)"', body))
            for arg in body.split(","):
                arg = arg.strip()
                if re.fullmatch(r"[a-z_][a-z0-9_]*", arg):
                    bare.add(arg)
        for field in bare:
            for m in re.finditer(rf"\blet {field}\s*=", text):
                stmt = text[m.end() : m.end() + 400].split(";", 1)[0]
                values |= set(re.findall(r'"([^"\\]*)"', stmt))
    return values


@pytest.fixture(scope="module")
def recorded_strings() -> set[str]:
    return _recorded_field_strings()


@pytest.fixture(scope="module")
def emitted_spans(rust_text: str) -> set[str]:
    return set(SPAN_MACRO.findall(rust_text))


def _catalog_section() -> str:
    """The span-name table plus the field conventions that follow it."""
    text = ROOT_ARCH.read_text(encoding="utf-8")
    start = text.index("### Span naming conventions")
    end = text.index("### Crate layering", start)
    return text[start:end]


def _backticked(chunk: str) -> list[str]:
    return re.findall(r"`([^`]+)`", chunk)


def _catalogued_span_names() -> set[str]:
    names = set()
    for token in _backticked(_catalog_section()):
        if "*" in token or "<" in token:
            continue  # a row pattern (`image.convert.<backend>`), not a name
        if SPAN_NAME.match(token):
            names.add(token)
    return names


def _field_conventions() -> str:
    section = _catalog_section()
    return section[section.index("Field conventions:") :]


def _tensor_memory_variants() -> set[str]:
    """The `TensorMemory` variant names, from the enum's own definition."""
    text = (REPO / "crates" / "tensor" / "src" / "lib.rs").read_text(encoding="utf-8")
    body = text[text.index("pub enum TensorMemory {") :]
    body = body[: body.index("\n    }")]
    return set(re.findall(r"^\s+([A-Z][A-Za-z0-9]*) = \d+,", body, re.MULTILINE))


def test_every_catalogued_span_name_is_emitted(emitted_spans: set[str]):
    """A span the catalog names must exist in the source."""
    catalogued = _catalogued_span_names()
    assert catalogued, "extracted no span names -- the section markers moved"

    phantoms = sorted(catalogued - emitted_spans)
    assert not phantoms, (
        f"{ROOT_ARCH.name} documents spans no source emits: {phantoms}. Either "
        f"the span was renamed or removed and the catalog was not updated, or "
        f"the name is a typo. Filtering a trace on one of these returns nothing."
    )


def test_every_catalogued_field_string_is_recorded(recorded_strings: set[str]):
    """A quoted field value in the catalog must appear in the source."""
    documented = {
        token.strip('"')
        for token in _backticked(_field_conventions())
        if token.startswith('"') and token.endswith('"') and len(token) > 2
    }
    assert documented, "extracted no quoted field values -- the list format moved"

    phantoms = sorted(documented - recorded_strings)
    assert not phantoms, (
        f"{ROOT_ARCH.name} documents field values that no span macro records: "
        f"{phantoms}. A reader filtering a trace on one of these gets an empty "
        f"result and concludes the code path never ran."
    )


def test_every_emitted_span_is_catalogued(emitted_spans: set[str]):
    """A span in the source must be documented in some ARCHITECTURE.md."""
    docs = "\n".join(
        p.read_text(encoding="utf-8")
        for p in [ROOT_ARCH, *sorted((REPO / "crates").glob("*/ARCHITECTURE.md"))]
    )
    undocumented = sorted(s for s in emitted_spans - TEST_ONLY_SPANS if s not in docs)
    assert not undocumented, (
        f"spans emitted by the source appear in no ARCHITECTURE.md: "
        f"{undocumented}. Add the span to its crate's document (and to the root "
        f"catalog's segment table when it starts a new segment); a span nobody "
        f"documented is a span nobody knows to filter on."
    )


def test_memory_field_lists_the_real_variants():
    """The `*_memory` row must list exactly the `TensorMemory` variants."""
    variants = _tensor_memory_variants()
    assert variants, "parsed no TensorMemory variants -- the enum's shape moved"

    row = next(
        line
        for line in _field_conventions().splitlines()
        if line.startswith("- `*_memory`")
    )
    documented = {
        t
        for t in _backticked(row)
        if re.fullmatch(r"[A-Z][A-Za-z0-9]*", t) and t != "TensorMemory"
    }

    assert documented == variants, (
        f"the `*_memory` row lists {sorted(documented)} but `TensorMemory` has "
        f"{sorted(variants)}. Missing: {sorted(variants - documented)}; "
        f"non-existent: {sorted(documented - variants)}."
    )
