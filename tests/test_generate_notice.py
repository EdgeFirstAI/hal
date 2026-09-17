# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Tests for NOTICE generation from a CycloneDX SBOM.

NOTICE is the attribution artifact shipped to downstream consumers. These tests
pin the three shapes a CycloneDX licence entry takes -- an SPDX id, an SPDX
expression, and a free-form name carrying the licence as text -- and the rule
that a component is never dropped because its licence could not be identified.
"""

import base64
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "generate_notice.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("generate_notice", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_notice = _load()


def component(licenses):
    """A CycloneDX component carrying the given licence entries."""
    return {"name": "example", "version": "1.0.0", "licenses": licenses}


def text_licence(body, encoding="base64"):
    """The shape cargo-cyclonedx emits for a crate that sets `license-file`."""
    content = base64.b64encode(body.encode()).decode() if encoding == "base64" else body
    entry = {"name": "Unknown", "text": {"content": content}}
    if encoding:
        entry["text"]["encoding"] = encoding
    return [{"license": entry}]


MIT_BODY = "MIT License\n\nPermission is hereby granted, free of charge, to any person"
ISC_BODY = (
    "Permission to use, copy, modify, and/or distribute this software for any purpose"
)


class TestLicenceExtraction:
    def test_spdx_id_is_used_directly(self):
        assert generate_notice.extract_license_from_component(
            component([{"license": {"id": "MIT"}}])
        ) == {"MIT"}

    def test_expression_is_split_into_its_terms(self):
        # "Unlicense OR MIT" must yield MIT, or every dual-licensed crate that
        # offers a public-domain option would fall out of NOTICE.
        assert generate_notice.extract_license_from_component(
            component([{"expression": "Unlicense OR MIT"}])
        ) == {"Unlicense", "MIT"}

    def test_meaningful_name_is_taken_when_there_is_no_id(self):
        assert generate_notice.extract_license_from_component(
            component([{"license": {"name": "BSD-3-Clause"}}])
        ) == {"BSD-3-Clause"}

    @pytest.mark.parametrize("placeholder", ["Unknown", "NOASSERTION", "none"])
    def test_uninformative_names_do_not_count_as_identification(self, placeholder):
        # These fall through to the licence text rather than being recorded as if
        # they named a licence. With no text to fall through to, the entry still
        # carried a claim, so it lands on Unclassified.
        assert generate_notice.extract_license_from_component(
            component([{"license": {"name": placeholder}}])
        ) == {"Unclassified"}

    @pytest.mark.parametrize("blank", ["", "  "])
    def test_an_empty_name_carries_no_claim_at_all(self, blank):
        # Distinct from "Unknown": there is nothing here to be unclassified.
        assert (
            generate_notice.extract_license_from_component(
                component([{"license": {"name": blank}}])
            )
            == set()
        )

    @pytest.mark.parametrize("name", ["MIT License", "AcmeCorp Internal", "Apache 2"])
    def test_unrecognised_name_does_not_shortcut_the_licence_text(self, name):
        # A free-form name is not an identifier. Treating one as identification
        # would mark the component classified, skip the text that can place it,
        # and then drop it for not matching the attribution set.
        assert generate_notice.extract_license_from_component(
            component([{"license": {"name": name, "text": {"content": MIT_BODY}}}])
        ) == {"MIT"}

    @pytest.mark.parametrize("name", ["MIT License", "AcmeCorp Internal"])
    def test_unrecognised_name_with_no_text_is_attributed_not_dropped(self, name):
        licences = generate_notice.extract_license_from_component(
            component([{"license": {"name": name}}])
        )
        assert licences == {"Unclassified"}
        assert generate_notice.requires_attribution(licences)

    @pytest.mark.parametrize("spdx", ["BSD-3-Clause", "ISC", "CC0-1.0", "Unlicense"])
    def test_a_recognised_identifier_given_as_a_name_is_taken(self, spdx):
        assert generate_notice.extract_license_from_component(
            component([{"license": {"name": spdx}}])
        ) == {spdx}

    def test_licence_carried_as_text_is_identified(self):
        # The shape a crate setting `license-file` produces: no SPDX id anywhere.
        assert generate_notice.extract_license_from_component(
            component(text_licence(MIT_BODY))
        ) == {"MIT"}

    def test_isc_text_is_not_mistaken_for_mit(self):
        assert generate_notice.extract_license_from_component(
            component(text_licence(ISC_BODY))
        ) == {"ISC"}

    def test_unencoded_text_is_read_too(self):
        assert generate_notice.extract_license_from_component(
            component(text_licence(MIT_BODY, encoding=None))
        ) == {"MIT"}


class TestNothingIsDroppedSilently:
    """A component carrying any licence claim always reaches NOTICE."""

    def test_unrecognised_text_is_attributed_rather_than_discarded(self):
        licences = generate_notice.extract_license_from_component(
            component(text_licence("Some bespoke corporate licence"))
        )
        assert licences == {"Unclassified"}
        assert generate_notice.requires_attribution(licences)

    def test_undecodable_text_is_attributed_rather_than_discarded(self):
        licences = generate_notice.extract_license_from_component(
            component(
                [
                    {
                        "license": {
                            "name": "Unknown",
                            "text": {
                                "encoding": "base64",
                                "content": "!!!not base64!!!",
                            },
                        }
                    }
                ]
            )
        )
        assert licences == {"Unclassified"}
        assert generate_notice.requires_attribution(licences)

    def test_component_with_no_licence_block_yields_nothing(self):
        assert generate_notice.extract_license_from_component({"name": "x"}) == set()


class TestAttributionPolicy:
    @pytest.mark.parametrize(
        "licence",
        ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib", "MPL-2.0"],
    )
    def test_attribution_required(self, licence):
        assert generate_notice.requires_attribution({licence})

    @pytest.mark.parametrize("licence", ["Unlicense", "CC0-1.0", "0BSD"])
    def test_attribution_not_required(self, licence):
        # Public-domain-equivalent licences ask for nothing. A component under
        # one of these alongside MIT is still attributed, via the MIT term.
        assert not generate_notice.requires_attribution({licence})

    def test_dual_licence_with_a_permissive_term_is_attributed(self):
        assert generate_notice.requires_attribution({"Unlicense", "MIT"})


class TestNoticeRendering:
    """`generate_notice` reads an SBOM from disk, so these go through a file."""

    @staticmethod
    def render(tmp_path, sbom):
        path = tmp_path / "sbom.json"
        path.write_text(json.dumps(sbom), encoding="utf-8")
        return generate_notice.generate_notice(str(path))

    def test_every_attributed_component_is_named_with_its_version(self, tmp_path):
        notice = self.render(
            tmp_path,
            {
                "components": [
                    {
                        "name": "alpha",
                        "version": "1.0.0",
                        "licenses": [{"license": {"id": "MIT"}}],
                    },
                    {
                        "name": "beta",
                        "version": "2.3.4",
                        "licenses": text_licence(MIT_BODY),
                    },
                    {
                        "name": "gamma",
                        "version": "0.1.0",
                        "licenses": [{"license": {"id": "CC0-1.0"}}],
                    },
                ]
            },
        )
        assert "  * alpha 1.0.0 (MIT)" in notice
        assert "  * beta 2.3.4 (MIT)" in notice
        # gamma asks for no attribution, so it is absent by policy, not by accident.
        assert "gamma" not in notice

    def test_copyright_line_is_present(self, tmp_path):
        # The org validator fails outright on a NOTICE with no copyright line.
        assert "Copyright" in self.render(tmp_path, {"components": []})

    def test_entries_are_sorted_so_regeneration_produces_a_readable_diff(
        self, tmp_path
    ):
        notice = self.render(
            tmp_path,
            {
                "components": [
                    {
                        "name": "zeta",
                        "version": "1.0.0",
                        "licenses": [{"license": {"id": "MIT"}}],
                    },
                    {
                        "name": "Alpha",
                        "version": "1.0.0",
                        "licenses": [{"license": {"id": "MIT"}}],
                    },
                    {
                        "name": "alpha",
                        "version": "0.9.0",
                        "licenses": [{"license": {"id": "MIT"}}],
                    },
                ]
            },
        )
        listed = [line for line in notice.splitlines() if line.startswith("  * ")]
        assert listed == sorted(listed, key=lambda s: s.lower())


class TestCheckMode:
    """`--check` gates on what the generator would WRITE, not on the SBOM."""

    @staticmethod
    def write(tmp_path, sbom, notice_text):
        sbom_path = tmp_path / "sbom.json"
        sbom_path.write_text(json.dumps(sbom), encoding="utf-8")
        notice_path = tmp_path / "NOTICE"
        notice_path.write_text(notice_text, encoding="utf-8")
        return str(sbom_path), str(notice_path)

    def test_missing_attributable_component_fails(self, tmp_path):
        sbom = {
            "components": [
                {
                    "name": "alpha",
                    "version": "1.0.0",
                    "licenses": [{"license": {"id": "MIT"}}],
                },
            ]
        }
        s, n = self.write(tmp_path, sbom, "Copyright someone\n")
        assert generate_notice.check_notice(s, n) == 1

    def test_named_attributable_component_passes(self, tmp_path):
        sbom = {
            "components": [
                {
                    "name": "alpha",
                    "version": "1.0.0",
                    "licenses": [{"license": {"id": "MIT"}}],
                },
            ]
        }
        s, n = self.write(
            tmp_path, sbom, "Copyright someone\n\n  * alpha 1.0.0 (MIT)\n"
        )
        assert generate_notice.check_notice(s, n) == 0

    def test_a_no_attribution_licence_cannot_wedge_the_gate(self, tmp_path):
        # CC0-1.0 asks for nothing, so the generator omits it. Gating on the full
        # component list would fail here with no way to clear it, since
        # regenerating NOTICE would correctly go on omitting it.
        sbom = {
            "components": [
                {
                    "name": "alpha",
                    "version": "1.0.0",
                    "licenses": [{"license": {"id": "MIT"}}],
                },
                {
                    "name": "public-domain-thing",
                    "version": "9.9.9",
                    "licenses": [{"license": {"id": "CC0-1.0"}}],
                },
            ]
        }
        s, n = self.write(
            tmp_path, sbom, "Copyright someone\n\n  * alpha 1.0.0 (MIT)\n"
        )
        assert generate_notice.check_notice(s, n) == 0

    def test_a_stale_version_is_caught(self, tmp_path):
        # Matching on name alone would miss a dependency bump that left NOTICE
        # naming the old version.
        sbom = {
            "components": [
                {
                    "name": "alpha",
                    "version": "2.0.0",
                    "licenses": [{"license": {"id": "MIT"}}],
                },
            ]
        }
        s, n = self.write(
            tmp_path, sbom, "Copyright someone\n\n  * alpha 1.0.0 (MIT)\n"
        )
        assert generate_notice.check_notice(s, n) == 1

    def test_duplicate_component_records_are_counted_once(self, tmp_path):
        # The merged SBOM carries the same (name, version) from several
        # per-target scans.
        entry = {
            "name": "alpha",
            "version": "1.0.0",
            "licenses": [{"license": {"id": "MIT"}}],
        }
        s, _ = self.write(tmp_path, {"components": [entry, dict(entry)]}, "")
        assert generate_notice.attributable_entries(s) == [("alpha", "1.0.0")]
