"""Unit tests for CVE and MITRE parsers."""
import pytest
from apps.ingestion.parsers.base import extract_cve_data, extract_mitre_technique, normalize_to_chunk


class TestExtractCveData:
    def test_extracts_valid_cve(self):
        cve = {
            "id": "CVE-2024-1234",
            "descriptions": [{"lang": "en", "value": "A buffer overflow vulnerability"}],
            "metrics": {
                "cvssMetricV31": [{"cvssData": {"baseSeverity": "HIGH"}}]
            },
            "published": "2024-01-15T10:00:00",
            "references": [{"url": "https://example.com/advisory"}],
        }
        result = extract_cve_data(cve)
        assert result is not None
        assert result["cve_id"] == "CVE-2024-1234"
        assert result["description"] == "A buffer overflow vulnerability"
        assert result["severity"] == "HIGH"
        assert result["published_date"] == "2024-01-15T10:00:00"
        assert result["references"] == ["https://example.com/advisory"]
        assert result["source"] == "nvd"

    def test_returns_none_for_missing_id(self):
        cve = {"descriptions": [{"lang": "en", "value": "test"}]}
        assert extract_cve_data(cve) is None

    def test_returns_none_for_reserved(self):
        cve = {
            "id": "CVE-2024-9999",
            "descriptions": [{"lang": "en", "value": "** RESERVED **"}],
        }
        assert extract_cve_data(cve) is None

    def test_returns_none_for_empty_description(self):
        cve = {"id": "CVE-2024-0001", "descriptions": []}
        assert extract_cve_data(cve) is None

    def test_uses_cvss_v2_fallback(self):
        cve = {
            "id": "CVE-2024-5678",
            "descriptions": [{"lang": "en", "value": "SQL injection"}],
            "metrics": {"cvssMetricV2": [{"baseSeverity": "MEDIUM"}]},
        }
        result = extract_cve_data(cve)
        assert result["severity"] == "MEDIUM"

    def test_unknown_severity_when_no_metrics(self):
        cve = {
            "id": "CVE-2024-0002",
            "descriptions": [{"lang": "en", "value": "XSS"}],
            "metrics": {},
        }
        result = extract_cve_data(cve)
        assert result["severity"] == "UNKNOWN"

    def test_skips_non_english_descriptions(self):
        cve = {
            "id": "CVE-2024-0003",
            "descriptions": [
                {"lang": "es", "value": "Descripcion en espanol"},
                {"lang": "en", "value": "English description"},
            ],
        }
        result = extract_cve_data(cve)
        assert result["description"] == "English description"


class TestExtractMitreTechnique:
    def test_extracts_valid_technique(self):
        obj = {
            "type": "attack-pattern",
            "name": "Phishing",
            "description": "Adversaries send [phishing](https://example.com) emails",
            "external_references": [
                {"source_name": "mitre-attack", "external_id": "T1566"}
            ],
            "kill_chain_phases": [
                {"kill_chain_name": "mitre-attack", "phase_name": "initial-access"}
            ],
            "x_mitre_platforms": ["Windows", "Linux"],
        }
        result = extract_mitre_technique(obj)
        assert result is not None
        assert result["cve_id"] == "T1566"
        assert result["name"] == "Phishing"
        assert result["tactic"] == ["initial-access"]
        assert result["platforms"] == ["Windows", "Linux"]
        assert "phishing" in result["description"]

    def test_returns_none_for_non_attack_pattern(self):
        obj = {"type": "malware", "name": "test"}
        assert extract_mitre_technique(obj) is None

    def test_returns_none_for_deprecated(self):
        obj = {
            "type": "attack-pattern",
            "x_mitre_deprecated": True,
            "external_references": [{"source_name": "mitre-attack", "external_id": "T9999"}],
        }
        assert extract_mitre_technique(obj) is None

    def test_returns_none_for_revoked(self):
        obj = {
            "type": "attack-pattern",
            "revoked": True,
            "external_references": [{"source_name": "mitre-attack", "external_id": "T9999"}],
        }
        assert extract_mitre_technique(obj) is None

    def test_returns_none_for_missing_technique_id(self):
        obj = {"type": "attack-pattern", "name": "Test"}
        assert extract_mitre_technique(obj) is None

    def test_strips_markdown_links(self):
        obj = {
            "type": "attack-pattern",
            "name": "Test",
            "description": "See [MITRE](https://example.com) for details",
            "external_references": [{"source_name": "mitre-attack", "external_id": "T0001"}],
        }
        result = extract_mitre_technique(obj)
        assert "MITRE" in result["description"]
        assert "https://example.com" not in result["description"]


class TestNormalizeToChunk:
    def test_normalizes_document(self):
        doc = {"cve_id": "CVE-2024-1234", "severity": "HIGH", "source": "nvd"}
        result = normalize_to_chunk(doc, "chunk text", 0)
        assert result["text"] == "chunk text"
        assert result["cve_id"] == "CVE-2024-1234"
        assert result["severity"] == "HIGH"
        assert result["chunk_index"] == 0

    def test_handles_missing_fields(self):
        result = normalize_to_chunk({}, "text", 1)
        assert result["cve_id"] == "unknown"
        assert result["severity"] == "UNKNOWN"
        assert result["source"] == "unknown"
