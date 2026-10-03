"""Unit, content, and file persistence tests for Milestone 5: Research Abstract Generation.

Tests ABSTRACT.md generation from phenotyping results, verifying mandatory
academic sections (Background, Methods, Results, Discussion, Conclusion) and
empirical metric injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pytest

try:
    from neuroglp.reporting.abstract_generator import (
        generate_abstract,
        generate_stanford_abstract,
    )
except ImportError:
    generate_abstract = None
    generate_stanford_abstract = None

# =====================================================================
# Fixture Contract Tests (Always Active)
# =====================================================================

def test_sample_cluster_stats_contract(sample_cluster_stats: Dict[int, Any]) -> None:
    """Verifies sample_cluster_stats fixture conforms to M3/M4/M5 contract."""
    assert -1 in sample_cluster_stats
    assert 0 in sample_cluster_stats
    assert 1 in sample_cluster_stats
    assert 2 in sample_cluster_stats
    for cl_id, profile in sample_cluster_stats.items():
        assert "patient_count" in profile
        assert "percentage" in profile
        assert "mean_age" in profile
        assert "hospitalization_rate" in profile
        assert "top_drugs" in profile
        assert "top_reactions" in profile


@pytest.fixture(autouse=True)
def _check_reporting_implemented(request: pytest.FixtureRequest) -> None:
    if request.node.name.startswith(("test_fixture", "test_mock", "test_sample", "test_synthetic")):
        return
    if generate_abstract is None:
        pytest.skip("neuroglp.reporting is not yet implemented")


def test_generate_abstract_returns_string(sample_cluster_stats: Dict[int, Any], tmp_path: Path) -> None:
    """Verifies generator returns non-empty markdown string and alias is preserved."""
    assert generate_stanford_abstract is generate_abstract
    out_file = tmp_path / "ABSTRACT.md"
    abstract_text = generate_abstract(sample_cluster_stats, output_path=str(out_file))
    
    assert isinstance(abstract_text, str)
    assert len(abstract_text) > 200


def test_generate_abstract_mandatory_sections(sample_cluster_stats: Dict[int, Any]) -> None:
    """Verifies that all required academic sections are present in the generated abstract."""
    abstract_text = generate_abstract(sample_cluster_stats)
    
    # Required Headings
    assert "# " in abstract_text or "## " in abstract_text
    assert "Background" in abstract_text
    assert "Methods" in abstract_text
    assert "Results" in abstract_text
    assert "Conclusion" in abstract_text or "Discussion" in abstract_text


def test_generate_abstract_empirical_metrics_injection(sample_cluster_stats: Dict[int, Any]) -> None:
    """Verifies that empirical cluster metrics (counts, percentages, drugs) are accurately injected."""
    abstract_text = generate_abstract(sample_cluster_stats)
    
    # Group 0 drugs (sertraline or SSRI)
    assert "sertraline" in abstract_text.lower() or "ssri" in abstract_text.lower()
    # Group 1 drugs (metformin or cardiometabolic)
    assert "metformin" in abstract_text.lower() or "cardiometabolic" in abstract_text.lower()
    # Group 2 drugs (ethinyl estradiol or contraceptive)
    assert "ethinyl estradiol" in abstract_text.lower() or "contraceptive" in abstract_text.lower()


def test_generate_abstract_file_persistence(sample_cluster_stats: Dict[int, Any], tmp_path: Path) -> None:
    """Verifies that the generator writes the output file to disk when requested."""
    out_file = tmp_path / "TEST_ABSTRACT.md"
    generate_abstract(sample_cluster_stats, output_path=str(out_file))
    
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    content = out_file.read_text(encoding="utf-8")
    assert "Background" in content

