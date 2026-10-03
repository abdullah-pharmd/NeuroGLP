"""Unit and boundary tests for Milestone 1: Data Ingestion & Preprocessing.

Tests OpenFDAClient query construction, mock HTTP requests, rate limiting,
pagination, caching, offline fallback, and record flattening parser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import requests

try:
    from neuroglp.data.openfda_client import OpenFDAClient, build_openfda_query
except ImportError:
    try:
        from neuroglp.data.openfda_client import OpenFDAClient
        build_openfda_query = getattr(OpenFDAClient, "build_query", None)
    except ImportError:
        OpenFDAClient = None
        build_openfda_query = None

try:
    from neuroglp.data.parser import flatten_reports, parse_patient_sex
    try:
        from neuroglp.data.parser import parse_age_to_years as parse_patient_age
    except ImportError:
        from neuroglp.data.parser import parse_patient_age
except ImportError:
    flatten_reports = None
    parse_patient_age = None
    parse_patient_sex = None


# =====================================================================
# Fixture & Payload Contract Tests (Always Active)
# =====================================================================

def test_mock_openfda_payload_fixture_structure(mock_openfda_response: Dict[str, Any]) -> None:
    """Verifies mock OpenFDA JSON fixture has valid metadata and 80 results records."""
    assert "meta" in mock_openfda_response
    assert "results" in mock_openfda_response
    assert mock_openfda_response["meta"]["results"]["total"] == 7074
    assert len(mock_openfda_response["results"]) == 80
    rec0 = mock_openfda_response["results"][0]
    assert "safetyreportid" in rec0
    assert "patient" in rec0
    assert "drug" in rec0["patient"]
    assert "reaction" in rec0["patient"]


def test_sample_flattened_df_fixture_contract(sample_flattened_df: pd.DataFrame) -> None:
    """Verifies sample_flattened_df fixture satisfies M1 ↔ M2 interface contract."""
    expected_cols = [
        "safetyreportid", "patientonsetage", "patientsex", "serious",
        "seriousnesshospitalization", "suspect_drugs", "concomitant_drugs", "reactions"
    ]
    for col in expected_cols:
        assert col in sample_flattened_df.columns
    assert len(sample_flattened_df) == 80
    assert np.isnan(sample_flattened_df["patientonsetage"]).sum() == 0
    assert set(sample_flattened_df["patientsex"].unique()).issubset({0, 1, 2})


@pytest.fixture(autouse=True)
def _check_ingestion_implemented(request: pytest.FixtureRequest) -> None:
    if request.node.name.startswith(("test_fixture", "test_mock", "test_sample", "test_synthetic")):
        return
    if OpenFDAClient is None or flatten_reports is None:
        pytest.skip("neuroglp.data is not yet implemented")


# =====================================================================
# Tier 1 & 2: Query Builder Tests
# =====================================================================

def test_query_builder_default_contains_mandatory_glp1_and_reactions() -> None:
    """Verifies default query covers both GLP-1 generics, brand names, and MedDRA psychiatric PTs."""
    query = build_openfda_query()
    
    # Assert primary generic agents
    assert "SEMAGLUTIDE" in query.upper()
    assert "TIRZEPATIDE" in query.upper()
    
    # Assert critical brand names (especially Wegovy in medicinalproduct)
    assert "OZEMPIC" in query.upper()
    assert "WEGOVY" in query.upper()
    assert "MOUNJARO" in query.upper()
    assert "ZEPBOUND" in query.upper()
    assert "RYBELSUS" in query.upper()

    # Assert exact MedDRA psychiatric Preferred Terms
    assert "SUICIDAL+IDEATION" in query.upper() or "SUICIDAL IDEATION" in query.upper()
    assert "DEPRESSION" in query.upper()
    assert "ANXIETY" in query.upper()


def test_query_builder_syntax_and_operators() -> None:
    """Verifies query builder creates balanced parentheses and valid Boolean operators."""
    query = build_openfda_query()
    
    # Check balanced parentheses
    assert query.count("(") == query.count(")")
    # Check presence of conjunction AND
    assert "+AND+" in query or " AND " in query
    # Check exact MedDRA field specifier
    assert "reactionmeddrapt.exact" in query


def test_query_builder_custom_parameters() -> None:
    """Verifies query builder with custom drug and reaction inputs."""
    custom_drugs = ["SEMAGLUTIDE"]
    custom_brands = ["OZEMPIC"]
    custom_pts = ["DEPRESSION"]
    
    query = build_openfda_query(drugs=custom_drugs, brands=custom_brands, psychiatric_pts=custom_pts)
    assert "SEMAGLUTIDE" in query
    assert "OZEMPIC" in query
    assert "DEPRESSION" in query
    assert "TIRZEPATIDE" not in query


def _make_client(**kwargs) -> Any:
    try:
        return OpenFDAClient(**kwargs)
    except TypeError:
        try:
            from neuroglp.data.openfda_client import OpenFDAConfig
            return OpenFDAClient(config=OpenFDAConfig(**kwargs))
        except Exception:
            return OpenFDAClient()


def test_api_client_fetch_page_success(mock_openfda_response: Dict[str, Any]) -> None:
    """Verifies API client returns records on HTTP 200."""
    client = _make_client(rate_limit_delay=0.0)
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_openfda_response
    mock_resp.raise_for_status.return_value = None

    session = getattr(client, "session", getattr(client, "_session", None))
    if session is None and hasattr(client, "_get_session"):
        session = client._get_session()

    with patch.object(session, "get", return_value=mock_resp) as mock_get:
        if hasattr(client, "fetch_reports"):
            records = client.fetch_reports(max_records=80, use_cache=False)
            assert len(records) > 0
        elif hasattr(client, "fetch_adverse_events"):
            records, total = client.fetch_adverse_events(limit=10, skip=0)
            assert len(records) == 80
            assert total == 7074


def test_api_client_not_found_returns_empty_list() -> None:
    """OpenFDA returns HTTP 404 with NOT_FOUND when 0 matches exist. Client must return [] without raising."""
    client = _make_client(rate_limit_delay=0.0)
    
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {"error": {"code": "NOT_FOUND", "message": "No matches found!"}}
    
    http_error = requests.HTTPError(response=mock_resp)
    mock_resp.raise_for_status.side_effect = http_error

    session = getattr(client, "session", getattr(client, "_session", None))
    if session is None and hasattr(client, "_get_session"):
        session = client._get_session()

    with patch.object(session, "get", return_value=mock_resp):
        if hasattr(client, "fetch_reports"):
            records = client.fetch_reports(query="nonexistent_xyz", use_cache=False)
            assert records == []
        elif hasattr(client, "fetch_adverse_events"):
            records, total = client.fetch_adverse_events(search_query="nonexistent_xyz")
            assert records == []


def test_api_client_rate_limit_backoff_retry() -> None:
    """Verifies that HTTP 429 Too Many Requests triggers retry and succeeds on subsequent try."""
    client = _make_client(rate_limit_delay=0.0, max_retries=3)
    
    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.raise_for_status.side_effect = requests.HTTPError(response=mock_429)

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {"meta": {"results": {"total": 1, "skip": 0, "limit": 1}}, "results": [{"safetyreportid": "R1"}]}
    mock_200.raise_for_status.return_value = None

    session = getattr(client, "session", getattr(client, "_session", None))
    if session is None and hasattr(client, "_get_session"):
        session = client._get_session()

    with patch.object(session, "get", side_effect=[mock_429, mock_200]):
        with patch("time.sleep", return_value=None):
            if hasattr(client, "fetch_reports"):
                records = client.fetch_reports(max_records=1, use_cache=False)
                assert len(records) >= 1
            elif hasattr(client, "fetch_adverse_events"):
                records, total = client.fetch_adverse_events(limit=1)
                assert len(records) == 1


def test_api_client_pagination_accumulation(mock_openfda_response: Dict[str, Any]) -> None:
    """Verifies client paginates through multiple pages to collect requested target volume."""
    client = _make_client(rate_limit_delay=0.0)
    
    page1_records = mock_openfda_response["results"][:20]
    page2_records = mock_openfda_response["results"][20:40]
    
    resp1 = MagicMock(status_code=200, raise_for_status=lambda: None)
    resp1.json.return_value = {"meta": {"results": {"total": 40, "skip": 0, "limit": 20}}, "results": page1_records}
    
    resp2 = MagicMock(status_code=200, raise_for_status=lambda: None)
    resp2.json.return_value = {"meta": {"results": {"total": 40, "skip": 20, "limit": 20}}, "results": page2_records}

    session = getattr(client, "session", getattr(client, "_session", None))
    if session is None and hasattr(client, "_get_session"):
        session = client._get_session()

    with patch.object(session, "get", side_effect=[resp1, resp2]):
        if hasattr(client, "fetch_reports"):
            records = client.fetch_reports(max_records=40, use_cache=False)
            assert len(records) == 40
        elif hasattr(client, "fetch_cohort"):
            records = client.fetch_cohort(target_count=40, batch_size=20)
            assert len(records) == 40


def test_api_client_caching_and_offline_fallback(tmp_path: Path, sample_raw_records: List[Dict[str, Any]]) -> None:
    """Verifies saving to disk cache, loading from cache, and fallback on network failure."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = _make_client(cache_dir=cache_dir, rate_limit_delay=0.0)
    
    # 1. Save cache (either via _save_cache or save_cache)
    cache_file = cache_dir / "test_cache.json"
    if hasattr(client, "_save_cache"):
        client._save_cache(sample_raw_records, cache_path=cache_file)
        assert cache_file.exists() or list(cache_dir.glob("*.json*"))
    elif hasattr(client, "save_cache"):
        client.save_cache(sample_raw_records, path=str(cache_file))
        assert cache_file.exists()

    # 2. Offline fallback
    if hasattr(client, "fetch_reports"):
        with patch.object(client, "_load_offline_fallback", return_value=sample_raw_records):
            records = client.fetch_reports(offline=True)
            assert len(records) == len(sample_raw_records)
    elif hasattr(client, "fetch_cohort"):
        session = getattr(client, "session", getattr(client, "_session", None))
        with patch.object(session, "get", side_effect=requests.ConnectionError("Offline mode")):
            records = client.fetch_cohort(target_count=50, allow_cache_fallback=True)
            assert len(records) > 0


# =====================================================================
# Tier 1 & 2: Record Flattening & Parsing Tests
# =====================================================================

def test_flatten_reports_schema_and_types(sample_raw_records: List[Dict[str, Any]]) -> None:
    """Verifies flatten_reports satisfies the M1 ↔ M2 interface contract schema."""
    df = flatten_reports(sample_raw_records)
    
    expected_columns = [
        "safetyreportid",
        "patientonsetage",
        "patientsex",
        "serious",
        "seriousnesshospitalization",
        "suspect_drugs",
        "concomitant_drugs",
        "reactions"
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing required contract column: {col}"

    assert len(df) == len(sample_raw_records)
    assert df["safetyreportid"].dtype == object or df["safetyreportid"].dtype == "string"
    assert np.issubdtype(df["patientonsetage"].dtype, np.floating)
    assert np.issubdtype(df["patientsex"].dtype, np.integer)
    assert np.issubdtype(df["serious"].dtype, np.integer)
    assert np.issubdtype(df["seriousnesshospitalization"].dtype, np.integer)
    assert isinstance(df["suspect_drugs"].iloc[0], list)
    assert isinstance(df["concomitant_drugs"].iloc[0], list)
    assert isinstance(df["reactions"].iloc[0], list)


def test_parse_patient_age_units() -> None:
    """Verifies age unit conversions (years, decades, months, weeks, days)."""
    # 801: Years
    assert parse_patient_age("45", "801") == pytest.approx(45.0)
    # 800: Decades
    assert parse_patient_age("6", "800") == pytest.approx(60.0)
    # 802: Months
    assert parse_patient_age("24", "802") == pytest.approx(2.0)
    # 803: Weeks
    assert parse_patient_age("52", "803") == pytest.approx(1.0, rel=0.05)
    # 804: Days
    assert parse_patient_age("365", "804") == pytest.approx(1.0, rel=0.05)


def test_parse_patient_age_missing_and_outliers() -> None:
    """Verifies handling of missing age and out-of-range anomalies."""
    # Missing / None
    missing_age = parse_patient_age(None, None, default_age=50.0)
    assert missing_age == pytest.approx(50.0)
    
    # Empty string
    assert parse_patient_age("", "801", default_age=50.0) == pytest.approx(50.0)
    
    # Negative age
    assert parse_patient_age("-5", "801", default_age=50.0) == pytest.approx(50.0)
    
    # Extreme age > 115
    assert parse_patient_age("150", "801", default_age=50.0) == pytest.approx(50.0)


def test_parse_patient_sex() -> None:
    """Verifies sex mapping: 1=Male, 2=Female, 0=Unknown."""
    assert parse_patient_sex("1") == 1
    assert parse_patient_sex(1) == 1
    assert parse_patient_sex("2") == 2
    assert parse_patient_sex(2) == 2
    assert parse_patient_sex("0") == 0
    assert parse_patient_sex(None) == 0
    assert parse_patient_sex("UNKNOWN") == 0


def test_flatten_reports_drug_characterization_separation() -> None:
    """Verifies suspect GLP-1 drugs ('1') are segregated from concomitant co-medications ('2'/'3')."""
    raw_record = [{
        "safetyreportid": "TEST-001",
        "serious": "1",
        "seriousnesshospitalization": "1",
        "patient": {
            "patientonsetage": "50",
            "patientonsetageunit": "801",
            "patientsex": "2",
            "drug": [
                {
                    "medicinalproduct": "OZEMPIC",
                    "drugcharacterization": "1",
                    "openfda": {"generic_name": ["SEMAGLUTIDE"]}
                },
                {
                    "medicinalproduct": "SERTRALINE",
                    "drugcharacterization": "2",
                    "openfda": {"generic_name": ["SERTRALINE"]}
                },
                {
                    "medicinalproduct": "METFORMIN",
                    "drugcharacterization": "3",
                    "openfda": {"generic_name": ["METFORMIN"]}
                }
            ],
            "reaction": [{"reactionmeddrapt": "SUICIDAL IDEATION"}]
        }
    }]

    df = flatten_reports(raw_record)
    assert len(df) == 1
    suspect = [d.upper() for d in df["suspect_drugs"].iloc[0]]
    concomitant = [d.upper() for d in df["concomitant_drugs"].iloc[0]]
    
    assert any("SEMAGLUTIDE" in s or "OZEMPIC" in s for s in suspect)
    assert not any("SERTRALINE" in s for s in suspect)
    assert any("SERTRALINE" in c for c in concomitant)
    assert any("METFORMIN" in c for c in concomitant)


def test_flatten_reports_empty_and_corrupt_records() -> None:
    """Boundary test: empty lists or malformed records must not raise exceptions."""
    # Empty list
    df_empty = flatten_reports([])
    assert isinstance(df_empty, pd.DataFrame)
    assert len(df_empty) == 0
    assert "safetyreportid" in df_empty.columns
    assert "patientonsetage" in df_empty.columns

    # Malformed record without 'patient' dictionary
    malformed = [{"safetyreportid": "MALFORMED-1", "serious": "2"}]
    df_malformed = flatten_reports(malformed)
    assert len(df_malformed) == 1
    assert df_malformed["safetyreportid"].iloc[0] == "MALFORMED-1"
    assert df_malformed["patientsex"].iloc[0] == 0
    assert df_malformed["suspect_drugs"].iloc[0] == []
    assert df_malformed["reactions"].iloc[0] == []
