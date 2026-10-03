"""
src/neuroglp/data/parser.py

Production-grade parser and flattener for OpenFDA FAERS adverse event JSON records.
Transforms deeply nested semi-structured FAERS JSON into a high-fidelity, zero-loss
structured Pandas DataFrame adhering to PROJECT.md § Interface Contracts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Known GLP-1 Receptor Agonist brand and generic identifiers
GLP1_IDENTIFIERS: Dict[str, str] = {
    # Semaglutide entities
    "SEMAGLUTIDE": "SEMAGLUTIDE",
    "OZEMPIC": "SEMAGLUTIDE",
    "WEGOVY": "SEMAGLUTIDE",
    "RYBELSUS": "SEMAGLUTIDE",
    # Tirzepatide entities
    "TIRZEPATIDE": "TIRZEPATIDE",
    "MOUNJARO": "TIRZEPATIDE",
    "ZEPBOUND": "TIRZEPATIDE",
}

# ICH E2B Age Unit Conversion Factors to Standard Years
# 800: Decades (1 decade = 10.0 years)
# 801: Years (1 year = 1.0 years)
# 802: Months (1 month = 1/12 years)
# 803: Weeks (1 week = 7/365.25 years ~ 1/52 years)
# 804: Days (1 day = 1/365.25 years)
# 805: Hours (1 hour = 1/8766.0 years)
AGE_UNIT_TO_YEARS: Dict[str, float] = {
    "800": 10.0,
    "801": 1.0,
    "802": 1.0 / 12.0,
    "803": 7.0 / 365.25,
    "804": 1.0 / 365.25,
    "805": 1.0 / 8766.0,
}

# Biological age plausibility window (in years)
MIN_PLAUSIBLE_AGE = 0.0
MAX_PLAUSIBLE_AGE = 120.0

# Empirical cohort default age for GLP-1 psychiatric adverse events
DEFAULT_COHORT_MEDIAN_AGE = 55.0


def parse_patient_age(
    raw_age: Any,
    raw_unit: Any = None,
    default_age: Optional[float] = None,
    min_age: float = MIN_PLAUSIBLE_AGE,
    max_age: float = MAX_PLAUSIBLE_AGE,
) -> float:
    """
    Parses and standardizes OpenFDA patient age to decimal years.

    Parameters
    ----------
    raw_age : Any
        Raw age string, float, or int from 'patientonsetage'.
    raw_unit : Any, optional
        Raw unit code from 'patientonsetageunit' (e.g. '800', '801', '802', '803', '804', '805').
    default_age : Optional[float]
        Value to return if age is missing, non-numeric, or outside [min_age, max_age].
        If None, returns np.nan.
    min_age : float, default 0.0
        Minimum biologically plausible age.
    max_age : float, default 120.0
        Maximum biologically plausible age.

    Returns
    -------
    float
        Standardized age in years, or default_age / np.nan if invalid.
    """
    fallback = float(default_age) if default_age is not None else np.nan

    if raw_age is None:
        return fallback

    age_str = str(raw_age).strip()
    if not age_str or age_str.upper() in {"UNK", "UNKNOWN", "NONE", "NULL", "NAN"}:
        return fallback

    try:
        val = float(age_str)
    except (ValueError, TypeError):
        return fallback

    # Negative age check
    if val < 0:
        return fallback

    # Parse unit code (default to '801' = years if missing or unspecified)
    unit_str = str(raw_unit).strip() if raw_unit is not None else "801"
    factor = AGE_UNIT_TO_YEARS.get(unit_str, 1.0)
    age_in_years = val * factor

    if age_in_years < min_age or age_in_years > max_age:
        return fallback

    return float(age_in_years)


def parse_age_to_years(
    raw_age: Any,
    raw_unit: Any = None,
    default_age: Optional[float] = None,
    min_age: float = MIN_PLAUSIBLE_AGE,
    max_age: float = MAX_PLAUSIBLE_AGE,
) -> float:
    """
    Standardizes patient age to decimal years, returning default_age or np.nan if invalid.
    """
    return parse_patient_age(
        raw_age,
        raw_unit,
        default_age=default_age,
        min_age=min_age,
        max_age=max_age,
    )


# Alias for backward compatibility
convert_age_to_years = parse_age_to_years


def parse_patient_sex(raw_sex: Any) -> int:
    """
    Parses OpenFDA patient sex code into standardized integer categories.

    Conforms to ICH E2B and PROJECT.md specifications:
    - 1: Male
    - 2: Female
    - 0: Unknown / unspecified

    Parameters
    ----------
    raw_sex : Any
        Raw sex field from 'patientsex'.

    Returns
    -------
    int
        1 (Male), 2 (Female), or 0 (Unknown).
    """
    if raw_sex is None:
        return 0

    sex_str = str(raw_sex).strip().upper()
    if sex_str in {"1", "MALE", "M"}:
        return 1
    elif sex_str in {"2", "FEMALE", "F"}:
        return 2
    return 0


def parse_outcome_flags(rec: Dict[str, Any]) -> Dict[str, int]:
    """
    Parses clinical outcome and seriousness criteria flags from a FAERS record.

    Parameters
    ----------
    rec : Dict[str, Any]
        Raw OpenFDA report dictionary.

    Returns
    -------
    Dict[str, int]
        Dictionary of standardized binary outcome indicators (1 or 0) and overall
        seriousness flag (1=serious, 2=non-serious).
    """
    def _to_bin(field_name: str) -> int:
        val = rec.get(field_name)
        if val is None:
            return 0
        return 1 if str(val).strip() == "1" else 0

    hosp = _to_bin("seriousnesshospitalization")
    death = _to_bin("seriousnessdeath")
    life = _to_bin("seriousnesslifethreatening")
    dis = _to_bin("seriousnessdisabling")
    cong = _to_bin("seriousnesscongenitalanomali")
    other = _to_bin("seriousnessother")

    raw_serious = rec.get("serious")
    if raw_serious is not None:
        ser_str = str(raw_serious).strip()
        if ser_str == "1":
            serious = 1
        elif ser_str == "2":
            serious = 2
        else:
            serious = 1 if (hosp or death or life or dis or cong or other) else 2
    else:
        serious = 1 if (hosp or death or life or dis or cong or other) else 2

    return {
        "serious": serious,
        "seriousnesshospitalization": hosp,
        "seriousnessdeath": death,
        "seriousnesslifethreatening": life,
        "seriousnessdisabling": dis,
        "seriousnesscongenitalanomali": cong,
        "seriousnessother": other,
    }


def extract_drug_name(drug_dict: Dict[str, Any]) -> Optional[str]:
    """
    Extracts the most specific and standardized drug name candidate from a FAERS drug object.

    Extraction priority hierarchy:
    1. openfda.generic_name[0]
    2. openfda.substance_name[0]
    3. openfda.brand_name[0]
    4. medicinalproduct
    5. activesubstance.activesubstancename

    Parameters
    ----------
    drug_dict : Dict[str, Any]
        Dictionary representing a single drug entry in 'patient.drug'.

    Returns
    -------
    Optional[str]
        Cleaned drug name string, or None if no valid name found.
    """
    if not isinstance(drug_dict, dict):
        return None

    openfda = drug_dict.get("openfda")
    if isinstance(openfda, dict):
        generic = openfda.get("generic_name")
        if isinstance(generic, list) and len(generic) > 0 and generic[0]:
            name = str(generic[0]).strip()
            if name:
                return name

        substance = openfda.get("substance_name")
        if isinstance(substance, list) and len(substance) > 0 and substance[0]:
            name = str(substance[0]).strip()
            if name:
                return name

        brand = openfda.get("brand_name")
        if isinstance(brand, list) and len(brand) > 0 and brand[0]:
            name = str(brand[0]).strip()
            if name:
                return name

    mp = drug_dict.get("medicinalproduct")
    if mp and isinstance(mp, str):
        name = mp.strip()
        if name:
            return name

    act = drug_dict.get("activesubstance")
    if isinstance(act, dict):
        act_name = act.get("activesubstancename")
        if act_name and isinstance(act_name, str):
            name = act_name.strip()
            if name:
                return name

    return None


def classify_patient_drugs(
    drug_list: List[Dict[str, Any]]
) -> Tuple[List[str], List[str], str]:
    """
    Extracts, deduplicates, and classifies patient drugs into suspect GLP-1 drugs
    and concomitant co-medications.

    Distinguishes roles using both FAERS drugcharacterization codes
    ('1'=Suspect, '2'=Concomitant, '3'=Interacting) and GLP-1 entity matching.

    Parameters
    ----------
    drug_list : List[Dict[str, Any]]
        List of drug dictionaries from 'patient.drug'.

    Returns
    -------
    Tuple[List[str], List[str], str]
        - suspect_drugs: List of unique suspect drug names (primarily GLP-1 agents).
        - concomitant_drugs: List of unique unnormalized co-medication names.
        - index_drug: Primary GLP-1 active moiety ('SEMAGLUTIDE', 'TIRZEPATIDE', or 'UNKNOWN').
    """
    if not isinstance(drug_list, list) or len(drug_list) == 0:
        return [], [], "UNKNOWN"

    suspect_drugs: List[str] = []
    concomitant_drugs: List[str] = []
    index_drug: str = "UNKNOWN"

    for d in drug_list:
        if not isinstance(d, dict):
            continue

        raw_char = str(d.get("drugcharacterization", "")).strip()
        name = extract_drug_name(d)
        if not name:
            continue

        name_upper = name.upper()

        # Check if this drug is a GLP-1 receptor agonist
        matched_glp1_family: Optional[str] = None
        for keyword, family in GLP1_IDENTIFIERS.items():
            if keyword in name_upper:
                matched_glp1_family = family
                break

        if matched_glp1_family is not None:
            suspect_drugs.append(name)
            if index_drug == "UNKNOWN":
                index_drug = matched_glp1_family
        else:
            # Non-GLP1 co-medication
            concomitant_drugs.append(name)
            # If the physician specifically flagged this non-GLP1 co-medication
            # as suspect ('1'), also record in suspect_drugs for clinical auditability.
            if raw_char == "1":
                suspect_drugs.append(name)

    # Deduplicate while preserving encounter order
    suspect_dedup = list(dict.fromkeys(suspect_drugs))
    concomitant_dedup = list(dict.fromkeys(concomitant_drugs))

    return suspect_dedup, concomitant_dedup, index_drug


# Alias for backward compatibility
extract_reported_drugs = classify_patient_drugs


def extract_reactions(reaction_list: List[Dict[str, Any]]) -> List[str]:
    """
    Extracts deduplicated MedDRA Preferred Terms from a patient's reaction records.

    Parameters
    ----------
    reaction_list : List[Dict[str, Any]]
        List of reaction dictionaries from 'patient.reaction'.

    Returns
    -------
    List[str]
        List of unique MedDRA PT strings in order of encounter.
    """
    if not isinstance(reaction_list, list) or len(reaction_list) == 0:
        return []

    reactions: List[str] = []
    for r in reaction_list:
        if not isinstance(r, dict):
            continue
        pt = r.get("reactionmeddrapt")
        if pt and isinstance(pt, str):
            clean_pt = pt.strip()
            if clean_pt:
                reactions.append(clean_pt)

    return list(dict.fromkeys(reactions))


def flatten_reports(
    raw_records: List[Dict[str, Any]],
    default_median_age: float = DEFAULT_COHORT_MEDIAN_AGE,
) -> pd.DataFrame:
    """
    Flattens deeply nested OpenFDA adverse drug event JSON records into a clean,
    structured Pandas DataFrame matching PROJECT.md § Interface Contracts.

    Handles:
    - Unique report identification and report version deduplication.
    - Patient age unit conversion (decades, months, weeks, days, hours -> years).
    - Median age imputation for missing or biologically implausible values.
    - Boolean tracking flag (`age_imputed`) indicating imputed ages.
    - Patient sex mapping (1=Male, 2=Female, 0=Unknown).
    - Clinical seriousness and hospitalization indicators.
    - Suspect GLP-1 drug identification and co-medication extraction.
    - Deduplicated MedDRA Preferred Terms (PT) extraction.
    - High-risk psychiatric adverse reaction indicator flags.
    - Guaranteed zero NaN values in contract columns.

    Parameters
    ----------
    raw_records : List[Dict[str, Any]]
        List of raw event dictionaries from OpenFDA API or local cache.
    default_median_age : float, default 55.0
        Fallback median age to use if all records in the batch have missing age.

    Returns
    -------
    pd.DataFrame
        Structured DataFrame with zero-loss schema adhering to PROJECT.md contracts.
    """
    columns = [
        "safetyreportid",
        "patientonsetage",
        "patientsex",
        "serious",
        "seriousnesshospitalization",
        "suspect_drugs",
        "concomitant_drugs",
        "reactions",
        # Zero-loss auxiliary columns
        "age_imputed",
        "index_drug",
        "seriousnessdeath",
        "seriousnesslifethreatening",
        "seriousnessdisabling",
        "has_suicidal_ideation",
        "has_depression",
        "has_anxiety",
    ]

    if not raw_records or not isinstance(raw_records, list):
        df_empty = pd.DataFrame(columns=columns)
        df_empty["safetyreportid"] = df_empty["safetyreportid"].astype(str)
        df_empty["patientonsetage"] = df_empty["patientonsetage"].astype(float)
        df_empty["patientsex"] = df_empty["patientsex"].astype(int)
        df_empty["serious"] = df_empty["serious"].astype(int)
        df_empty["seriousnesshospitalization"] = df_empty["seriousnesshospitalization"].astype(int)
        df_empty["age_imputed"] = df_empty["age_imputed"].astype(bool)
        df_empty["index_drug"] = df_empty["index_drug"].astype(str)
        df_empty["seriousnessdeath"] = df_empty["seriousnessdeath"].astype(int)
        df_empty["seriousnesslifethreatening"] = df_empty["seriousnesslifethreatening"].astype(int)
        df_empty["seriousnessdisabling"] = df_empty["seriousnessdisabling"].astype(int)
        df_empty["has_suicidal_ideation"] = df_empty["has_suicidal_ideation"].astype(int)
        df_empty["has_depression"] = df_empty["has_depression"].astype(int)
        df_empty["has_anxiety"] = df_empty["has_anxiety"].astype(int)
        return df_empty

    # Pass 1: Deduplicate reports by safetyreportid (retaining highest safetyreportversion)
    records_by_id: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for idx, rec in enumerate(raw_records):
        if not isinstance(rec, dict):
            continue

        raw_id = rec.get("safetyreportid")
        if raw_id is None or str(raw_id).strip() == "":
            sid = f"REPORT_{idx:08d}"
        else:
            sid = str(raw_id).strip()

        raw_ver = rec.get("safetyreportversion", "1")
        try:
            ver = int(str(raw_ver).strip())
        except (ValueError, TypeError):
            ver = 1

        if sid in records_by_id:
            existing_ver, _ = records_by_id[sid]
            if ver > existing_ver:
                records_by_id[sid] = (ver, rec)
        else:
            records_by_id[sid] = (ver, rec)

    unique_records = [rec for _, rec in records_by_id.values()]

    # Pass 2: Calculate decimal ages and determine cohort median for imputation
    raw_ages_years: List[float] = []
    for rec in unique_records:
        patient = rec.get("patient")
        if isinstance(patient, dict):
            raw_age = patient.get("patientonsetage")
            raw_unit = patient.get("patientonsetageunit")
            age_years = parse_age_to_years(raw_age, raw_unit)
        else:
            age_years = np.nan
        raw_ages_years.append(age_years)

    valid_ages = [a for a in raw_ages_years if not np.isnan(a)]
    if len(valid_ages) > 0:
        median_age = float(np.median(valid_ages))
    else:
        median_age = float(default_median_age)

    # Pass 3: Construct flat patient records
    parsed_rows: List[Dict[str, Any]] = []
    for idx, rec in enumerate(unique_records):
        sid = str(rec.get("safetyreportid", f"REPORT_{idx:08d}")).strip()
        patient = rec.get("patient") if isinstance(rec.get("patient"), dict) else {}

        # Standardized age & imputation flag
        age_val = raw_ages_years[idx]
        if np.isnan(age_val):
            final_age = float(round(median_age, 1))
            age_imputed = True
        else:
            final_age = float(round(age_val, 1))
            age_imputed = False

        # Sex (1=Male, 2=Female, 0=Unknown)
        final_sex = parse_patient_sex(patient.get("patientsex"))

        # Seriousness and clinical outcomes
        outcomes = parse_outcome_flags(rec)

        # Drugs and co-medications
        drugs_raw = patient.get("drug", [])
        suspects, concomitants, index_drug = classify_patient_drugs(drugs_raw)

        # MedDRA reactions
        reactions_raw = patient.get("reaction", [])
        reactions = extract_reactions(reactions_raw)

        # High-risk psychiatric endpoint flags
        rx_upper = " ".join(reactions).upper()
        has_suicide = (
            1
            if any(
                term in rx_upper
                for term in [
                    "SUICID",
                    "SELF-INJURY",
                    "SELF HARM",
                    "INTENTIONAL SELF-INJURY",
                ]
            )
            else 0
        )
        has_depression = (
            1 if any(term in rx_upper for term in ["DEPRESS", "DYSPHOR"]) else 0
        )
        has_anxiety = (
            1
            if any(
                term in rx_upper
                for term in ["ANXIET", "PANIC", "NERVOUSNESS", "AGITATION"]
            )
            else 0
        )

        parsed_rows.append(
            {
                "safetyreportid": sid,
                "patientonsetage": final_age,
                "patientsex": final_sex,
                "serious": outcomes["serious"],
                "seriousnesshospitalization": outcomes["seriousnesshospitalization"],
                "suspect_drugs": suspects,
                "concomitant_drugs": concomitants,
                "reactions": reactions,
                "age_imputed": age_imputed,
                "index_drug": index_drug,
                "seriousnessdeath": outcomes["seriousnessdeath"],
                "seriousnesslifethreatening": outcomes["seriousnesslifethreatening"],
                "seriousnessdisabling": outcomes["seriousnessdisabling"],
                "has_suicidal_ideation": has_suicide,
                "has_depression": has_depression,
                "has_anxiety": has_anxiety,
            }
        )

    df = pd.DataFrame(parsed_rows, columns=columns)

    # Strict type enforcement
    if not df.empty:
        df["safetyreportid"] = df["safetyreportid"].astype(str)
        df["patientonsetage"] = df["patientonsetage"].astype(float)
        df["patientsex"] = df["patientsex"].astype(int)
        df["serious"] = df["serious"].astype(int)
        df["seriousnesshospitalization"] = df["seriousnesshospitalization"].astype(int)
        df["age_imputed"] = df["age_imputed"].astype(bool)
        df["index_drug"] = df["index_drug"].astype(str)
        df["seriousnessdeath"] = df["seriousnessdeath"].astype(int)
        df["seriousnesslifethreatening"] = df["seriousnesslifethreatening"].astype(int)
        df["seriousnessdisabling"] = df["seriousnessdisabling"].astype(int)
        df["has_suicidal_ideation"] = df["has_suicidal_ideation"].astype(int)
        df["has_depression"] = df["has_depression"].astype(int)
        df["has_anxiety"] = df["has_anxiety"].astype(int)

    return df


def validate_flattened_schema(df: pd.DataFrame) -> bool:
    """
    Validates that a DataFrame strictly adheres to the M1 -> M2 contract schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by flatten_reports().

    Returns
    -------
    bool
        True if all contract assertions pass, raises AssertionError otherwise.
    """
    required_columns = {
        "safetyreportid",
        "patientonsetage",
        "patientsex",
        "serious",
        "seriousnesshospitalization",
        "suspect_drugs",
        "concomitant_drugs",
        "reactions",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise AssertionError(f"DataFrame is missing required contract columns: {missing}")

    if not df.empty:
        assert not df["safetyreportid"].isna().any(), "safetyreportid contains NaNs"
        assert not df["patientonsetage"].isna().any(), "patientonsetage contains NaNs"
        assert set(df["patientsex"].unique()).issubset({0, 1, 2}), "Invalid patientsex values"
        assert set(df["serious"].unique()).issubset({1, 2}), "Invalid serious values"
        assert set(df["seriousnesshospitalization"].unique()).issubset({0, 1}), (
            "Invalid seriousnesshospitalization values"
        )
        assert df["suspect_drugs"].apply(lambda x: isinstance(x, list)).all(), (
            "suspect_drugs must be a list of strings"
        )
        assert df["concomitant_drugs"].apply(lambda x: isinstance(x, list)).all(), (
            "concomitant_drugs must be a list of strings"
        )
        assert df["reactions"].apply(lambda x: isinstance(x, list)).all(), (
            "reactions must be a list of strings"
        )

    return True
