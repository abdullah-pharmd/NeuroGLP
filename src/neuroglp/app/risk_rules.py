"""Clinical Polypharmacy Risk Rules Engine.

Evaluates cluster drug enrichment profiles against high-risk clinical pharmacology
and pharmacovigilance interaction heuristics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

POLYPHARMACY_RULES = [
    {
        "rule_id": "SSRI_SNRI",
        "title": "SSRI/SNRI Co-Prescription Alert: Pharmacokinetic Bioavailability Shifts",
        "threshold": 20.0,
        "drugs": [
            "sertraline",
            "escitalopram",
            "citalopram",
            "fluoxetine",
            "paroxetine",
            "venlafaxine",
            "duloxetine",
            "fluvoxamine",
        ],
        "severity": "HIGH",
        "mechanism": "Delayed gastric emptying from GLP-1 alters oral antidepressant pharmacokinetic absorption and peak concentrations, precipitating acute mood destabilization or withdrawal-like psychiatric decompensation.",
        "action": "Monitor therapeutic drug levels and mental status closely; consider non-oral formulations or dosing adjustments during GLP-1 dose escalation.",
    },
    {
        "rule_id": "CONTRACEPTIVE",
        "title": "Oral Contraceptive Efficacy Reduction Alert: Ovulation Breakthrough Risk",
        "threshold": 10.0,
        "drugs": [
            "ethinyl estradiol",
            "levonorgestrel",
            "drospirenone",
            "norethindrone",
            "norgestimate",
            "desogestrel",
            "contraceptive",
        ],
        "severity": "HIGH",
        "mechanism": "GLP-1-mediated delayed gastric emptying impairs Cmax and absorption kinetics of oral hormonal contraceptives, risking breakthrough ovulation and unintended pregnancy.",
        "action": "Advise barrier backup contraception for 4 weeks post-initiation and after each subsequent dose increase (CDC & FDA warning).",
    },
    {
        "rule_id": "ANTIDIABETIC",
        "title": "Severe Hypoglycemia & Neuroglycopenic Crisis Risk",
        "threshold": 25.0,
        "drugs": [
            "metformin",
            "insulin",
            "insulin glargine",
            "glimepiride",
            "glipizide",
            "glyburide",
            "pioglitazone",
        ],
        "severity": "MEDIUM",
        "mechanism": "Synergistic glucose-lowering with sulfonylureas or exogenous insulins increases incidence of profound neuroglycopenic events masquerading as acute agitation or anxiety.",
        "action": "Preemptively down-titrate background secretagogue or insulin doses by 20-50% upon GLP-1 initiation to prevent severe hypoglycemic episodes.",
    },
    {
        "rule_id": "BUPROPION",
        "title": "Seizure & Neuropsychiatric Agitation Risk",
        "threshold": 8.0,
        "drugs": [
            "bupropion",
            "methylphenidate",
            "amphetamine",
            "lisdexamfetamine",
            "atomoxetine",
        ],
        "severity": "CRITICAL",
        "mechanism": "Dual dopamine/norepinephrine reuptake inhibition combined with metabolic shifts elevates neuro-excitability, agitation, insomnia, and lowers seizure threshold.",
        "action": "Screen for history of seizure disorder or severe panic disorders; avoid rapid upward titration and monitor for hyper-adrenergic symptoms.",
    },
]


def evaluate_polypharmacy_risks(
    top_drugs: List[Tuple[str, float, float]]
) -> List[Dict[str, Any]]:
    """Evaluate cluster top drug list against clinical polypharmacy alert rules.

    Parameters
    ----------
    top_drugs : List[Tuple[str, float, float]]
        List of tuples formatted as (drug_name, prevalence_percentage, enrichment_ratio).

    Returns
    -------
    List[Dict[str, Any]]
        List of triggered polypharmacy risk alert dictionaries.
    """
    triggered_alerts: List[Dict[str, Any]] = []

    # Map drug name (lowercased) to max prevalence
    drug_prev_map: Dict[str, float] = {}
    for item in top_drugs:
        if len(item) >= 2:
            d_name = str(item[0]).lower().strip()
            prev = float(item[1])
            drug_prev_map[d_name] = max(drug_prev_map.get(d_name, 0.0), prev)

    for rule in POLYPHARMACY_RULES:
        rule_threshold = rule["threshold"]
        matching_drugs = []
        max_matched_prev = 0.0

        for candidate_drug, prev in drug_prev_map.items():
            for target in rule["drugs"]:
                if target in candidate_drug or candidate_drug in target:
                    if prev >= rule_threshold:
                        matching_drugs.append((candidate_drug, prev))
                        max_matched_prev = max(max_matched_prev, prev)

        if matching_drugs:
            alert = dict(rule)
            alert["matched_drugs"] = matching_drugs
            alert["highest_prevalence"] = max_matched_prev
            triggered_alerts.append(alert)

    return triggered_alerts
