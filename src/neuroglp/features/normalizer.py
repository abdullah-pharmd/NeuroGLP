"""
src/neuroglp/features/normalizer.py

Production-grade pharmacovigilance drug name normalization and entity standardization engine.
Handles regex cleaning, salt stripping, dosage/strength removal, formulation stripping,
and comprehensive brand-to-generic mapping (>200 drug entities).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Union
import pandas as pd

logger = logging.getLogger(__name__)

# =====================================================================
# Comprehensive Brand-to-Generic Mapping Dictionary (>= 150 entries)
# =====================================================================

BRAND_TO_GENERIC: Dict[str, str] = {
    # -----------------------------------------------------------------
    # GLP-1 Receptor Agonists & Dual Agonists (Cohort & Comparator Entities)
    # -----------------------------------------------------------------
    "ozempic": "semaglutide",
    "wegovy": "semaglutide",
    "rybelsus": "semaglutide",
    "mounjaro": "tirzepatide",
    "zepbound": "tirzepatide",
    "trulicity": "dulaglutide",
    "victoza": "liraglutide",
    "saxenda": "liraglutide",
    "byetta": "exenatide",
    "bydureon": "exenatide",
    "bydureon bcise": "exenatide",
    "adlyxin": "lixisenatide",
    "tanzeum": "albiglutide",

    # -----------------------------------------------------------------
    # Antidepressants - SSRIs
    # -----------------------------------------------------------------
    "zoloft": "sertraline",
    "prozac": "fluoxetine",
    "sarafem": "fluoxetine",
    "lexapro": "escitalopram",
    "celexa": "citalopram",
    "paxil": "paroxetine",
    "paxil cr": "paroxetine",
    "pexeva": "paroxetine",
    "luvox": "fluvoxamine",
    "luvox cr": "fluvoxamine",
    "viibryd": "vilazodone",
    "trintellix": "vortioxetine",
    "brintellix": "vortioxetine",

    # -----------------------------------------------------------------
    # Antidepressants - SNRIs
    # -----------------------------------------------------------------
    "effexor": "venlafaxine",
    "effexor xr": "venlafaxine",
    "cymbalta": "duloxetine",
    "drizalma": "duloxetine",
    "pristiq": "desvenlafaxine",
    "khedezla": "desvenlafaxine",
    "fetzima": "levomilnacipran",
    "savella": "milnacipran",

    # -----------------------------------------------------------------
    # Antidepressants - NDRI / TeCA / Serotonin Modulators / Atypicals
    # -----------------------------------------------------------------
    "wellbutrin": "bupropion",
    "wellbutrin sr": "bupropion",
    "wellbutrin xl": "bupropion",
    "zyban": "bupropion",
    "aplenzin": "bupropion",
    "forfivo xl": "bupropion",
    "remeron": "mirtazapine",
    "remeron soltab": "mirtazapine",
    "desyrel": "trazodone",
    "oleptro": "trazodone",
    "serzone": "nefazodone",

    # -----------------------------------------------------------------
    # Tricyclic Antidepressants (TCAs)
    # -----------------------------------------------------------------
    "elavil": "amitriptyline",
    "pamelor": "nortriptyline",
    "tofranil": "imipramine",
    "anafranil": "clomipramine",
    "sinequan": "doxepin",
    "silenor": "doxepin",
    "norpramin": "desipramine",
    "vivactil": "protriptyline",
    "surmontil": "trimipramine",
    "ludiomil": "maprotiline",

    # -----------------------------------------------------------------
    # MAOIs
    # -----------------------------------------------------------------
    "nardil": "phenelzine",
    "parnate": "tranylcypromine",
    "marplan": "isocarboxazid",
    "emsam": "selegiline",

    # -----------------------------------------------------------------
    # Antipsychotics (First & Second Generation)
    # -----------------------------------------------------------------
    "seroquel": "quetiapine",
    "seroquel xr": "quetiapine",
    "abilify": "aripiprazole",
    "abilify maintena": "aripiprazole",
    "zyprexa": "olanzapine",
    "zyprexa zydis": "olanzapine",
    "risperdal": "risperidone",
    "risperdal consta": "risperidone",
    "geodon": "ziprasidone",
    "latuda": "lurasidone",
    "rexulti": "brexpiprazole",
    "vraylar": "cariprazine",
    "invega": "paliperidone",
    "invega sustenna": "paliperidone",
    "clozaril": "clozapine",
    "fazaclo": "clozapine",
    "haldol": "haloperidol",
    "trilafon": "perphenazine",
    "thorazine": "chlorpromazine",
    "stelazine": "trifluoperazine",
    "prolixin": "fluphenazine",
    "navane": "thiothixene",
    "saphris": "asenapine",
    "caplyta": "lumateperone",

    # -----------------------------------------------------------------
    # Mood Stabilizers & Anticonvulsants
    # -----------------------------------------------------------------
    "lamictal": "lamotrigine",
    "lamictal xr": "lamotrigine",
    "depakote": "divalproex",
    "depakote er": "divalproex",
    "divalproex sodium": "divalproex",
    "depakene": "valproic acid",
    "tegretol": "carbamazepine",
    "tegretol xr": "carbamazepine",
    "carbatrol": "carbamazepine",
    "trileptal": "oxcarbazepine",
    "oxtellar xr": "oxcarbazepine",
    "aptiom": "eslicarbazepine",
    "topamax": "topiramate",
    "trokendi xr": "topiramate",
    "neurontin": "gabapentin",
    "gralise": "gabapentin",
    "horizant": "gabapentin enacarbil",
    "lyrica": "pregabalin",
    "lyrica cr": "pregabalin",
    "keppra": "levetiracetam",
    "keppra xr": "levetiracetam",
    "vimpat": "lacosamide",
    "zonegran": "zonisamide",
    "dilantin": "phenytoin",
    "phenytek": "phenytoin",
    "mysoline": "primidone",
    "zarontin": "ethosuximide",
    "briviact": "brivaracetam",

    # -----------------------------------------------------------------
    # Anxiolytics, Hypnotics & Sedatives
    # -----------------------------------------------------------------
    "xanax": "alprazolam",
    "xanax xr": "alprazolam",
    "klonopin": "clonazepam",
    "ativan": "lorazepam",
    "valium": "diazepam",
    "restoril": "temazepam",
    "halcion": "triazolam",
    "versed": "midazolam",
    "tranxene": "clorazepate",
    "librium": "chlordiazepoxide",
    "ambien": "zolpidem",
    "ambien cr": "zolpidem",
    "lunesta": "eszopiclone",
    "sonata": "zaleplon",
    "belsomra": "suvorexant",
    "dayvigo": "lemborexant",
    "quviviq": "daridorexant",
    "buspar": "buspirone",
    "vistaril": "hydroxyzine",
    "atarax": "hydroxyzine",
    "rozerem": "ramelteon",

    # -----------------------------------------------------------------
    # ADHD & Psychostimulants / Wakefulness Agents
    # -----------------------------------------------------------------
    "adderall": "amphetamine / dextroamphetamine",
    "adderall xr": "amphetamine / dextroamphetamine",
    "ritalin": "methylphenidate",
    "ritalin la": "methylphenidate",
    "concerta": "methylphenidate",
    "focalin": "dexmethylphenidate",
    "focalin xr": "dexmethylphenidate",
    "vyvanse": "lisdexamfetamine",
    "strattera": "atomoxetine",
    "intuniv": "guanfacine",
    "kapvay": "clonidine",
    "qelbree": "viloxazine",
    "provigil": "modafinil",
    "nuvigil": "armodafinil",

    # -----------------------------------------------------------------
    # Antidiabetics - Biguanides & Sulfonylureas
    # -----------------------------------------------------------------
    "glucophage": "metformin",
    "glucophage xr": "metformin",
    "fortamet": "metformin",
    "glumetza": "metformin",
    "riomet": "metformin",
    "amaryl": "glimepiride",
    "glucotrol": "glipizide",
    "glucotrol xl": "glipizide",
    "micronase": "glyburide",
    "diabeta": "glyburide",
    "glynase": "glyburide",
    "prandin": "repaglinide",
    "starlix": "nateglinide",

    # -----------------------------------------------------------------
    # Antidiabetics - SGLT2i, DPP-4i, TZDs & Combinations
    # -----------------------------------------------------------------
    "jardiance": "empagliflozin",
    "farxiga": "dapagliflozin",
    "invokana": "canagliflozin",
    "steglatro": "ertugliflozin",
    "januvia": "sitagliptin",
    "tradjenta": "linagliptin",
    "onglyza": "saxagliptin",
    "nesina": "alogliptin",
    "actos": "pioglitazone",
    "avandia": "rosiglitazone",
    "janumet": "metformin / sitagliptin",
    "janumet xr": "metformin / sitagliptin",
    "synjardy": "empagliflozin / metformin",
    "xigduo xr": "dapagliflozin / metformin",
    "invokamet": "canagliflozin / metformin",
    "kombiglyze xr": "metformin / saxagliptin",
    "jentadueto": "linagliptin / metformin",

    # -----------------------------------------------------------------
    # Insulins
    # -----------------------------------------------------------------
    "lantus": "insulin glargine",
    "toujeo": "insulin glargine",
    "basaglar": "insulin glargine",
    "semglee": "insulin glargine",
    "levemir": "insulin detemir",
    "tresiba": "insulin degludec",
    "humalog": "insulin lispro",
    "novolog": "insulin aspart",
    "fiasp": "insulin aspart",
    "apidra": "insulin glulisine",
    "humulin": "insulin human",
    "novolin": "insulin human",

    # -----------------------------------------------------------------
    # Cardiovascular - ACE Inhibitors, ARBs, CCBs, Beta Blockers, Diuretics
    # -----------------------------------------------------------------
    "prinivil": "lisinopril",
    "zestril": "lisinopril",
    "vasotec": "enalapril",
    "altace": "ramipril",
    "capoten": "captopril",
    "lotensin": "benazepril",
    "accupril": "quinapril",
    "mavik": "trandolapril",
    "cozaar": "losartan",
    "diovan": "valsartan",
    "benicar": "olmesartan",
    "avapro": "irbesartan",
    "micardis": "telmisartan",
    "atacand": "candesartan",
    "norvasc": "amlodipine",
    "plendil": "felodipine",
    "procardia": "nifedipine",
    "procardia xl": "nifedipine",
    "adalat cc": "nifedipine",
    "cardizem": "diltiazem",
    "cardizem cd": "diltiazem",
    "tiazac": "diltiazem",
    "calan": "verapamil",
    "calan sr": "verapamil",
    "isoptin": "verapamil",
    "verelan": "verapamil",
    "lopressor": "metoprolol",
    "toprol xl": "metoprolol",
    "tenormin": "atenolol",
    "coreg": "carvedilol",
    "coreg cr": "carvedilol",
    "bystolic": "nebivolol",
    "inderal": "propranolol",
    "inderal la": "propranolol",
    "zebeta": "bisoprolol",
    "trandate": "labetalol",
    "lasix": "furosemide",
    "bumex": "bumetanide",
    "demadex": "torsemide",
    "aldactone": "spironolactone",
    "inspra": "eplerenone",
    "microzide": "hydrochlorothiazide",
    "thalitone": "chlorthalidone",
    "zaroxolyn": "metolazone",

    # -----------------------------------------------------------------
    # Cardiovascular - Statins & Lipid Regulators
    # -----------------------------------------------------------------
    "lipitor": "atorvastatin",
    "crestor": "rosuvastatin",
    "zocor": "simvastatin",
    "pravachol": "pravastatin",
    "mevacor": "lovastatin",
    "lescol": "fluvastatin",
    "livalo": "pitavastatin",
    "zetia": "ezetimibe",
    "vytorin": "ezetimibe / simvastatin",
    "tricor": "fenofibrate",
    "trilipix": "fenofibric acid",
    "lopid": "gemfibrozil",
    "niaspan": "niacin",
    "lovaza": "omega-3-acid ethyl esters",
    "vascepa": "icosapent ethyl",
    "repatha": "evolocumab",
    "praluent": "alirocumab",

    # -----------------------------------------------------------------
    # Antithrombotics & Anticoagulants
    # -----------------------------------------------------------------
    "plavix": "clopidogrel",
    "brilinta": "ticagrelor",
    "effient": "prasugrel",
    "eliquis": "apixaban",
    "xarelto": "rivaroxaban",
    "pradaxa": "dabigatran",
    "savaysa": "edoxaban",
    "coumadin": "warfarin",
    "jantoven": "warfarin",
    "lovenox": "enoxaparin",
    "fragmin": "dalteparin",

    # -----------------------------------------------------------------
    # Analgesics, NSAIDs & Pain Agents
    # -----------------------------------------------------------------
    "tylenol": "acetaminophen",
    "motrin": "ibuprofen",
    "advil": "ibuprofen",
    "aleve": "naproxen",
    "naprosyn": "naproxen",
    "anaprox": "naproxen",
    "celebrex": "celecoxib",
    "mobic": "meloxicam",
    "voltaren": "diclofenac",
    "cataflam": "diclofenac",
    "toradol": "ketorolac",
    "feldene": "piroxicam",
    "relafen": "nabumetone",
    "lodine": "etodolac",
    "ultram": "tramadol",
    "vicodin": "hydrocodone",
    "norco": "hydrocodone",
    "lortab": "hydrocodone",
    "percocet": "oxycodone",
    "oxycontin": "oxycodone",
    "roxicodone": "oxycodone",
    "dilaudid": "hydromorphone",
    "ms contin": "morphine",
    "duragesic": "fentanyl",

    # -----------------------------------------------------------------
    # Gastrointestinal (PPIs, H2 Blockers, Antiemetics)
    # -----------------------------------------------------------------
    "prilosec": "omeprazole",
    "prilosec otc": "omeprazole",
    "nexium": "esomeprazole",
    "prevacid": "lansoprazole",
    "protonix": "pantoprazole",
    "aciphex": "rabeprazole",
    "dexilant": "dexlansoprazole",
    "pepcid": "famotidine",
    "zantac": "ranitidine",
    "tagamet": "cimetidine",
    "carafate": "sucralfate",
    "reglan": "metoclopramide",
    "zofran": "ondansetron",
    "phenergan": "promethazine",
    "compazine": "prochlorperazine",
    "linzess": "linaclotide",

    # -----------------------------------------------------------------
    # Oral Contraceptives & Sex Hormones
    # -----------------------------------------------------------------
    "alesse": "ethinyl estradiol / levonorgestrel",
    "yasmin": "drospirenone / ethinyl estradiol",
    "yaz": "drospirenone / ethinyl estradiol",
    "ortho tri-cyclen": "ethinyl estradiol / norgestimate",
    "ortho-cyclen": "ethinyl estradiol / norgestimate",
    "seasonale": "ethinyl estradiol / levonorgestrel",
    "seasonique": "ethinyl estradiol / levonorgestrel",
    "loestrin": "ethinyl estradiol / norethindrone",
    "loestrin fe": "ethinyl estradiol / norethindrone",
    "nuvaring": "ethinyl estradiol / etonogestrel",
    "plan b": "levonorgestrel",
    "mirena": "levonorgestrel",
    "kyleena": "levonorgestrel",
    "skyla": "levonorgestrel",
    "premarin": "conjugated estrogens",
    "estrace": "estradiol",
    "climara": "estradiol",
    "prometrium": "progesterone",
    "provera": "medroxyprogesterone",
    "depo-provera": "medroxyprogesterone",
    "androgel": "testosterone",

    # -----------------------------------------------------------------
    # Thyroid & Endocrine
    # -----------------------------------------------------------------
    "synthroid": "levothyroxine",
    "levoxyl": "levothyroxine",
    "tirosint": "levothyroxine",
    "unithroid": "levothyroxine",
    "cytomel": "liothyronine",
    "tapazole": "methimazole",

    # -----------------------------------------------------------------
    # Respiratory & Allergy
    # -----------------------------------------------------------------
    "singulair": "montelukast",
    "ventolin": "albuterol",
    "proair": "albuterol",
    "proventil": "albuterol",
    "xopenex": "levalbuterol",
    "advair": "fluticasone / salmeterol",
    "symbicort": "budesonide / formoterol",
    "flonase": "fluticasone",
    "nasacort": "triamcinolone",
    "rhinocort": "budesonide",
    "zyrtec": "cetirizine",
    "claritin": "loratadine",
    "allegra": "fexofenadine",
    "xyzal": "levocetirizine",
    "benadryl": "diphenhydramine",
}

# Standalone dosage units that might linger after stripping
DOSAGE_UNITS: Set[str] = {
    "mg", "mcg", "ug", "µg", "g", "gm", "kg", "ml", "l", "cc", "unit", "units", "u", "iu", "%"
}

# Pre-compiled regular expressions for high-throughput batch normalization
RE_DOSAGE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|µg|g|gm|kg|ml|l|cc|units?|u|iu|meq|mmol|%)(?:\s*/\s*\d*(?:\.\d+)?\s*(?:mg|mcg|ug|µg|g|gm|kg|ml|l|cc|units?|u|iu|spray|actuation|dose|hr|day|24hr))?\b",
    re.IGNORECASE,
)

RE_STANDALONE_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")

RE_FORMS = re.compile(
    r"\b(?:tablets?|tabs?|capsules?|caps?|caplets?|pills?|injections?|injectable|inj|solutions?|soln?|suspensions?|susp|syrups?|creams?|ointments?|gels?|lotions?|patch(?:es)?|sprays?|inhalers?|drops?|suppositor(?:y|ies)|powders?)\b",
    re.IGNORECASE,
)

RE_RELEASE = re.compile(
    r"\b(?:delayed-release|delayed release|dr|extended-release|extended release|er|xr|xl|sustained-release|sustained release|sr|controlled-release|controlled release|cr|modified-release|modified release|mr|immediate-release|immediate release|ir|film-coated|enteric-coated|ec|long-acting|la|orally disintegrating|odt)\b",
    re.IGNORECASE,
)

RE_SALTS = re.compile(
    r"\b(?:hydrochloride|dihydrochloride|hcl|hydrobromide|hbr|sodium|sod|potassium|pot|calcium|cal|magnesium|maleate|tartrate|bitartrate|besylate|besilate|mesylate|mesilate|oxalate|succinate|fumarate|sulfate|sulphate|bisulfate|pamoate|acetate|citrate|phosphate|diphosphate|gluconate|lactate|stearate|valerate|propionate|dipropionate|valproate|tosylate|nitrate|chloride|bromide|carbonate|bicarbonate|monohydrate|dihydrate|trihydrate|hydrate)\b",
    re.IGNORECASE,
)

RE_COMBO_SPLIT = re.compile(r"\s+(?:and)\s+|\s*[/\\+&]\s*", re.IGNORECASE)


# =====================================================================
# Core Normalization Functions
# =====================================================================

def normalize_drug_name(raw_name: Any) -> str:
    """
    Standardizes a raw drug string by stripping salts, dosages, formulations,
    release modifiers, and mapping brand names to standardized generic entities.

    Parameters
    ----------
    raw_name : Any
        Raw drug input string from OpenFDA record (e.g. 'METFORMIN HCL 1000 MG TABLET', 'OZEMPIC').
        Safely handles None, NaN, numeric types, and empty strings.

    Returns
    -------
    str
        Standardized generic drug name in lower case, or empty string '' if input is invalid/empty.
    """
    if raw_name is None:
        return ""
    if not isinstance(raw_name, str):
        if pd.isna(raw_name):
            return ""
        raw_name = str(raw_name)

    text = raw_name.strip()
    if not text:
        return ""

    if text.lower() in {"nan", "none", "null", "unknown", "n/a", "na", "<na>"}:
        return ""

    # Check if string contains at least one alphanumeric character
    if not re.search(r"[a-zA-Z0-9]", text):
        return ""

    text = text.lower()

    # Pre-clean punctuation and brackets
    text = re.sub(r"[()\[\]{}\"\']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Fast-path: Check exact brand match before regex stripping
    if text in BRAND_TO_GENERIC:
        return BRAND_TO_GENERIC[text]

    # Strip dosage with units (e.g., 500mg, 100 units/ml)
    text = RE_DOSAGE.sub(" ", text)

    # Strip release modifiers (e.g., delayed-release, extended-release, dr, xr)
    text = RE_RELEASE.sub(" ", text)

    # Strip delivery formulations (e.g., tablet, capsule, injection)
    text = RE_FORMS.sub(" ", text)

    # Strip pharmaceutical salts (e.g., hydrochloride, hcl, oxalate, calcium)
    text = RE_SALTS.sub(" ", text)

    # Strip standalone numbers
    text = RE_STANDALONE_NUM.sub(" ", text)

    # Clean residual punctuation and normalize whitespace
    text = re.sub(r"[,./:;_\-+*?!#<>~]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Secondary check in brand mapping after stripping suffixes (e.g., 'glucophage xr' -> 'glucophage')
    if text in BRAND_TO_GENERIC:
        text = BRAND_TO_GENERIC[text]

    # Discard residual standalone units or empty strings
    if text in DOSAGE_UNITS or not text:
        return ""

    return text


def normalize_drug_list(
    drug_list: Any,
    split_combinations: bool = True,
    drop_empty: bool = True,
) -> List[str]:
    """
    Normalizes an iterable of raw drug strings into a deduplicated list of standardized drug entities.
    Handles combinations (e.g. 'ETHINYL ESTRADIOL / LEVONORGESTREL'), drops empty strings,
    and preserves original encounter order.

    Parameters
    ----------
    drug_list : Any
        Iterable of raw drug strings (e.g. from patient's 'concomitant_drugs').
    split_combinations : bool, default True
        Whether to split multi-active combination strings across '/', 'and', or '&'.
    drop_empty : bool, default True
        Whether to exclude empty strings resulting from normalization.

    Returns
    -------
    List[str]
        Deduplicated list of normalized drug generic entities.
    """
    if drug_list is None:
        return []
    if isinstance(drug_list, (str, bytes)):
        items = [str(drug_list)]
    elif isinstance(drug_list, (list, tuple, set)):
        items = list(drug_list)
    else:
        try:
            items = list(drug_list)
        except TypeError:
            return []

    normalized_set: List[str] = []
    seen: Set[str] = set()

    for item in items:
        if item is None or pd.isna(item):
            continue
        item_str = str(item).strip()
        if not item_str or item_str.lower() in {"nan", "none", "null", "unknown", "n/a", "na", "<na>"}:
            continue

        # Strip dosages first so compound units (e.g. 100 units/ml) are not split across slashes
        item_cleaned = RE_DOSAGE.sub(" ", item_str).strip()

        sub_items = [item_cleaned]
        if split_combinations:
            parts = RE_COMBO_SPLIT.split(item_cleaned)
            if len(parts) > 1:
                sub_items = parts

        for sub in sub_items:
            norm = normalize_drug_name(sub)
            if not norm and drop_empty:
                continue

            # If brand mapping itself resolved to a combination, split if requested
            if split_combinations and (" / " in norm or " and " in norm):
                for part in RE_COMBO_SPLIT.split(norm):
                    part_norm = normalize_drug_name(part)
                    if drop_empty and not part_norm:
                        continue
                    if part_norm in DOSAGE_UNITS:
                        continue
                    if part_norm not in seen:
                        seen.add(part_norm)
                        normalized_set.append(part_norm)
            else:
                if norm in DOSAGE_UNITS and drop_empty:
                    continue
                if norm not in seen:
                    seen.add(norm)
                    normalized_set.append(norm)

    return normalized_set


def is_known_drug(drug_name: str) -> bool:
    """Checks whether a given generic or brand name exists in the normalized dictionary."""
    if not drug_name:
        return False
    clean = drug_name.strip().lower()
    return clean in BRAND_TO_GENERIC or clean in BRAND_TO_GENERIC.values()


def get_brand_mapping(brand_name: str) -> Optional[str]:
    """Retrieves generic name for a brand name, returning None if unmapped."""
    if not brand_name:
        return None
    return BRAND_TO_GENERIC.get(brand_name.strip().lower())
