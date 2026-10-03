"""
OpenFDA API Ingestion Client for GLP-1 Pharmacovigilance.

Provides robust, zero-cost data ingestion for adverse drug event reports
from the public OpenFDA drug event database (https://api.fda.gov/drug/event.json),
with exponential backoff retries, polite rate limiting, pagination,
local disk caching (JSON / Gzip JSON), and offline fallback.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# =====================================================================
# Constants & Default Query Parameters
# =====================================================================

OPENFDA_EVENT_ENDPOINT: str = "https://api.fda.gov/drug/event.json"

DEFAULT_GLP1_GENERICS: List[str] = [
    "SEMAGLUTIDE",
    "TIRZEPATIDE",
]

DEFAULT_GLP1_BRANDS: List[str] = [
    "OZEMPIC",
    "WEGOVY",
    "RYBELSUS",
    "MOUNJARO",
    "ZEPBOUND",
]

DEFAULT_PSYCHIATRIC_MEDDRA_PTS: List[str] = [
    "SUICIDAL IDEATION",
    "DEPRESSION",
    "ANXIETY",
    "SUICIDAL BEHAVIOUR",
    "DEPRESSION SUICIDAL",
    "INTENTIONAL SELF-INJURY",
    "COMPLETED SUICIDE",
]

# Additional high-risk psychiatric endpoints for broader analysis
EXTENDED_PSYCHIATRIC_MEDDRA_PTS: List[str] = [
    "SUICIDE ATTEMPT",
    "DEPRESSED MOOD",
]


# =====================================================================
# Exceptions
# =====================================================================

class OpenFDAError(Exception):
    """Base exception for all OpenFDA client errors."""
    pass


class OpenFDAQueryError(OpenFDAError):
    """Raised when an OpenFDA query syntax is malformed or invalid."""
    pass


class OpenFDARateLimitError(OpenFDAError):
    """Raised when rate limit is exceeded and all retries are exhausted."""
    pass


class OpenFDANetworkError(OpenFDAError):
    """Raised when network connectivity fails and no fallback is available."""
    pass


# =====================================================================
# Configuration Dataclass
# =====================================================================

@dataclass
class OpenFDAConfig:
    """Configuration parameters for OpenFDAClient."""
    base_url: str = OPENFDA_EVENT_ENDPOINT
    api_key: Optional[str] = None
    batch_size: int = 100
    max_records: int = 10000
    rate_limit_delay: float = 1.2  # Seconds between sequential requests (unauthenticated: 40/min)
    max_retries: int = 5
    backoff_factor: float = 1.2
    timeout_seconds: Tuple[float, float] = (10.0, 30.0)  # (connect, read)
    cache_path: Optional[Union[str, Path]] = None
    cache_dir: Optional[Union[str, Path]] = None
    cache_ttl_hours: Optional[int] = 168  # 7 days default cache validity
    offline_fallback_path: Path = field(
        default_factory=lambda: Path("data/curated_glp1_psychiatric.json")
    )
    user_agent: str = "NeuroGLP-OpenFDAClient/1.0 (+https://github.com/neuroglp)"


# =====================================================================
# Custom Session Preserving Lucene '+' Operators
# =====================================================================

class OpenFDASession(requests.Session):
    """
    Custom requests.Session that preserves literal '+' operators in URLs.
    
    Standard requests.Session converts literal '+' inside query params to '%2B'.
    OpenFDA's Lucene query parser interprets '%2B' literally instead of as a boolean
    OR operator, causing queries to return 404 NOT_FOUND. This session ensures
    Lucene '+' operators are transmitted unescaped.
    """
    def prepare_request(self, request: requests.Request) -> requests.PreparedRequest:
        prepared = super().prepare_request(request)
        if prepared.url and "%2B" in prepared.url:
            prepared.url = prepared.url.replace("%2B", "+")
        return prepared


# =====================================================================
# Query Builder Functions
# =====================================================================

def build_openfda_query(
    drugs: Optional[List[str]] = None,
    brands: Optional[List[str]] = None,
    psychiatric_pts: Optional[List[str]] = None,
    include_extended_reactions: bool = False,
    generic_drugs: Optional[List[str]] = None,
    brand_drugs: Optional[List[str]] = None,
    reaction_pts: Optional[List[str]] = None,
) -> str:
    """
    Constructs an authoritative Lucene query string for OpenFDA drug adverse events.

    Combines:
    1. GLP-1 generic drug names via openfda.generic_name
    2. GLP-1 brand names via medicinalproduct (captures Wegovy and unmapped entries)
    3. Exact psychiatric MedDRA Preferred Terms via reactionmeddrapt.exact

    Parameters
    ----------
    drugs : Optional[List[str]]
        List of generic drug names (default: SEMAGLUTIDE, TIRZEPATIDE)
    brands : Optional[List[str]]
        List of brand names (default: OZEMPIC, WEGOVY, RYBELSUS, MOUNJARO, ZEPBOUND)
    psychiatric_pts : Optional[List[str]]
        List of exact MedDRA PT reactions
    include_extended_reactions : bool
        If True, adds suicide attempt and depressed mood
    generic_drugs : Optional[List[str]]
        Alias for drugs
    brand_drugs : Optional[List[str]]
        Alias for brands
    reaction_pts : Optional[List[str]]
        Alias for psychiatric_pts

    Returns
    -------
    str
        Formatted Lucene query string with balanced parentheses and +AND+ conjunction.
    """
    resolved_generics = drugs or generic_drugs or DEFAULT_GLP1_GENERICS
    resolved_brands = brands or brand_drugs or DEFAULT_GLP1_BRANDS
    resolved_pts = psychiatric_pts or reaction_pts or DEFAULT_PSYCHIATRIC_MEDDRA_PTS

    generics = [d.upper().strip() for d in resolved_generics]
    brand_list = [b.upper().strip() for b in resolved_brands]
    pts = list(resolved_pts)

    if include_extended_reactions:
        for ext in EXTENDED_PSYCHIATRIC_MEDDRA_PTS:
            if ext not in pts:
                pts.append(ext)
    pts = [p.upper().strip() for p in pts]

    # 1. Generic clause: patient.drug.openfda.generic_name:("SEMAGLUTIDE"+"TIRZEPATIDE")
    generic_terms = "+".join(f'"{d}"' for d in generics)
    generic_clause = f'patient.drug.openfda.generic_name:({generic_terms})'

    # 2. Brand clause: patient.drug.medicinalproduct:("OZEMPIC"+"WEGOVY"+...)
    brand_terms = "+".join(f'"{b}"' for b in brand_list)
    brand_clause = f'patient.drug.medicinalproduct:({brand_terms})'

    # Drug disjunction (union of mapped generic and free-text brand)
    drug_clause = f'({generic_clause}+{brand_clause})'

    # 3. Exact reaction clause: patient.reaction.reactionmeddrapt.exact:("SUICIDAL+IDEATION"+"DEPRESSION"+...)
    reaction_terms = "+".join(f'"{p.replace(" ", "+")}"' for p in pts)
    reaction_clause = f'patient.reaction.reactionmeddrapt.exact:({reaction_terms})'

    # Final boolean conjunction
    full_query = f'({drug_clause}+AND+{reaction_clause})'
    return full_query


# =====================================================================
# Transparent Dataset Loader
# =====================================================================

def load_curated_dataset(
    filepath: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Loads curated benchmark OpenFDA dataset. Transparently supports both
    uncompressed .json and .json.gz, as well as automatic fallback if .gz exists.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of OpenFDA adverse drug event record dictionaries.
    """
    if filepath is not None:
        candidate_paths = [Path(filepath)]
    else:
        pkg_data_dir = Path(__file__).resolve().parent
        candidate_paths = [
            pkg_data_dir / "curated_glp1_psychiatric.json.gz",
            pkg_data_dir / "curated_glp1_psychiatric.json",
            Path("data/curated_glp1_psychiatric.json.gz"),
            Path("data/curated_glp1_psychiatric.json"),
        ]

    resolved_path: Optional[Path] = None
    for cand in candidate_paths:
        if cand.exists():
            resolved_path = cand
            break
        gz_cand = cand.with_suffix(cand.suffix + ".gz")
        if gz_cand.exists():
            resolved_path = gz_cand
            break
        if cand.suffix == "" and cand.with_suffix(".json.gz").exists():
            resolved_path = cand.with_suffix(".json.gz")
            break
        if cand.suffix == "" and cand.with_suffix(".json").exists():
            resolved_path = cand.with_suffix(".json")
            break

    if resolved_path is None:
        target = filepath or "curated_glp1_psychiatric.json.gz"
        raise FileNotFoundError(f"Curated dataset not found at {target}")

    path = resolved_path
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data["results"]
    elif isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized dataset schema in {path}")


# =====================================================================
# OpenFDAClient Class
# =====================================================================

class OpenFDAClient:
    """
    Production-grade client for querying the public OpenFDA drug event API.

    Features:
    - Precise Lucene query builder for GLP-1 agonists and psychiatric MedDRA PTs
    - OpenFDASession preserving literal '+' operators across network requests
    - urllib3 Retry adapter combined with application-level retry loop on HTTP 429 & 5xx
    - Client-side polite rate-limiting (1.2-1.5s delay for unauthenticated traffic)
    - Pagination loop supporting up to 25,000 records
    - Gzipped / plain JSON local disk caching with data integrity checks
    - Offline fallback to bundled benchmark dataset
    """

    def __init__(
        self,
        config: Optional[OpenFDAConfig] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        rate_limit_delay: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_factor: Optional[float] = None,
        cache_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        timeout: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.config = config or OpenFDAConfig()
        if base_url is not None:
            self.config.base_url = base_url
        if api_key is not None:
            self.config.api_key = api_key
        if rate_limit_delay is not None:
            self.config.rate_limit_delay = rate_limit_delay
        if max_retries is not None:
            self.config.max_retries = max_retries
        if backoff_factor is not None:
            self.config.backoff_factor = backoff_factor
        if cache_path is not None:
            self.config.cache_path = Path(cache_path)
        if cache_dir is not None:
            self.config.cache_dir = Path(cache_dir)
        if timeout is not None:
            self.config.timeout_seconds = timeout

        self.base_url = self.config.base_url
        self._last_request_time: float = 0.0
        self._init_cache_dir()
        self.session = self._create_session()

    def _init_cache_dir(self) -> None:
        """Initializes and resolves the local cache directory."""
        if self.config.cache_path is not None:
            self.cache_path = Path(self.config.cache_path)
            self.cache_dir = self.cache_path.parent
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        elif self.config.cache_dir is not None:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path = None
        else:
            try:
                user_cache = Path.home() / ".cache" / "neuroglp"
                user_cache.mkdir(parents=True, exist_ok=True)
                self.cache_dir = user_cache
            except Exception:
                local_cache = Path("data") / "cache"
                local_cache.mkdir(parents=True, exist_ok=True)
                self.cache_dir = local_cache
            self.cache_path = None

        logger.debug(f"OpenFDAClient cache directory initialized at: {self.cache_dir}")

    def _create_session(self) -> OpenFDASession:
        """Constructs an HTTP session mounted with resilient urllib3 retry strategies."""
        session = OpenFDASession()
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "User-Agent": self.config.user_agent,
            "Accept": "application/json",
        })
        return session

    def close(self) -> None:
        """Closes the underlying HTTP session."""
        if self.session is not None:
            self.session.close()

    def __enter__(self) -> OpenFDAClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # -----------------------------------------------------------------
    # Query Builder Delegate
    # -----------------------------------------------------------------

    @staticmethod
    def build_query(
        generic_drugs: Optional[List[str]] = None,
        brand_drugs: Optional[List[str]] = None,
        reaction_pts: Optional[List[str]] = None,
        include_extended_reactions: bool = False,
        drugs: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        psychiatric_pts: Optional[List[str]] = None,
    ) -> str:
        """Static method delegate for query construction."""
        return build_openfda_query(
            drugs=drugs,
            brands=brands,
            psychiatric_pts=psychiatric_pts,
            include_extended_reactions=include_extended_reactions,
            generic_drugs=generic_drugs,
            brand_drugs=brand_drugs,
            reaction_pts=reaction_pts,
        )

    # -----------------------------------------------------------------
    # Rate Limiting & URL Assembly
    # -----------------------------------------------------------------

    def _enforce_rate_limit(self) -> None:
        """Enforces client-side rate limiting between sequential HTTP requests."""
        if self.config.rate_limit_delay <= 0.0:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        target_delay = self.config.rate_limit_delay if not self.config.api_key else 0.25

        if elapsed < target_delay:
            sleep_time = target_delay - elapsed
            logger.debug(f"Rate limiter sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def _prepare_url(self, query: str, limit: int, skip: int) -> str:
        """
        Assembles the request URL while preserving Lucene '+' operators.
        """
        query_params = {
            "limit": str(limit),
            "skip": str(skip),
        }
        if self.config.api_key:
            query_params["api_key"] = self.config.api_key

        param_str = urllib.parse.urlencode(query_params)
        url = f"{self.config.base_url}?search={query}&{param_str}"
        return url

    # -----------------------------------------------------------------
    # HTTP Request Execution (Single Page)
    # -----------------------------------------------------------------

    def fetch_adverse_events(
        self,
        search_query: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Fetches a single page of adverse drug events from OpenFDA.

        Handles:
        - Polite client rate limiting
        - Exponential backoff retry on HTTP 429 and 5xx errors
        - Graceful HTTP 404 index exhaustion (returns ([], 0))
        - Query syntax error detection (HTTP 400 raises OpenFDAQueryError)

        Parameters
        ----------
        search_query : Optional[str]
            Lucene search query (defaults to standard GLP-1 psychiatric query)
        limit : int, default 100
            Number of records to fetch per page (max 100 for unauthenticated)
        skip : int, default 0
            Number of matching records to skip

        Returns
        -------
        Tuple[List[Dict[str, Any]], int]
            (records_list, total_matching_records)
        """
        query = search_query or build_openfda_query()
        params: Dict[str, Any] = {
            "limit": limit,
            "skip": skip,
            "search": query,
        }
        if self.config.api_key:
            params["api_key"] = self.config.api_key

        max_attempts = max(1, self.config.max_retries)
        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            self._enforce_rate_limit()

            try:
                resp = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=self.config.timeout_seconds,
                )
            except requests.ConnectionError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    sleep_sec = (self.config.backoff_factor ** attempt) * 1.5
                    time.sleep(sleep_sec)
                    continue
                raise OpenFDANetworkError(f"Connection error: {e}") from e
            except requests.RequestException as e:
                last_error = e
                # Check if it contains an HTTPError with 404
                if isinstance(e, requests.HTTPError) and getattr(e, "response", None) is not None:
                    code = e.response.status_code
                    if code == 404:
                        return [], 0
                    if code == 429 and attempt < max_attempts - 1:
                        sleep_sec = (self.config.backoff_factor ** attempt) * 2.0
                        time.sleep(sleep_sec)
                        continue
                if attempt < max_attempts - 1:
                    sleep_sec = (self.config.backoff_factor ** attempt) * 1.5
                    time.sleep(sleep_sec)
                    continue
                raise OpenFDAError(f"Request failed: {e}") from e

            # Inspect status code
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except Exception as e:
                    raise OpenFDAError(f"Invalid JSON returned from OpenFDA: {e}") from e
                records = payload.get("results", [])
                total = payload.get("meta", {}).get("results", {}).get("total", len(records))
                return records, total

            if resp.status_code == 404:
                return [], 0

            if resp.status_code == 429:
                last_error = OpenFDARateLimitError("HTTP 429 Rate Limit Exceeded")
                if attempt < max_attempts - 1:
                    sleep_sec = (self.config.backoff_factor ** attempt) * 2.0
                    time.sleep(sleep_sec)
                    continue
                raise OpenFDARateLimitError("Exceeded OpenFDA rate limit after retries.")

            if resp.status_code in (500, 502, 503, 504):
                last_error = OpenFDAError(f"HTTP {resp.status_code} Server Error")
                if attempt < max_attempts - 1:
                    sleep_sec = (self.config.backoff_factor ** attempt) * 1.5
                    time.sleep(sleep_sec)
                    continue
                resp.raise_for_status()

            if resp.status_code == 400:
                raise OpenFDAQueryError(f"HTTP 400 Bad Request: {resp.text}")

            resp.raise_for_status()

        if last_error:
            raise last_error
        return [], 0

    # -----------------------------------------------------------------
    # Caching Mechanics
    # -----------------------------------------------------------------

    def _get_cache_path(self, query: str) -> Path:
        """Computes deterministic cache file path based on SHA-256 query hash."""
        if self.cache_path is not None:
            return self.cache_path
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        filename = f"openfda_events_{query_hash}.json.gz"
        return self.cache_dir / filename

    def is_cache_valid(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Validates whether a cache file exists, is non-empty, and valid JSON / Gzip."""
        target_path = Path(path) if path is not None else self._get_cache_path(build_openfda_query())
        if not target_path.exists():
            return False

        try:
            if target_path.stat().st_size == 0:
                return False

            if self.config.cache_ttl_hours is not None:
                mtime = target_path.stat().st_mtime
                age_hours = (time.time() - mtime) / 3600.0
                if age_hours > self.config.cache_ttl_hours:
                    return False

            if str(target_path).endswith(".gz"):
                with gzip.open(target_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            if isinstance(data, list) and len(data) > 0:
                return True
            if isinstance(data, dict) and "results" in data and len(data["results"]) > 0:
                return True
            return False
        except Exception as e:
            logger.debug(f"Cache validation failed for {target_path}: {e}")
            return False

    def save_cache(
        self,
        records: List[Dict[str, Any]],
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Atomically saves records to disk cache."""
        target_path = Path(path) if path is not None else self._get_cache_path(build_openfda_query())
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")

        try:
            if str(target_path).endswith(".gz"):
                with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                    json.dump(records, f)
            else:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(records, f)
            temp_path.replace(target_path)
            logger.info(f"Successfully cached {len(records)} records to {target_path}.")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            logger.warning(f"Failed to write cache to {target_path}: {e}")

    def load_cache(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """Loads records from disk cache file."""
        target_path = Path(path) if path is not None else self._get_cache_path(build_openfda_query())
        if not target_path.exists():
            raise FileNotFoundError(f"Cache file not found at {target_path}")

        if str(target_path).endswith(".gz"):
            with gzip.open(target_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "results" in data:
            return data["results"]
        raise OpenFDAError(f"Unexpected cache format in {target_path}")

    def _load_offline_fallback(self) -> List[Dict[str, Any]]:
        """Loads records from the bundled curated benchmark dataset."""
        fallback_path = self.config.offline_fallback_path
        return load_curated_dataset(fallback_path)

    # -----------------------------------------------------------------
    # Pagination & Cohort Ingestion
    # -----------------------------------------------------------------

    def fetch_cohort(
        self,
        target_count: int = 10000,
        batch_size: int = 100,
        search_query: Optional[str] = None,
        allow_cache_fallback: bool = True,
        use_cache: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Paginates through OpenFDA event endpoint to accumulate requested record volume.

        Parameters
        ----------
        target_count : int, default 10000
            Target number of adverse event records to ingest
        batch_size : int, default 100
            Batch size per request (max 100 for unauthenticated OpenFDA)
        search_query : Optional[str]
            Lucene query (defaults to standard GLP-1 psychiatric query)
        allow_cache_fallback : bool, default True
            If True, falls back to local cache or offline dataset on network failure
        use_cache : bool, default True
            If True, reads from and writes to disk cache
        progress_callback : Optional[Callable[[int, int], None]]
            Optional callback(records_fetched, target_count) for progress tracking

        Returns
        -------
        List[Dict[str, Any]]
            Accumulated list of adverse event records.
        """
        query = search_query or build_openfda_query()
        cache_target = self._get_cache_path(query)

        # 1. Check existing cache
        if use_cache and self.is_cache_valid(cache_target):
            cached = self.load_cache(cache_target)
            if len(cached) >= min(target_count, 5000):
                logger.info(f"Loaded {len(cached)} records from cache {cache_target}")
                if progress_callback:
                    progress_callback(len(cached), len(cached))
                return cached[:target_count]

        # 2. Live pagination loop
        all_records: List[Dict[str, Any]] = []
        batch_limit = min(batch_size, 100)
        skip = 0
        total_available = None

        try:
            while len(all_records) < target_count:
                if skip > 25000:
                    logger.warning("Reached OpenFDA maximum skip limit (25,000). Stopping pagination.")
                    break

                current_limit = min(batch_limit, target_count - len(all_records))
                records, total = self.fetch_adverse_events(
                    search_query=query,
                    limit=current_limit,
                    skip=skip,
                )

                if not records:
                    logger.info(f"Pagination completed: 0 records returned at skip={skip}.")
                    break

                if total_available is None:
                    total_available = total
                    target_count = min(target_count, total_available)

                all_records.extend(records)
                skip += len(records)

                if progress_callback:
                    progress_callback(len(all_records), target_count)

                if len(all_records) >= target_count:
                    break

                if total_available is not None and (len(all_records) >= total_available or skip >= total_available):
                    break

                if total_available is None and len(records) < current_limit:
                    logger.info("Received partial batch with unknown total; end of index reached.")
                    break

            if use_cache and all_records:
                self.save_cache(all_records, cache_target)

            return all_records

        except (OpenFDANetworkError, requests.RequestException) as e:
            logger.warning(f"Network ingestion failed: {e}")
            if allow_cache_fallback:
                if use_cache and self.is_cache_valid(cache_target):
                    return self.load_cache(cache_target)[:target_count]
                try:
                    return self._load_offline_fallback()[:target_count]
                except Exception as fallback_err:
                    logger.warning(f"Offline fallback also failed: {fallback_err}")
            raise

    def fetch_reports(
        self,
        query: Optional[str] = None,
        max_records: Optional[int] = None,
        use_cache: bool = True,
        offline: bool = False,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        allow_cache_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Primary interface matching PROJECT.md for adverse event report ingestion.

        Parameters
        ----------
        query : Optional[str]
            Custom Lucene query string (default: authoritative GLP-1 psych query)
        max_records : Optional[int]
            Maximum records to retrieve (default: from config, e.g. 10000)
        use_cache : bool, default True
            If True, reads from and writes to local disk cache
        offline : bool, default False
            If True, bypasses network entirely and loads from cache or curated dataset
        force_refresh : bool, default False
            If True, ignores existing cache and queries live API
        progress_callback : Optional[Callable[[int, int], None]]
            Optional callable(records_fetched, total_target) for progress UI
        allow_cache_fallback : bool, default True
            If True, falls back to cache/offline dataset on network error

        Returns
        -------
        List[Dict[str, Any]]
            Raw OpenFDA adverse event records.
        """
        active_query = query or build_openfda_query()
        target_limit = max_records or self.config.max_records
        cache_path = self._get_cache_path(active_query)

        if offline:
            logger.info("Offline mode requested. Bypassing network.")
            if use_cache and self.is_cache_valid(cache_path):
                return self.load_cache(cache_path)[:target_limit]
            return self._load_offline_fallback()[:target_limit]

        if force_refresh:
            use_cache_for_reading = False
        else:
            use_cache_for_reading = use_cache

        return self.fetch_cohort(
            target_count=target_limit,
            batch_size=self.config.batch_size,
            search_query=active_query,
            allow_cache_fallback=allow_cache_fallback,
            use_cache=use_cache_for_reading,
            progress_callback=progress_callback,
        )


# Convenience module-level wrapper
def fetch_openfda_reports(
    query: Optional[str] = None,
    max_records: int = 10000,
    use_cache: bool = True,
    offline: bool = False,
) -> List[Dict[str, Any]]:
    """Convenience functional wrapper to fetch reports using OpenFDAClient."""
    with OpenFDAClient() as client:
        return client.fetch_reports(
            query=query,
            max_records=max_records,
            use_cache=use_cache,
            offline=offline,
        )
