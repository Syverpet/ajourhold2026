"""Regional, non-overlapping 2025 RGB sampling guided by S2 change maps.

Requires Python >=3.10. Install: pip install numpy rasterio geopandas pyogrio shapely pyproj
Run this saved .py file. Counts are TOTALS for the whole region, not per S2 raster.
BATCH_NUMBER is the random seed and is included in all batch filenames.
Each batch has a separate folder and GeoPackage. Existing batches are never overwritten.
Norge i Bilder project footprints filter every eligible grid candidate.
Represent eligible MGRS tiles first, then balance counts and spread crop centres.
Internet is needed on cache misses. Failed coverage queries skip the affected source.
"""

import csv
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import sqlite3
import time
from math import isfinite
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from contextlib import contextmanager
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import pyogrio
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import ColorInterp, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely import make_valid, from_wkt
from shapely.geometry import Polygon, box, shape, mapping
from shapely.errors import GEOSException
from shapely.ops import unary_union
from shapely.prepared import prep

# ======================= USER SETTINGS =======================
BATCH_NUMBER = 1  # Change this number for the next batch.
TARGET_YEAR = 2025
S2_INPUT_FOLDER = Path(r"F:\Data\S2_kartverket\2025")
PREDICTION_FOLDER = Path(
    r"F:\Ajourhold 2026\ML_model_S2\predictions_national_epsg25833"
)
OUTPUT_FOLDER = Path(r"F:/Ajourhold 2026/test_tiles_256x256")
# True also prevents overlap with earlier sampler TIFFs under OUTPUT_FOLDER.
# Reproduction then requires the same previous tiles, inputs and settings.
AVOID_PREVIOUS_BATCHES = True
S2_FILE_PATTERN = "*.tif*"
PREDICTION_FILE_PATTERN = "*change_class*.tif*"

# Requested totals for THIS BATCH across ALL matched S2 rasters.
N_URBAN_TOTAL = 30
N_CLEARCUT_TOTAL = 30
N_RANDOM_TOTAL = 30

TILE_SIZE = 256
# Input [B2, B3, B4, B8] -> output [red, green, blue].
# Change to (1, 2, 3) if the source is already RGB.
RGB_BAND_INDEXES = (3, 2, 1)
PREDICTION_BAND = 1
URBAN_CLASS_VALUE = 2
CLEARCUT_CLASS_VALUE = 1
PREDICTION_EXTRA_NODATA_VALUES = []
MIN_URBAN_PERCENT = 1.0
MIN_CLEARCUT_PERCENT = 5.0
MIN_VALID_PREDICTION_PERCENT = 95.0
MIN_VALID_S2_PERCENT = 99.0
EXCLUDE_ALL_ZERO_S2_PIXELS = True

# AR5 path/layer used in your previous training scripts.
AR5_PATH = Path(
    r"F:\Data\AR5\Basisdata_0000_Norge_25833_FKB-AR5_FGDB\Basisdata_0000_Norge_25833_FKB-AR5_FGDB.gdb"
)
AR5_LAYER = "fkb_ar5_omrade"
AR5_FIELD = "arealtype"
AR5_WATER_CODES = (81, 82)  # Freshwater, sea
AR5_KNOWN_CODES = (11, 12, 21, 22, 23, 30, 50, 60, 70, 81, 82)
MAX_WATER_PERCENT = 1.0
# Missing AR5 coverage and code 99 are unknown, not confirmed dry land.
MIN_KNOWN_AR5_PERCENT = 95.0

# Require the WHOLE crop to be covered by the combined municipality polygons.
# Fastlands-Norge includes coastal islands. AR5 independently filters water.
NORWAY_BOUNDARY_PATH = Path(
    r"F:\Data\Administrative grenser\Norge_25833_Kommuner_FGDB\Basisdata_0000_Norge_25833_Kommuner_FGDB.gdb"
)
NORWAY_BOUNDARY_LAYER = "kommune"
MUNICIPALITY_ID_FIELD = "kommunenummer"  # Matched case-insensitively
EXCLUDED_MUNICIPALITY_PREFIXES = ("21", "22")  # Svalbard and Jan Mayen

# Norge i Bilder: score each candidate's ENTIRE footprint, not its MGRS average.
YEARS = tuple(range(2018, 2026))
MIN_COVERAGE_PERCENT = 90.0
MIN_SCORE = 40.0
MIN_AVAILABLE_YEARS = 2
RECENT_YEARS = (2024, 2025)
REQUIRE_RECENT_IMAGE = True
NO_RECENT_IMAGE_FACTOR = 0.4
YEAR_COUNT_WEIGHT = 40.0
TEMPORAL_WEIGHT = 35.0
START_WEIGHT = 12.5
END_WEIGHT = 12.5
TEMPORAL_DISTANCE_LIMIT = 3.0
NIB_PROJECTS_URL = "https://backend-api.klienter-prod-k8s2.norgeibilder.no/projects"
NIB_CACHE_FOLDER = OUTPUT_FOLDER / "orthophoto_cache"
NIB_CACHE_MAX_AGE_DAYS = 30
NIB_REFRESH_CACHE = False  # True fetches fresh project coverage.
NIB_TIMEOUT_SECONDS = 90
NIB_REQUEST_ATTEMPTS = 6
NIB_MIN_REQUEST_INTERVAL_SECONDS = 5.0  # Shared by ALL sources and retries.
NIB_RETRY_BASE_SECONDS = 60.0  # 60, 120, 240, ... after throttling/errors.
NIB_RETRY_MAX_SECONDS = 900.0  # Retry-After may require a longer wait.
NIB_SATELLITE_TYPES = (6,)  # Type 6 = satellittbilde in Norge i Bilder.
# API errors or incomplete metadata skip the whole source, never imply 0%.
# Orthophoto score is an eligibility threshold, NOT a spatial ranking weight.
# First represent each eligible MGRS, then favour sources with fewer accepted crops.
# Totals/category limits still apply; impossible representation is reported.
MIN_SAMPLE_CENTER_DISTANCE_METRES = 5000.0  # All categories, sources and prior batches.
# Set 0 to disable the hard distance rule; farthest-first spreading still applies.

# Randomly offset candidate grid per S2 raster. Smaller = more placements.
# Candidate windows may overlap; exported windows cannot overlap.
CANDIDATE_STRIDE = 128
CHANGE_WEIGHT_POWER = 2.0
MAX_ATTEMPTS_PER_REQUESTED_TILE = 200
# Preparation runs in parallel. Final regional selection is coordinated centrally.
PREP_WORKERS = min(8, max(1, (os.cpu_count() or 4) // 4))
GDAL_CACHE_MB_PER_WORKER = 256
WARP_MEMORY_MB_PER_WORKER = 128
COMPRESSION_THREADS = 2
# Common metric CRS for AR5 area and cross-raster overlap checks.
REGION_CRS = "EPSG:25833"
# Conservative margin for transformed edges between different S2 sources.
CROSS_SOURCE_MARGIN_METRES = 0.05
# =============================================================

CATEGORIES = ("urban_change", "clearcut_change", "random")
PRED_NODATA = np.iinfo(np.int32).min
MGRS_PATTERN = re.compile(
    r"(?<![A-Z0-9])T?(\d{2}[C-HJ-NP-X][A-HJ-NP-Z]{2})(?![A-Z0-9])", re.I
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def score_orthophoto_coverage(
    coverage_by_year: dict[int, float],
    *,
    min_coverage_percent: float = MIN_COVERAGE_PERCENT,
    min_score: float = MIN_SCORE,
) -> dict:
    """User's scoring rules; missing coverage is an error, not confirmed absence."""
    if not 0 < min_coverage_percent <= 100:
        raise ValueError("min_coverage_percent må være > 0 og <= 100.")
    if not 0 <= min_score <= 100:
        raise ValueError("min_score må være mellom 0 og 100.")
    missing_years = [year for year in YEARS if year not in coverage_by_year]
    if missing_years:
        raise ValueError(
            f"Mangler dekningsverdier for: {missing_years}. "
            "Bruk 0 bare når manglende dekning er bekreftet."
        )
    coverage = {}
    for year in YEARS:
        try:
            value = float(coverage_by_year[year])
        except (TypeError, ValueError):
            raise ValueError(f"Ugyldig dekningsverdi for {year}.") from None
        if not isfinite(value) or not 0 <= value <= 100:
            raise ValueError(
                f"Dekningen for {year} må være mellom 0 og 100, fikk {value}."
            )
        coverage[year] = value
    available_years = [y for y in YEARS if coverage[y] >= min_coverage_percent]
    available_set = set(available_years)
    has_recent_image = bool(available_set.intersection(RECENT_YEARS))
    year_count_points = YEAR_COUNT_WEIGHT * len(available_years) / len(YEARS)
    temporal_points = 0.0
    if available_years:
        contributions = [
            max(
                0.0,
                1.0
                - min(abs(y - a) for a in available_years) / TEMPORAL_DISTANCE_LIMIT,
            )
            for y in YEARS
        ]
        temporal_points = TEMPORAL_WEIGHT * sum(contributions) / len(YEARS)
    first_year, last_year = YEARS[0], YEARS[-1]
    start_points = (
        START_WEIGHT
        if first_year in available_set
        else START_WEIGHT / 2
        if first_year + 1 in available_set
        else 0.0
    )
    end_points = (
        END_WEIGHT
        if last_year in available_set
        else END_WEIGHT / 2
        if last_year - 1 in available_set
        else 0.0
    )
    base_score = year_count_points + temporal_points + start_points + end_points
    recency_factor = (
        1.0 if has_recent_image else NO_RECENT_IMAGE_FACTOR if available_years else 0.0
    )
    final_score = base_score * recency_factor
    rejection_reasons = []
    if len(available_years) < MIN_AVAILABLE_YEARS:
        rejection_reasons.append(
            f"Færre enn {MIN_AVAILABLE_YEARS} kvalifiserende bildeår."
        )
    if REQUIRE_RECENT_IMAGE and not has_recent_image:
        rejection_reasons.append(f"Mangler kvalifiserende ortofoto fra {RECENT_YEARS}.")
    if final_score < min_score:
        rejection_reasons.append(
            f"Score {final_score:.2f} er under terskelen {min_score:.2f}."
        )
    return dict(
        score=round(final_score, 2),
        base_score=round(base_score, 2),
        recency_factor=recency_factor,
        accepted=not rejection_reasons,
        rejection_reasons=rejection_reasons,
        available_years=available_years,
        number_of_years=len(available_years),
        has_recent_image=has_recent_image,
        coverage_by_year=coverage,
        components=dict(
            year_count=round(year_count_points, 2),
            temporal_distribution=round(temporal_points, 2),
            start=round(start_points, 2),
            end=round(end_points, 2),
        ),
    )


def orthophoto_year_union(response, year, query_polygon):
    """Validate a complete projects response; union aerial footprints in EPSG:25833."""
    if not isinstance(response, dict) or not isinstance(response.get("projects"), list):
        raise ValueError("Norge i Bilder response has no valid projects list.")
    geometries, project_ids = [], []
    for project in response["projects"]:
        metadata = project.get("metadata") if isinstance(project, dict) else None
        if not isinstance(metadata, dict):
            raise ValueError("Missing Norge i Bilder project metadata.")
        if metadata.get("year") != year:
            raise ValueError(
                f"Norge i Bilder returned a project outside requested year {year}."
            )
        kind = metadata.get("orthophotoType")
        if type(kind) is not int or kind not in range(1, 13):
            raise ValueError(
                f"Unknown orthophotoType: {kind!r}; cannot confirm aerial coverage."
            )
        if kind in NIB_SATELLITE_TYPES:
            continue
        # Extra defence against incorrectly categorized satellite projects.
        description = " ".join(
            str(v or "")
            for v in (
                project.get("projectName"),
                metadata.get("aircraftCompany"),
                metadata.get("dekningsnummer"),
            )
        ).lower()
        if any(
            word in description
            for word in ("sentinel", "satellitt", "satellite", "landsat")
        ):
            continue
        raw_geometry = project.get("geometry")
        if not isinstance(raw_geometry, dict) or raw_geometry.get("type") not in (
            "Polygon",
            "MultiPolygon",
        ):
            raise ValueError(f"Missing polygon geometry: {project.get('projectName')}")
        geometry = make_valid(shape(raw_geometry))
        if geometry.is_empty or not isfinite(geometry.area) or geometry.area <= 0:
            raise ValueError(f"Empty project footprint: {project.get('projectName')}")
        geometries.append(geometry.intersection(query_polygon))
        project_ids.append(project.get("projectId"))
    return unary_union(geometries), project_ids


def retry_after_seconds(value):
    """Retry-After can contain seconds or an HTTP date; invalid values use backoff."""
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, seconds) if isfinite(seconds) else 0.0


class OrthophotoCoverage:
    """One all-years query per source; split by year locally and cache across batches.

    API reference: https://backend-api.klienter-prod-k8s2.norgeibilder.no/swagger/v1/swagger.json
    Only complete, validated responses are cached. Cached project footprints are
    coverage metadata, not downloaded photographs or an image-quality assessment.
    """

    def __init__(self):
        self.audit = []
        self.next_request_at = 0.0

    def wait_for_request(self):
        while True:
            remaining = self.next_request_at - time.monotonic()
            if remaining <= 0:
                return
            if remaining >= 10:
                logging.info(
                    "Norge i Bilder cooldown: %.0f seconds remaining", remaining
                )
            time.sleep(min(remaining, 30.0))

    def fetch_years(self, query_polygon):
        # No year filter: one response includes all project years. Only YEARS are scored.
        payload = dict(
            geometry=mapping(query_polygon),
            inputWkid=25833,
            outputWkid=25833,
            returnMetadata=True,
            returnGeometry=True,
            stopOnCover=False,
        )
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        key = hashlib.sha256(NIB_PROJECTS_URL.encode() + body).hexdigest()
        path = NIB_CACHE_FOLDER / f"{key}.json"

        def split_years(response, fetched_at, cached):
            if not isinstance(response, dict) or not isinstance(
                response.get("projects"), list
            ):
                raise ValueError("Norge i Bilder response has no valid projects list.")
            groups = {year: [] for year in YEARS}
            for project in response["projects"]:
                metadata = (
                    project.get("metadata") if isinstance(project, dict) else None
                )
                year = metadata.get("year") if isinstance(metadata, dict) else None
                if type(year) is not int:
                    raise ValueError(
                        "Missing or invalid project year; annual coverage is unknown."
                    )
                if year in groups:
                    groups[year].append(project)
            result = {}
            for year, projects in groups.items():
                union, ids = orthophoto_year_union(
                    {"projects": projects}, year, query_polygon
                )
                result[year] = (
                    union,
                    dict(
                        year=year,
                        cache_key=key,
                        project_ids=ids,
                        fetched_at=fetched_at,
                        cached=cached,
                    ),
                )
            return result

        if not NIB_REFRESH_CACHE and path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                age = time.time() - cached["fetched_at_epoch"]
                if 0 <= age <= NIB_CACHE_MAX_AGE_DAYS * 86400:
                    return split_years(cached["response"], cached["fetched_at"], True)
            except (OSError, ValueError, KeyError, TypeError, GEOSException):
                logging.warning("Ignoring invalid orthophoto cache: %s", path)
        request = Request(
            NIB_PROJECTS_URL,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "S2TileSampler/1.0",
            },
        )
        for attempt in range(NIB_REQUEST_ATTEMPTS):
            self.wait_for_request()
            self.next_request_at = time.monotonic() + NIB_MIN_REQUEST_INTERVAL_SECONDS
            try:
                with urlopen(request, timeout=NIB_TIMEOUT_SECONDS) as handle:
                    response = json.load(handle)
                self.next_request_at = (
                    time.monotonic() + NIB_MIN_REQUEST_INTERVAL_SECONDS
                )
                break
            except (URLError, TimeoutError, OSError) as exc:
                if isinstance(exc, HTTPError) and exc.code not in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise
                server_wait = (
                    retry_after_seconds(exc.headers.get("Retry-After"))
                    if isinstance(exc, HTTPError) and exc.headers
                    else 0.0
                )
                delay = max(
                    NIB_MIN_REQUEST_INTERVAL_SECONDS,
                    server_wait,
                    min(NIB_RETRY_BASE_SECONDS * 2**attempt, NIB_RETRY_MAX_SECONDS),
                )
                # Keep the cooldown even if this source exhausts its attempts.
                self.next_request_at = time.monotonic() + delay
                if attempt + 1 == NIB_REQUEST_ATTEMPTS:
                    raise
                logging.warning(
                    "Norge i Bilder: %s; waiting %.0f seconds before retry %d/%d",
                    exc,
                    delay,
                    attempt + 2,
                    NIB_REQUEST_ATTEMPTS,
                )
        fetched_at = datetime.now(timezone.utc).isoformat()
        result = split_years(response, fetched_at, False)
        cached = dict(
            fetched_at=fetched_at, fetched_at_epoch=time.time(), response=response
        )
        NIB_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial.json")
        temporary.write_text(json.dumps(cached), encoding="utf-8")
        temporary.replace(path)
        return result

    def rank_source(self, state, mainland, occupied):
        """Filter all grid candidates by coverage; leave seeded tie order intact."""
        requested = (N_URBAN_TOTAL, N_CLEARCUT_TOTAL, N_RANDOM_TOTAL)
        pools = [order for order, n in zip(state["orders"], requested) if n]
        ids = np.unique(np.concatenate(pools))
        polygons, candidates = [], []
        with rasterio.open(state["source"]) as s2:
            transformer = Transformer.from_crs(s2.crs, REGION_CRS, always_xy=True)
            for candidate in ids:
                rr, cc = divmod(int(candidate), len(state["cols"]))
                window = Window(
                    int(state["cols"][cc]), int(state["rows"][rr]), TILE_SIZE, TILE_SIZE
                )
                polygon = footprint(s2, window, REGION_CRS, transformer=transformer)
                if (
                    mainland.covers(polygon)
                    and not occupied.overlaps(polygon, state["source"])
                    and not occupied.too_close(polygon)
                ):
                    candidates.append(int(candidate))
                    polygons.append(polygon)
        size = len(state["rows"]) * len(state["cols"])
        state["orthophoto_coverage"] = np.full(
            (size, len(YEARS)), np.nan, dtype="float64"
        )
        state["orthophoto_scores"] = np.full(size, -np.inf)
        state["candidate_centers"] = np.full((size, 2), np.nan)
        state["nearest_selected_squared"] = np.full(size, np.inf)
        for candidate, polygon in zip(candidates, polygons):
            state["candidate_centers"][candidate] = polygon.centroid.coords[0]
        audit = dict(
            source=state["source"], mgrs=mgrs_code(Path(state["source"])), years=[]
        )
        self.audit.append(audit)
        if not polygons:
            state["orders"] = [np.array([], dtype=int) for _ in CATEGORIES]
            audit.update(
                status="no_candidates_after_land_overlap_spacing_filters",
                accepted_candidates=0,
            )
            return
        # Rounded metric envelope can reuse queries when candidate extents match.
        bounds = np.array([g.bounds for g in polygons])
        low = np.floor(bounds[:, :2].min(axis=0) / 1000) * 1000
        high = np.ceil(bounds[:, 2:].max(axis=0) / 1000) * 1000
        query = box(*low, *high)
        try:
            logging.info(
                "ORTHOPHOTO %s | all years in one request (or cache) | candidates=%d",
                audit["mgrs"],
                len(candidates),
            )
            annual = self.fetch_years(query)
            for j, year in enumerate(YEARS):
                union, evidence = annual[year]
                audit["years"].append(evidence)
                prepared = prep(union)
                for candidate, polygon in zip(candidates, polygons):
                    if prepared.covers(polygon):
                        coverage = 100.0
                    elif prepared.disjoint(polygon):
                        coverage = 0.0
                    else:
                        coverage = min(
                            100.0,
                            max(
                                0.0,
                                100 * union.intersection(polygon).area / polygon.area,
                            ),
                        )
                    state["orthophoto_coverage"][candidate, j] = coverage
        except (OSError, ValueError, TypeError, KeyError, GEOSException) as exc:
            logging.error(
                "SKIP %s | orthophoto coverage could not be confirmed: %s",
                audit["mgrs"],
                exc,
            )
            state["orders"] = [np.array([], dtype=int) for _ in CATEGORIES]
            audit.update(status="query_failed", error=str(exc), accepted_candidates=0)
            return
        for candidate in candidates:
            result = score_orthophoto_coverage(
                dict(zip(YEARS, state["orthophoto_coverage"][candidate])),
                min_coverage_percent=MIN_COVERAGE_PERCENT,
                min_score=MIN_SCORE,
            )
            if result["accepted"]:
                state["orthophoto_scores"][candidate] = result["score"]
        for c, order in enumerate(state["orders"]):
            eligible = (
                order[np.isfinite(state["orthophoto_scores"][order])]
                if requested[c]
                else order[:0]
            )
            state["orders"][c] = eligible
        accepted = int(np.isfinite(state["orthophoto_scores"]).sum())
        audit.update(
            status="scored",
            accepted_candidates=accepted,
            evaluated_candidates=len(candidates),
        )
        logging.info(
            "ORTHOPHOTO %s | approved=%d/%d | category candidates=%s",
            audit["mgrs"],
            accepted,
            len(candidates),
            [len(o) for o in state["orders"]],
        )


def mgrs_code(path):
    codes = {"T" + c.upper() for c in MGRS_PATTERN.findall(path.stem)}
    if len(codes) != 1:
        raise ValueError(f"Expected one MGRS code: {path.name}")
    return codes.pop()


def rasters(folder, pattern):
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        p
        for p in folder.rglob(pattern)
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and not p.resolve().is_relative_to(OUTPUT_FOLDER.resolve())
    )


def matched_jobs():
    sources = rasters(S2_INPUT_FOLDER, S2_FILE_PATTERN)
    if not sources:
        raise FileNotFoundError(f"No S2 rasters in {S2_INPUT_FOLDER}")
    codes = {}
    for path in sources:
        years = set(re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", path.stem))
        if years and years != {str(TARGET_YEAR)}:
            raise ValueError(f"Wrong year in S2 filename: {path.name}")
        codes[path] = mgrs_code(path)
    wanted = set(codes.values())
    if len(wanted) != len(codes):
        raise ValueError(
            "Use exactly one 2025 input mosaic per MGRS code to balance source representation."
        )
    predictions = {}
    for path in rasters(PREDICTION_FOLDER, PREDICTION_FILE_PATTERN):
        try:
            code = mgrs_code(path)
        except ValueError:
            continue
        if code not in wanted:
            continue
        if code in predictions:
            raise ValueError(
                f"Ambiguous prediction for {code}: {predictions[code]} and {path}"
            )
        predictions[code] = path
    jobs = []
    for path, code in codes.items():
        if code not in predictions:
            logging.warning("SKIP %s | no matching change_class prediction", path.name)
        else:
            jobs.append(
                (
                    str(path),
                    str(predictions[code]),
                    BATCH_NUMBER,
                    path.relative_to(S2_INPUT_FOLDER).as_posix(),
                )
            )
    logging.info("Matched %d/%d S2 rasters", len(jobs), len(sources))
    return jobs


def prediction_vrt(pred, s2):
    if pred.crs is None or not 1 <= PREDICTION_BAND <= pred.count:
        raise ValueError(f"Missing prediction CRS or invalid band: {pred.name}")
    if not np.issubdtype(np.dtype(pred.dtypes[PREDICTION_BAND - 1]), np.integer):
        raise ValueError(f"Expected integer change_class raster: {pred.name}")
    return WarpedVRT(
        pred,
        crs=s2.crs,
        transform=s2.transform,
        width=s2.width,
        height=s2.height,
        dtype="int32",
        resampling=Resampling.nearest,
        nodata=int(PRED_NODATA),
        warp_mem_limit=WARP_MEMORY_MB_PER_WORKER,
    )


def prediction_values(vrt, window=None):
    values = vrt.read(PREDICTION_BAND, window=window, masked=True).filled(PRED_NODATA)
    for value in PREDICTION_EXTRA_NODATA_VALUES:
        values[values == value] = PRED_NODATA
    return values


def coverage_grid(mask, rows, cols):
    dtype = np.uint32 if mask.size < 2**32 else np.uint64
    table = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=dtype)
    np.cumsum(mask, axis=0, dtype=dtype, out=table[1:, 1:])
    np.cumsum(table[1:, 1:], axis=1, dtype=dtype, out=table[1:, 1:])
    top, left = rows[:, None], cols[None, :]
    bottom, right = top + TILE_SIZE, left + TILE_SIZE
    result = table[bottom, right].astype(np.int64)
    result -= table[top, right].astype(np.int64)
    result -= table[bottom, left].astype(np.int64)
    result += table[top, left].astype(np.int64)
    return result.ravel().astype(np.float32) * (100.0 / TILE_SIZE**2)


def prepare_source(job):
    source, prediction, seed, source_key = job
    digest = hashlib.blake2b(source_key.encode(), digest_size=8).hexdigest()
    rng = np.random.default_rng([seed, int(digest, 16)])
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_MB_PER_WORKER * 1024**2):
        with rasterio.open(source) as s2:
            if s2.crs is None or min(s2.shape) < TILE_SIZE:
                raise ValueError(f"Missing CRS or image too small: {source}")
            if max(RGB_BAND_INDEXES) > s2.count:
                raise ValueError(f"RGB bands not available: {source}")

            def starts(length):
                last = length - TILE_SIZE
                offset = int(rng.integers(CANDIDATE_STRIDE))
                return np.unique(
                    np.r_[0, np.arange(offset, last + 1, CANDIDATE_STRIDE), last]
                )

            rows, cols = starts(s2.height), starts(s2.width)
            with rasterio.open(prediction) as pred, prediction_vrt(pred, s2) as vrt:
                labels = prediction_values(vrt)
            coverage = coverage_grid(labels != PRED_NODATA, rows, cols)
            scores = [
                coverage_grid(labels == value, rows, cols)
                for value in (URBAN_CLASS_VALUE, CLEARCUT_CLASS_VALUE)
            ]
            orders = []
            for score, threshold in zip(
                scores, (MIN_URBAN_PERCENT, MIN_CLEARCUT_PERCENT)
            ):
                ids = np.flatnonzero(
                    (score >= threshold) & (coverage >= MIN_VALID_PREDICTION_PERCENT)
                )
                weights = (score[ids].astype(float) / 100) ** CHANGE_WEIGHT_POWER
                keys = (
                    -np.log(np.maximum(rng.random(len(ids)), np.finfo(float).tiny))
                    / weights
                )
                orders.append(ids[np.argsort(keys)])
            orders.append(rng.permutation(len(coverage)))
    logging.info(
        "Prepared %s | candidate counts %s", Path(source).name, [len(a) for a in orders]
    )
    return dict(
        source=source,
        prediction=prediction,
        rows=rows,
        cols=cols,
        orders=orders,
        used=set(),
        rejected=set(),
    )


def footprint(s2, window, target_crs, transformer=None):
    """Densify crop edges before transforming between UTM zones."""
    r, c, n = window.row_off, window.col_off, TILE_SIZE
    points = []
    for t in np.linspace(0, n, 65, endpoint=False):
        points.append((c + t, r))
    for t in np.linspace(0, n, 65, endpoint=False):
        points.append((c + n, r + t))
    for t in np.linspace(0, n, 65, endpoint=False):
        points.append((c + n - t, r + n))
    for t in np.linspace(0, n, 65, endpoint=False):
        points.append((c, r + n - t))
    x, y = zip(*(s2.transform * p for p in points))
    if transformer is None:
        transformer = Transformer.from_crs(s2.crs, target_crs, always_xy=True)
    x, y = transformer.transform(x, y)
    polygon = Polygon(zip(x, y))
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"Invalid transformed footprint: {s2.name}")
    return polygon


class OverlapIndex:
    """Spatial hash shared by all categories and source rasters."""

    def __init__(self):
        self.cells = defaultdict(list)
        self.items = []

    def keys(self, geometry):
        x0, y0, x1, y1 = geometry.bounds
        for x in range(int(np.floor(x0 / 5000)), int(np.floor(x1 / 5000)) + 1):
            for y in range(int(np.floor(y0 / 5000)), int(np.floor(y1 / 5000)) + 1):
                yield x, y

    def overlaps(self, geometry, source):
        padded = geometry.buffer(CROSS_SOURCE_MARGIN_METRES)
        ids = {i for key in self.keys(padded) for i in self.cells.get(key, [])}
        for i in ids:
            other, other_source = self.items[i]
            test = geometry if source == other_source else padded
            if test.intersects(other) and test.intersection(other).area > 0:
                return True
        return False

    def add(self, geometry, source):
        index = len(self.items)
        self.items.append((geometry, source))
        for key in self.keys(geometry):
            self.cells[key].append(index)

    def too_close(self, geometry):
        distance = MIN_SAMPLE_CENTER_DISTANCE_METRES
        if distance <= 0:
            return False
        center = geometry.centroid
        query = box(
            center.x - distance,
            center.y - distance,
            center.x + distance,
            center.y + distance,
        )
        ids = {i for key in self.keys(query) for i in self.cells.get(key, [])}
        return any(center.distance(self.items[i][0].centroid) < distance for i in ids)


# Review fields are initialized only for NEW rows. Existing rows are never rewritten.
REVIEW_SCHEMA = {
    "tile_id": "object",
    "batch_number": "Int64",
    "raster_relpath": "object",
    "filename": "object",
    "mgrs": "object",
    "category": "object",
    "source_s2": "object",
    "source_prediction": "object",
    "year": "Int32",
    "row": "Int32",
    "col": "Int32",
    "urban_percent": "float64",
    "clearcut_percent": "float64",
    "valid_s2_percent": "float64",
    "valid_prediction_percent": "float64",
    "water_percent": "float64",
    "known_ar5_percent": "float64",
    "orthophoto_score": "float64",
    "orthophoto_base_score": "float64",
    "orthophoto_years": "object",
    "orthophoto_number_of_years": "Int32",
    "orthophoto_has_recent": "Int32",
    "orthophoto_coverage_by_year": "object",
    "created_at": "object",
    "review_status": "object",
    "annotator": "object",
    "reviewer": "object",
    "reviewed_at": "object",
    "split": "object",
    "notes": "object",
}
POLYGON_SCHEMA = {
    "tile_id": "object",
    "class_code": "Int32",
    "change_year": "Int32",
    "confidence": "object",
    "annotator": "object",
    "review_status": "object",
    "notes": "object",
}


@contextmanager
def sampler_lock():
    """OS lock prevents two sampler processes appending to this project at once."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_FOLDER / "sampler.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another sampler is using this output folder.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class BatchDatabase:
    """One new GeoPackage per batch; no previous database is opened for writing."""

    def __init__(self, folder, occupied):
        self.root = folder.resolve()
        self.path = self.root / f"outlines_batch_{BATCH_NUMBER}.gpkg"
        self.occupied = occupied
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite {self.path}")
        for name, schema, geometry_type in (
            ("tile_review", REVIEW_SCHEMA, "Polygon"),
            ("change_polygons", POLYGON_SCHEMA, "MultiPolygon"),
        ):
            empty = gpd.GeoDataFrame(
                {k: pd.Series(dtype=v) for k, v in schema.items()},
                geometry=gpd.GeoSeries([], crs=REGION_CRS),
            )
            pyogrio.write_dataframe(
                empty, self.path, layer=name, driver="GPKG", geometry_type=geometry_type
            )
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE UNIQUE INDEX tile_id_unique ON tile_review(tile_id)")
        guide = self.root / f"annotation_guide_batch_{BATCH_NUMBER}.txt"
        with guide.open("x", encoding="utf-8") as handle:
            handle.write(f"""BATCH {BATCH_NUMBER} / RANDOM SEED {BATCH_NUMBER}
Open {self.path.name} in your GIS.
tile_review contains the tile outlines and review attributes.
change_polygons is an empty layer for your manual annotations.
Copy the containing outline's tile_id into each annotation polygon's tile_id.
Polygon class_code: 1=clear-cut, 2=urban change, 255=ignore/uncertain.
Set change_year when known; confidence: high/medium/low.
Polygon review_status: draft/checked.
Tile review_status: not_started/in_progress/complete/checked.
Only mark a tile complete after inspecting its whole area.
Edit annotator, reviewer, reviewed_at, split and notes as needed.
Split: unassigned/train/validation/test; no automatic assignment.
Keep outline geometry and tile_id unchanged. Raster paths are relative to this folder.
New batches do not alter this GeoPackage or its TIFFs.
Reusing this batch number stops before writing; choose a new number for more tiles.
Sampling category is not a ground-truth label. No masks are generated here.
orthophoto_score is an eligibility threshold, not a ranking weight.
orthophoto_years lists years meeting the minimum full-tile coverage threshold.
orthophoto_coverage_by_year is JSON with annual footprint coverage percentages.
Overlapping projects from one year count once; satellite imagery is excluded.
Coverage describes project footprints, not sharpness, clouds or visible change.
The random category has no change-class threshold and uses spatial spreading.
Eligible MGRS sources are represented first; remaining counts favour less sampled sources.
Crop centres are selected farthest-first within each MGRS across all categories.
Minimum centre spacing: {MIN_SAMPLE_CENTER_DISTANCE_METRES} metres, also across sources.
Previous batches are included in the spacing check when AVOID_PREVIOUS_BATCHES is True.
Seeded sampling order breaks distance ties; orthophoto score does not favour small hotspots.
See mgrs_summary_batch_{BATCH_NUMBER}.csv for counts and unrepresented sources.
Reproduction requires unchanged inputs, settings, earlier tiles and API/cache data.
""")

    def append_tile(self, record, raster_path, polygon):
        values = {k: record.get(k) for k in REVIEW_SCHEMA}
        values.update(
            raster_relpath=raster_path.name,
            review_status="not_started",
            annotator="",
            reviewer="",
            reviewed_at="",
            split="unassigned",
            notes="",
        )
        frame = gpd.GeoDataFrame([values], geometry=[polygon], crs=REGION_CRS)
        for field, dtype in REVIEW_SCHEMA.items():
            if dtype != "object":
                frame[field] = pd.to_numeric(frame[field], errors="coerce").astype(
                    dtype
                )
        pyogrio.write_dataframe(
            frame, self.path, layer="tile_review", driver="GPKG", append=True
        )
        self.occupied.add(polygon, record["source_s2"])


def previous_footprints():
    """Read existing sampler TIFFs, including outputs from earlier script versions."""
    occupied = OverlapIndex()
    if not AVOID_PREVIOUS_BATCHES:
        return occupied
    for path in sorted(OUTPUT_FOLDER.rglob("*")):
        if path.suffix.lower() not in {".tif", ".tiff"} or ".partial." in path.name:
            continue
        if not re.search(
            r"_(urban_change|clearcut_change|random)_\d+_r\d+_c\d+", path.stem
        ):
            continue
        with rasterio.open(path) as src:
            if src.shape != (256, 256) or src.count != 3 or src.crs is None:
                raise ValueError(f"Invalid existing sampler tile: {path}")
            polygon = footprint(src, Window(0, 0, 256, 256), REGION_CRS)
            occupied.add(polygon, src.tags(ns="SAMPLING").get("source_s2", ""))
    logging.info("Avoiding %d existing tile footprints", len(occupied.items))
    return occupied


class MainlandFilter:
    """Prepared union permits crossing internal municipality boundaries."""

    def __init__(self):
        info = pyogrio.read_info(NORWAY_BOUNDARY_PATH, layer=NORWAY_BOUNDARY_LAYER)
        if not info["crs"]:
            raise ValueError("Municipality layer has no CRS.")
        fields = {field.lower(): field for field in info["fields"]}
        field = fields.get(MUNICIPALITY_ID_FIELD.lower())
        if field is None:
            raise ValueError(
                f"Municipality field {MUNICIPALITY_ID_FIELD!r} not found. "
                f"Set MUNICIPALITY_ID_FIELD to the correct field; available: {info['fields']}"
            )
        logging.info(
            "Reading mainland boundary: %s | layer=%s",
            NORWAY_BOUNDARY_PATH,
            NORWAY_BOUNDARY_LAYER,
        )
        frame = pyogrio.read_dataframe(
            NORWAY_BOUNDARY_PATH, layer=NORWAY_BOUNDARY_LAYER, columns=[field]
        )
        codes = pd.to_numeric(frame[field], errors="coerce")
        if (codes.isna() | (codes % 1 != 0) | (codes < 0) | (codes > 9999)).any():
            raise ValueError(
                "Municipality layer has missing or invalid municipality numbers."
            )
        codes = codes.astype("int64").astype(str).str.zfill(4)
        exclude = codes.str[:2].isin(EXCLUDED_MUNICIPALITY_PREFIXES)
        logging.info(
            "Excluded %d municipality features for Svalbard/Jan Mayen.",
            int(exclude.sum()),
        )
        frame = frame.loc[~exclude].copy()
        if frame.empty or frame.geometry.isna().any() or frame.geometry.is_empty.any():
            raise ValueError(
                "Mainland boundary is empty or contains missing/empty geometries."
            )
        frame.geometry = frame.geometry.map(make_valid)
        frame = frame.to_crs(REGION_CRS)

        def polygon_parts(geometry):
            if geometry.geom_type == "Polygon":
                yield geometry
            elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
                for part in geometry.geoms:
                    yield from polygon_parts(part)

        parts = []
        for geometry in frame.geometry:
            polygons = list(polygon_parts(make_valid(geometry)))
            if not polygons:
                raise ValueError(
                    "A municipality feature has no usable polygon geometry."
                )
            parts.extend(polygons)
        logging.info(
            "Combining %d municipality polygons; internal borders are dissolved.",
            len(parts),
        )
        self.geometry = unary_union(parts)
        if self.geometry.is_empty or not self.geometry.is_valid:
            raise ValueError("Could not construct a valid mainland boundary.")
        self.prepared = prep(self.geometry)

    def covers(self, polygon):
        # No buffer or percentage allowance: all of the footprint must fit.
        # Touching the outer boundary is allowed; extending beyond it is not.
        return bool(self.prepared.covers(polygon))


class WaterFilter:
    def __init__(self):
        info = pyogrio.read_info(AR5_PATH, layer=AR5_LAYER)
        if not info["crs"]:
            raise ValueError("AR5 layer has no CRS")
        self.crs = info["crs"]
        fields = {f.lower(): f for f in info["fields"]}
        if AR5_FIELD.lower() not in fields:
            raise ValueError(
                f"AR5 field {AR5_FIELD} missing; available: {info['fields']}"
            )
        self.field = fields[AR5_FIELD.lower()]

    def percentages(self, query_polygon, metric_polygon):
        frame = pyogrio.read_dataframe(
            AR5_PATH, layer=AR5_LAYER, columns=[self.field], bbox=query_polygon.bounds
        )
        if frame.empty:
            return 0.0, 0.0
        # Repair before clipping; only transform the local clipped geometry.
        frame.geometry = frame.geometry.map(
            lambda g: make_valid(g) if g is not None else None
        )
        frame.geometry = frame.geometry.intersection(query_polygon)
        frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
        frame = frame.to_crs(REGION_CRS)
        codes = pd.to_numeric(frame[self.field], errors="coerce")

        def percent(selected):
            geometries = frame.loc[selected, "geometry"]
            if geometries.empty:
                return 0.0
            area = unary_union(list(geometries)).intersection(metric_polygon).area
            return min(100.0, 100.0 * area / metric_polygon.area)

        return percent(codes.isin(AR5_WATER_CODES)), percent(
            codes.isin(AR5_KNOWN_CODES)
        )


def attempt_crop(state, candidate, category, number, folder, occupied, water, mainland):
    rr, cc = divmod(int(candidate), len(state["cols"]))
    row, col = int(state["rows"][rr]), int(state["cols"][cc])
    window = Window(col, row, TILE_SIZE, TILE_SIZE)
    with rasterio.open(state["source"]) as s2:
        polygon = footprint(s2, window, REGION_CRS)
        if not mainland.covers(polygon):
            return None, "outside_mainland_boundary"
        if occupied.overlaps(polygon, state["source"]):
            return None, "overlap"
        if occupied.too_close(polygon):
            return None, "minimum_spacing"
        coverage = state.get("orthophoto_coverage")
        if coverage is None:
            raise ValueError(
                "Source must be scored for orthophoto coverage before sampling."
            )
        ortho = score_orthophoto_coverage(
            dict(zip(YEARS, coverage[candidate])),
            min_coverage_percent=MIN_COVERAGE_PERCENT,
            min_score=MIN_SCORE,
        )
        if not ortho["accepted"]:
            return None, "orthophoto_threshold"
        image = s2.read(RGB_BAND_INDEXES, window=window, masked=True)
        data = image.data
        valid = ~np.ma.getmaskarray(image).any(axis=0) & np.isfinite(data).all(axis=0)
        if EXCLUDE_ALL_ZERO_S2_PIXELS:
            valid &= np.any(data != 0, axis=0)
        valid_percent = float(valid.mean() * 100)
        if valid_percent < MIN_VALID_S2_PERCENT:
            return None, "invalid_s2"
        with (
            rasterio.open(state["prediction"]) as pred,
            prediction_vrt(pred, s2) as vrt,
        ):
            labels = prediction_values(vrt, window)
        scores = [
            float(np.mean((labels == value) & valid) * 100)
            for value in (URBAN_CLASS_VALUE, CLEARCUT_CLASS_VALUE)
        ]
        pred_percent = float(np.mean(labels != PRED_NODATA) * 100)
        if category < 2 and (
            pred_percent < MIN_VALID_PREDICTION_PERCENT
            or scores[category] < (MIN_URBAN_PERCENT, MIN_CLEARCUT_PERCENT)[category]
        ):
            return None, "change_threshold"
        query = footprint(s2, window, water.crs)
        water_percent, known_percent = water.percentages(query, polygon)
        if water_percent > MAX_WATER_PERCENT or known_percent < MIN_KNOWN_AR5_PERCENT:
            return None, "water_or_unknown_ar5"
        code = mgrs_code(Path(state["source"]))
        tile_id = f"batch_{BATCH_NUMBER}_{CATEGORIES[category]}_{number:06d}"
        filename = f"batch_{BATCH_NUMBER}_{code}_{CATEGORIES[category]}_{number:06d}_r{row:06d}_c{col:06d}.tif"
        record = dict(
            tile_id=tile_id,
            batch_number=BATCH_NUMBER,
            created_at=datetime.now(timezone.utc).isoformat(),
            filename=filename,
            category=CATEGORIES[category],
            mgrs=code,
            source_s2=state["source"],
            source_prediction=state["prediction"],
            year=TARGET_YEAR,
            row=row,
            col=col,
            urban_percent=scores[0],
            clearcut_percent=scores[1],
            valid_s2_percent=valid_percent,
            valid_prediction_percent=pred_percent,
            water_percent=water_percent,
            known_ar5_percent=known_percent,
            footprint_crs=REGION_CRS,
            footprint_wkt=polygon.wkt,
        )
        record.update(
            orthophoto_score=ortho["score"],
            orthophoto_base_score=ortho["base_score"],
            orthophoto_years=",".join(map(str, ortho["available_years"])),
            orthophoto_number_of_years=ortho["number_of_years"],
            orthophoto_has_recent=int(ortho["has_recent_image"]),
            orthophoto_coverage_by_year=json.dumps(
                ortho["coverage_by_year"], sort_keys=True
            ),
        )
        profile = dict(
            driver="GTiff",
            width=TILE_SIZE,
            height=TILE_SIZE,
            count=3,
            dtype=data.dtype,
            crs=s2.crs,
            transform=s2.window_transform(window),
            nodata=s2.nodata,
            photometric="RGB",
            compress="deflate",
            predictor=3 if data.dtype.kind == "f" else 2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            num_threads=COMPRESSION_THREADS,
        )
        data[:, ~valid] = s2.nodata if s2.nodata is not None else 0
        temporary = folder / (filename + ".partial.tif")
        if temporary.exists() or (folder / filename).exists():
            raise FileExistsError(f"Refusing to overwrite an existing tile: {filename}")
        try:
            with rasterio.open(temporary, "w", **profile) as dst:
                dst.write(data)
                dst.write_mask(valid.astype("uint8") * 255)
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                dst.descriptions = ("Red", "Green", "Blue")
                dst.scales = tuple(s2.scales[b - 1] for b in RGB_BAND_INDEXES)
                dst.offsets = tuple(s2.offsets[b - 1] for b in RGB_BAND_INDEXES)
                dst.update_tags(
                    ns="SAMPLING",
                    **{k: str(v) for k, v in record.items() if k != "footprint_wkt"},
                )
            temporary.replace(folder / filename)
        finally:
            temporary.unlink(missing_ok=True)
    return record, "accepted"


def representation_plan(states, remaining, rng):
    """Match unrepresented sources to category slots, preserving scarce categories.

    Capacitated bipartite matching avoids spending the last random slot on a
    source that could use urban when another source can only use random.
    Availability is provisional until the raster/AR5/spacing checks pass.
    """
    choices = {}
    for i, state in enumerate(states):
        if sum(state["selection_counts"]) == 0:
            options = [
                c for c in range(3) if remaining[c] > 0 and len(state["orders"][c])
            ]
            if options:
                choices[i] = list(map(int, rng.permutation(options)))
    owners, assigned = [[], [], []], {}

    def assign(i, visited):
        for c in choices[i]:
            if c in visited:
                continue
            visited.add(c)
            if len(owners[c]) < remaining[c]:
                owners[c].append(i)
                assigned[i] = c
                return True
            for k, other in enumerate(owners[c]):
                if assign(other, visited):
                    owners[c][k] = i
                    assigned[i] = c
                    return True
        return False

    order = list(map(int, rng.permutation(list(choices))))
    order.sort(key=lambda i: len(choices[i]))
    for i in order:
        assign(i, set())
    return assigned


def next_spread_candidate(state, category):
    """Farthest from accepted centres in this source, across all categories.

    First selection and equal-distance ties follow the seeded input order.
    Rejected trials do not affect distances or occupy space.
    """
    order = state["orders"][category]
    order = np.array(
        [
            i
            for i in order
            if int(i) not in state["used"] and int(i) not in state["rejected"]
        ],
        dtype=np.int64,
    )
    if not len(order):
        state["orders"][category] = order
        return None
    position = int(np.argmax(state["nearest_selected_squared"][order]))
    candidate = int(order[position])
    state["orders"][category] = np.delete(order, position)
    return candidate


def sample_region(states, folder, water, database, mainland):
    requested = [N_URBAN_TOTAL, N_CLEARCUT_TOTAL, N_RANDOM_TOTAL]
    counts, attempts = [0, 0, 0], [0, 0, 0]
    limits = [max(1000, n * MAX_ATTEMPTS_PER_REQUESTED_TILE) for n in requested]
    rng = np.random.default_rng(BATCH_NUMBER)
    occupied, reasons = database.occupied, Counter()
    for state in states:
        state["selection_counts"] = [0, 0, 0]
        state["selection_outcomes"] = Counter()
        state["initial_candidate_counts"] = [len(o) for o in state["orders"]]
    eligible = sum(any(state["initial_candidate_counts"]) for state in states)
    if sum(requested) < eligible:
        logging.warning(
            "Only %d requested crops for %d eligible MGRS sources; "
            "representing all requires a larger regional total.",
            sum(requested),
            eligible,
        )
    with (folder / f"samples_batch_{BATCH_NUMBER}.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = None
        while True:
            remaining = [
                requested[c] - counts[c] if attempts[c] < limits[c] else 0
                for c in range(3)
            ]
            plan = representation_plan(states, remaining, rng)
            if plan:
                # Reserve a first crop for as many distinct sources as category budgets allow.
                i = int(rng.choice(list(plan)))
                c = plan[i]
            else:
                pairs = [
                    (i, c)
                    for i, state in enumerate(states)
                    for c in range(3)
                    if remaining[c] > 0 and len(state["orders"][c])
                ]
                if not pairs:
                    break
                # Balance TOTAL crops per source first, category counts second.
                best = min(
                    (
                        sum(states[i]["selection_counts"]),
                        states[i]["selection_counts"][c],
                    )
                    for i, c in pairs
                )
                pairs = [
                    (i, c)
                    for i, c in pairs
                    if (
                        sum(states[i]["selection_counts"]),
                        states[i]["selection_counts"][c],
                    )
                    == best
                ]
                i, c = pairs[int(rng.integers(len(pairs)))]
            state = states[i]
            candidate = next_spread_candidate(state, c)
            if candidate is None:
                continue
            attempts[c] += 1
            record, reason = attempt_crop(
                state, candidate, c, counts[c] + 1, folder, occupied, water, mainland
            )
            reasons[reason] += 1
            state["selection_outcomes"][reason] += 1
            if record is not None:
                database.append_tile(
                    record,
                    folder / record["filename"],
                    from_wkt(record["footprint_wkt"]),
                )
                state["used"].add(candidate)
                counts[c] += 1
                state["selection_counts"][c] += 1
                delta = (
                    state["candidate_centers"] - state["candidate_centers"][candidate]
                )
                squared = np.einsum("ij,ij->i", delta, delta)
                state["nearest_selected_squared"] = np.minimum(
                    state["nearest_selected_squared"], squared
                )
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
                handle.flush()
                logging.info(
                    "REGION | urban=%d/%d clearcut=%d/%d random=%d/%d | MGRS represented=%d/%d",
                    counts[0],
                    requested[0],
                    counts[1],
                    requested[1],
                    counts[2],
                    requested[2],
                    sum(sum(s["selection_counts"]) > 0 for s in states),
                    eligible,
                )
            elif reason != "change_threshold":
                state["rejected"].add(candidate)
            if sum(attempts) % 100 == 0:
                logging.info("Attempts=%d | outcomes=%s", sum(attempts), dict(reasons))
    for c in range(3):
        log = logging.info if counts[c] == requested[c] else logging.warning
        log(
            "FINAL %s: %d/%d from whole region (%d attempts)",
            CATEGORIES[c],
            counts[c],
            requested[c],
            attempts[c],
        )
    logging.info("Rejection counts: %s", dict(reasons))
    return dict(zip(CATEGORIES, counts))


def write_mgrs_summary(states, audit, folder):
    """Include every input MGRS, including missing predictions and API failures."""
    by_source = {state["source"]: state for state in states}
    evidence = {item["source"]: item for item in audit}
    rows = []
    for source in rasters(S2_INPUT_FOLDER, S2_FILE_PATTERN):
        state = by_source.get(str(source))
        counts = state.get("selection_counts", [0, 0, 0]) if state else [0, 0, 0]
        status = evidence.get(str(source), {}).get("status", "missing_prediction")
        if sum(counts):
            status = "represented"
        elif status == "scored":
            status = (
                "no_qualifying_orthophoto_candidates"
                if not state.get("initial_candidate_counts")
                or not any(state["initial_candidate_counts"])
                else "unrepresented_limits_or_rejections"
            )
        row = dict(
            mgrs=mgrs_code(source),
            source_s2=str(source),
            total=sum(counts),
            **dict(zip(CATEGORIES, counts)),
            status=status,
            candidate_counts=json.dumps(
                state.get("initial_candidate_counts", []) if state else []
            ),
            outcomes=json.dumps(
                dict(state.get("selection_outcomes", {})) if state else {}
            ),
            coverage_error=evidence.get(str(source), {}).get("error", ""),
        )
        rows.append(row)
        if not row["total"]:
            logging.warning(
                "UNREPRESENTED %s | %s | outcomes=%s",
                row["mgrs"],
                status,
                row["outcomes"],
            )
    if rows:
        path = folder / f"mgrs_summary_batch_{BATCH_NUMBER}.csv"
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logging.info(
            "MGRS representation: %d/%d input sources | summary: %s",
            sum(row["total"] > 0 for row in rows),
            len(rows),
            path,
        )


def main():
    setup_logging()
    totals = (N_URBAN_TOTAL, N_CLEARCUT_TOTAL, N_RANDOM_TOTAL)
    if any(type(n) is not int or n < 0 for n in totals) or not sum(totals):
        raise ValueError(
            "Regional totals must be nonnegative integers with a positive sum"
        )
    if TILE_SIZE != 256 or not 1 <= CANDIDATE_STRIDE <= TILE_SIZE:
        raise ValueError("Use TILE_SIZE=256 and CANDIDATE_STRIDE from 1 to 256")
    if (
        len(RGB_BAND_INDEXES) != 3
        or len(set(RGB_BAND_INDEXES)) != 3
        or any(type(b) is not int or b < 1 for b in RGB_BAND_INDEXES)
    ):
        raise ValueError("Set three different positive RGB band indexes")
    if any(
        not 0 < p <= 100
        for p in (
            MIN_URBAN_PERCENT,
            MIN_CLEARCUT_PERCENT,
            MIN_VALID_PREDICTION_PERCENT,
            MIN_VALID_S2_PERCENT,
        )
    ):
        raise ValueError("Change and valid-data thresholds must be in (0,100]")
    if not 0 <= MAX_WATER_PERCENT <= 100 or not 0 <= MIN_KNOWN_AR5_PERCENT <= 100:
        raise ValueError("AR5 percentages must be in [0,100]")
    if (
        PREP_WORKERS < 1
        or MAX_ATTEMPTS_PER_REQUESTED_TILE < 1
        or CHANGE_WEIGHT_POWER < 0
    ):
        raise ValueError("Invalid workers, attempts, random seed or weight")
    if URBAN_CLASS_VALUE == CLEARCUT_CLASS_VALUE or any(
        type(v) is not int or not PRED_NODATA < v <= np.iinfo(np.int32).max
        for v in (URBAN_CLASS_VALUE, CLEARCUT_CLASS_VALUE)
    ):
        raise ValueError("Use distinct integer class codes within int32 range")
    if {URBAN_CLASS_VALUE, CLEARCUT_CLASS_VALUE}.intersection(
        PREDICTION_EXTRA_NODATA_VALUES
    ):
        raise ValueError("Target classes cannot be NoData")
    if not CRS(REGION_CRS).is_projected or any(
        a.unit_name != "metre" for a in CRS(REGION_CRS).axis_info
    ):
        raise ValueError("REGION_CRS must be projected in metres")
    if (
        not isfinite(MIN_SAMPLE_CENTER_DISTANCE_METRES)
        or MIN_SAMPLE_CENTER_DISTANCE_METRES < 0
    ):
        raise ValueError("Minimum centre spacing must be finite and nonnegative.")
    if CROSS_SOURCE_MARGIN_METRES < 0:
        raise ValueError("Cross-source margin must be nonnegative")
    if S2_INPUT_FOLDER.resolve().is_relative_to(OUTPUT_FOLDER.resolve()):
        raise ValueError("Output folder must not contain the S2 input folder")
    if type(BATCH_NUMBER) is not int or not 0 <= BATCH_NUMBER <= 2**32 - 1:
        raise ValueError(
            "BATCH_NUMBER must be a nonnegative integer smaller than 2**32"
        )
    if (
        NIB_REQUEST_ATTEMPTS < 1
        or NIB_TIMEOUT_SECONDS <= 0
        or NIB_CACHE_MAX_AGE_DAYS < 0
    ):
        raise ValueError("Invalid Norge i Bilder retry, timeout or cache-age setting.")
    if any(
        not isfinite(v) or v <= 0
        for v in (
            NIB_MIN_REQUEST_INTERVAL_SECONDS,
            NIB_RETRY_BASE_SECONDS,
            NIB_RETRY_MAX_SECONDS,
        )
    ):
        raise ValueError(
            "Norge i Bilder request interval and backoff must be finite and positive."
        )
    if tuple(sorted(set(YEARS))) != YEARS or not YEARS or TEMPORAL_DISTANCE_LIMIT <= 0:
        raise ValueError(
            "YEARS must be nonempty, unique and ascending; temporal distance must be positive."
        )
    if not 1 <= MIN_AVAILABLE_YEARS <= len(YEARS) or not set(RECENT_YEARS).issubset(
        YEARS
    ):
        raise ValueError("Invalid minimum year count or recent years.")
    score_orthophoto_coverage(
        dict.fromkeys(YEARS, 0.0),
        min_coverage_percent=MIN_COVERAGE_PERCENT,
        min_score=MIN_SCORE,
    )
    folder = OUTPUT_FOLDER / f"batch_{BATCH_NUMBER}"
    with sampler_lock():
        if folder.exists():
            raise FileExistsError(
                f"Batch {BATCH_NUMBER} already exists: {folder}. "
                "Choose a new BATCH_NUMBER. Existing files were not changed."
            )
        jobs = matched_jobs()
        if not jobs:
            logging.warning("No matched S2 rasters. Nothing to export.")
            return
        water = WaterFilter()
        mainland = MainlandFilter()
        occupied = previous_footprints()
        if PREP_WORKERS == 1:
            states = [prepare_source(job) for job in jobs]
        else:
            with ProcessPoolExecutor(
                max_workers=min(PREP_WORKERS, len(jobs)),
                mp_context=mp.get_context("spawn"),
                initializer=setup_logging,
            ) as pool:
                states = list(pool.map(prepare_source, jobs))
        orthophotos = OrthophotoCoverage()
        for state in states:
            orthophotos.rank_source(state, mainland, occupied)
        # No output batch is reserved if coverage cannot yield any candidate.
        if not any(len(order) for state in states for order in state["orders"]):
            logging.warning(
                "No qualifying orthophoto candidates. No batch created. "
                "Check the per-source coverage/query messages above."
            )
            return
        folder.mkdir(parents=True, exist_ok=False)
        database = BatchDatabase(folder, occupied)
        settings = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in globals().items()
            if k.isupper() and isinstance(v, (str, int, float, bool, list, tuple, Path))
        }
        with (folder / f"settings_batch_{BATCH_NUMBER}.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(settings, handle, indent=2)
        with (folder / f"orthophoto_sources_batch_{BATCH_NUMBER}.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(orthophotos.audit, handle, indent=2, ensure_ascii=False)
        logging.info(
            "Batch=%d | random seed=%d | output=%s", BATCH_NUMBER, BATCH_NUMBER, folder
        )
        with rasterio.Env(
            GDAL_CACHEMAX=GDAL_CACHE_MB_PER_WORKER * 1024**2,
            GDAL_TIFF_INTERNAL_MASK=True,
        ):
            sample_region(states, folder, water, database, mainland)
        write_mgrs_summary(states, orthophotos.audit, folder)
        logging.info("Output: %s | outlines: %s", folder, database.path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
