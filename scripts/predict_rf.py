#!/usr/bin/env python3
"""
Per-tile AlphaEarth + Sentinel-2 Random Forest prediction.

This predictor uses the same preprocessing and model inputs as the completed
training workflow, but it predicts and FINISHES one Sentinel-2 tile at a time.

For each tile it writes separate EPSG:25833 rasters:

    <TILE>_change_class_2019_2025_EPSG25833.tif
    <TILE>_change_confidence_2019_2025_EPSG25833.tif
    <TILE>_change_year_2019_2025_EPSG25833.tif   (optional)

Each tile GeoTIFF is closed and its overviews are built before prediction moves
to the next tile. This means completed tiles can be opened immediately in
ArcGIS/QGIS while the national job continues.

Training compatibility
----------------------
The predictor reuses:

    models/national/rf_final.joblib
    input_indexes/s2_tile_index.csv
    metrics/alphaearth_tile_source_selection.csv
    training_config.json
    feature_schema.json

Prediction therefore uses the same S2 tile list and the same AlphaEarth source
zone selected during training.

Target grid
-----------
Every output tile is a 10 m EPSG:25833 raster. The tile grid is derived from
the 2019 S2 raster exactly as in training.

S2 source CRS:
    2018       -> stored native CRS, warped to EPSG:25833
    2019-2025  -> interpreted as EPSG:25833

S2 raster indexes:
    B2 -> 1
    B3 -> 2
    B4 -> 3
    B8 -> 7

Features for visible year Y
---------------------------
    AE Y-1                         64
    AE Y                           64
    AE Y - AE Y-1                 64
    S2 Y-1 indexes 1,2,3,7         4
    S2 Y indexes 1,2,3,7           4
    S2 Y - S2 Y-1                  4
    visible year                    1
                                  ---
                                  205

Temporal decision
-----------------
For every visible year, change is compared with SAME-YEAR nochange.

Clearcut is accepted when:
    p(clearcut) >= CLEARCUT_PROBABILITY_THRESHOLD
    and p(clearcut) - p(nochange) >= CHANGE_VS_NOCHANGE_MARGIN

Urban is accepted when:
    p(urban) >= URBAN_PROBABILITY_THRESHOLD
    and p(urban) - p(nochange) >= CHANGE_VS_NOCHANGE_MARGIN

The strongest accepted change across the valid years wins. If no change is
accepted, the final class is nochange and confidence is the highest nochange
probability from the valid years.

Output class codes
------------------
    0 = nodata / invalid
    1 = clearcut
    2 = urban
    3 = nochange

Confidence is uint16 probability * 10,000.
"""

from __future__ import annotations

# =============================================================================
# ENVIRONMENT -- set before NumPy / Rasterio / scikit-learn imports
# =============================================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "65536"
os.environ["GDAL_NUM_THREADS"] = "16"

# =============================================================================
# IMPORTS
# =============================================================================

import gc
import json
import logging
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

# =============================================================================
# USER SETTINGS
# =============================================================================

# Completed training run.
TRAINING_RUN_ROOT = Path(
    r"F:\Ajourhold 2026\ML_model_S2"
    r"\rf_alphaearth_s2_national_epsg25833_fixed_s2_bands_aoi_filtered_fast"
)

MODEL_PATH = TRAINING_RUN_ROOT / "models" / "national" / "rf_final.joblib"
TRAINING_S2_INDEX_CSV = TRAINING_RUN_ROOT / "input_indexes" / "s2_tile_index.csv"
TRAINING_AE_SELECTION_CSV = (
    TRAINING_RUN_ROOT / "metrics" / "alphaearth_tile_source_selection.csv"
)
TRAINING_CONFIG_JSON = TRAINING_RUN_ROOT / "training_config.json"
TRAINING_FEATURE_SCHEMA_JSON = TRAINING_RUN_ROOT / "feature_schema.json"

ALPHA_ROOT = Path(r"F:\Data\AE\data\data")
AE_MOSAIC_TEMPLATE = "AlphaEarth_mosaic_{year}_{zone}"

OUTPUT_ROOT = Path(r"F:\Ajourhold 2026\ML_model_S2\predictions_national_epsg25833")

# Per-tile outputs are written directly into the SAME output folder as the
# previous national predictor. The filenames are unique and will not overwrite
# the existing Norway_change_*.tif national rasters.
TILE_SUMMARY_CSV = OUTPUT_ROOT / "prediction_per_tile_summary.csv"
PREFLIGHT_SUMMARY_CSV = OUTPUT_ROOT / "prediction_per_tile_preflight_summary.csv"
COMPLETED_TILES_JSON = OUTPUT_ROOT / "completed_tiles_per_tile.json"
RUN_SUMMARY_JSON = OUTPUT_ROOT / "prediction_per_tile_run_summary.json"
LOG_PATH = OUTPUT_ROOT / "predict_per_tile_epsg25833.log"

# None = predict every tile from the training S2 index.
# For a test, e.g. PREDICTION_TILE_IDS = ["T32VKM"]
PREDICTION_TILE_IDS: list[str] | None = None

# Tiles known to be outside the actual AOI. Training already excludes them,
# but prediction filters them again as a safety check.
EXCLUDED_S2_TILE_IDS = {
    "T32VKK",
    "T32VLJ",
    "T32VLR",
    "T32VMJ",
    "T32WNA",
    "T32WNU",
    "T32WNV",
    "T32WPB",
    "T32WPC",
    "T33WVT",
    "T33WWU",
    "T34WEE",
    "T35WMV",
    "T35WNV",
    "T35WPS",
    "T35WPV",
}

SOURCE_YEARS = list(range(2018, 2026))
VISIBLE_YEARS = list(range(2019, 2026))

TARGET_EPSG = 25833
TARGET_CRS_TEXT = "EPSG:25833"
TARGET_RESOLUTION_METRES = 10.0
S2_REFERENCE_YEAR = 2019
S2_2019_PLUS_SOURCE_EPSG = 25833

S2_RESAMPLING = Resampling.nearest
AE_RESAMPLING = Resampling.nearest

# Exact same fixed raster indexes as training.
S2_BAND_NAMES = ["B2", "B3", "B4", "B8"]
S2_BAND_INDEXES = [1, 2, 3, 7]

EXPECTED_AE_BANDS = 64
EXPECTED_FEATURES = 205
EXPECTED_MODEL_CLASSES = [0, 1, 2]

# Decision thresholds.
CLEARCUT_PROBABILITY_THRESHOLD = 0.55
URBAN_PROBABILITY_THRESHOLD = 0.60
CHANGE_VS_NOCHANGE_MARGIN = 0.05

# Output codes.
OUTPUT_NODATA = 0
OUTPUT_CLEARCUT = 1
OUTPUT_URBAN = 2
OUTPUT_NOCHANGE = 3
CONFIDENCE_SCALE = 10000

# Performance for the high-end workstation.
RF_N_JOBS = 96
PREDICTION_BLOCK_SIZE = 1024
PIXEL_PREDICTION_BATCH_SIZE = 500_000
GDAL_WARP_MEMORY_MB = 4096
OUTPUT_BLOCK_SIZE = 1024

# Diagnostics. Coverage failures skip the affected year/tile instead of
# terminating the full national run. Unexpected errors still raise.
RUN_TILE_PREFLIGHT = True
PREFLIGHT_COARSE_MAX_DIM = 1024
PREFLIGHT_EXACT_BLOCK_SIZE = 512
SKIP_TILE_IF_NO_VALID_YEARS = True
STRICT_TRAINING_CONFIG_COMPATIBILITY = True

# Outputs / resume.
WRITE_YEAR_RASTER = True

# True is recommended. A tile is skipped only when its _SUCCESS.json and all
# requested output rasters exist. Incomplete tile outputs are deleted and
# rebuilt from scratch.
RESUME_COMPLETED_TILES = True
OVERWRITE_INCOMPLETE_TILE_OUTPUTS = True

# Build overviews immediately after EACH tile has been closed.
BUILD_OVERVIEWS = True
OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64, 128]

# Count final classes per tile after prediction.
COMPUTE_TILE_COUNTS = True

# GeoTIFF.
GTIFF_COMPRESSION = "ZSTD"
GTIFF_ZSTD_LEVEL = 6
GTIFF_PREDICTOR = 2

LOG_EVERY_N_BLOCKS = 10

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class GridDefinition:
    crs_wkt: str
    transform_values: tuple[float, float, float, float, float, float]
    width: int
    height: int
    bounds_values: tuple[float, float, float, float]

    @property
    def crs(self) -> CRS:
        return CRS.from_wkt(self.crs_wkt)

    @property
    def transform(self):
        return rasterio.Affine(*self.transform_values)


@dataclass(frozen=True)
class TileTask:
    tile_id: str
    grid: GridDefinition
    s2_paths: dict[int, Path]
    ae_source_zone: str


# =============================================================================
# SETUP / VALIDATION
# =============================================================================


def setup() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def normalise_tile_id(value: object) -> str:
    return str(value).strip().upper()


def load_training_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    require_path(TRAINING_CONFIG_JSON, "training configuration")
    require_path(TRAINING_FEATURE_SCHEMA_JSON, "training feature schema")
    config = json.loads(TRAINING_CONFIG_JSON.read_text(encoding="utf-8"))
    schema = json.loads(TRAINING_FEATURE_SCHEMA_JSON.read_text(encoding="utf-8"))
    return config, schema


def validate_training_metadata(config: dict[str, Any], schema: dict[str, Any]) -> None:
    problems: list[str] = []

    if str(config.get("TARGET_CRS")) != TARGET_CRS_TEXT:
        problems.append(
            f"TARGET_CRS training={config.get('TARGET_CRS')} prediction={TARGET_CRS_TEXT}"
        )

    if int(config.get("S2_2019_PLUS_SOURCE_EPSG", -1)) != S2_2019_PLUS_SOURCE_EPSG:
        problems.append(
            "S2_2019_PLUS_SOURCE_EPSG training="
            f"{config.get('S2_2019_PLUS_SOURCE_EPSG')} prediction={S2_2019_PLUS_SOURCE_EPSG}"
        )

    expected_mapping = {"B2": 1, "B3": 2, "B4": 3, "B8": 7}
    if config.get("S2_BAND_FALLBACK_INDEXES") != expected_mapping:
        problems.append(
            "S2 band mapping training="
            f"{config.get('S2_BAND_FALLBACK_INDEXES')} prediction={expected_mapping}"
        )

    if int(schema.get("expected_features_for_64_band_ae", -1)) != EXPECTED_FEATURES:
        problems.append(
            "feature count training="
            f"{schema.get('expected_features_for_64_band_ae')} prediction={EXPECTED_FEATURES}"
        )

    if problems:
        message = "Training/prediction mismatch:\n- " + "\n- ".join(problems)
        if STRICT_TRAINING_CONFIG_COMPATIBILITY:
            raise RuntimeError(message)
        logging.warning(message)


def load_model():
    require_path(MODEL_PATH, "trained national model")
    logging.info("Loading model: %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    if hasattr(model, "n_jobs"):
        model.n_jobs = RF_N_JOBS

    n_features = getattr(model, "n_features_in_", None)
    classes = np.asarray(getattr(model, "classes_", [])).astype(int).tolist()

    if n_features is None or int(n_features) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Model has {n_features} features; expected {EXPECTED_FEATURES}."
        )
    if classes != EXPECTED_MODEL_CLASSES:
        raise RuntimeError(
            f"Model classes are {classes}; expected {EXPECTED_MODEL_CLASSES}."
        )

    logging.info(
        "Model loaded | type=%s | trees=%d | features=%d | classes=%s | n_jobs=%s",
        type(model).__name__,
        len(getattr(model, "estimators_", [])),
        int(n_features),
        classes,
        getattr(model, "n_jobs", None),
    )
    return model


# =============================================================================
# TRAINING INDEX / AE SOURCE SELECTION
# =============================================================================


def read_training_s2_index() -> pd.DataFrame:
    require_path(TRAINING_S2_INDEX_CSV, "training S2 index")
    table = pd.read_csv(TRAINING_S2_INDEX_CSV)

    required = {"year", "tile_id", "path"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"S2 index missing columns: {sorted(missing)}")

    table = table.copy()
    table["year"] = pd.to_numeric(table["year"], errors="raise").astype(int)
    table["tile_id"] = table["tile_id"].map(normalise_tile_id)
    table = table[~table["tile_id"].isin(EXCLUDED_S2_TILE_IDS)].copy()

    if PREDICTION_TILE_IDS is not None:
        requested = {normalise_tile_id(x) for x in PREDICTION_TILE_IDS}
        available = set(table["tile_id"].unique())
        missing_requested = sorted(requested - available)
        if missing_requested:
            raise ValueError(
                f"Requested tiles not present in training S2 index: {missing_requested}"
            )
        table = table[table["tile_id"].isin(requested)].copy()

    duplicates = table.groupby(["year", "tile_id"]).size()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        raise RuntimeError(f"Duplicate S2 year/tile rows:\n{duplicates}")

    for value in table["path"]:
        require_path(Path(value), "Sentinel-2 raster")

    logging.info(
        "Using training S2 index | rows=%d | tiles=%d | years=%s",
        len(table),
        table["tile_id"].nunique(),
        sorted(table["year"].unique()),
    )
    return table


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def read_training_ae_selection() -> dict[str, str]:
    require_path(TRAINING_AE_SELECTION_CSV, "training AE selection table")
    table = pd.read_csv(TRAINING_AE_SELECTION_CSV)

    required = {"tile_id", "source_zone", "selected"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"AE selection table missing columns: {sorted(missing)}")

    table = table.copy()
    table["tile_id"] = table["tile_id"].map(normalise_tile_id)
    selected = table[parse_bool_series(table["selected"])].copy()

    counts = selected.groupby("tile_id").size()
    bad = counts[counts != 1]
    if not bad.empty:
        raise RuntimeError(
            f"Expected one selected AE source per tile, but found:\n{bad}"
        )

    mapping = {str(row.tile_id): str(row.source_zone) for row in selected.itertuples()}
    logging.info("Loaded %d AE tile-source selections from training.", len(mapping))
    return mapping


# =============================================================================
# SOURCE LOOKUP / GRID LOGIC -- same rules as training
# =============================================================================


def find_ae_mosaic(year: int, zone: str) -> Path:
    folder = ALPHA_ROOT / str(year) / zone
    stem = AE_MOSAIC_TEMPLATE.format(year=year, zone=zone)

    if not folder.exists():
        raise FileNotFoundError(f"Missing AE folder: {folder}")

    for suffix in (".tif", ".tiff"):
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    candidates: list[Path] = []
    for pattern in (f"{stem}*.tif", f"{stem}*.tiff"):
        candidates.extend(folder.glob(pattern))
    candidates = sorted(set(candidates))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple AE mosaics found for {year} {zone}:\n"
            + "\n".join(str(p) for p in candidates)
        )
    raise FileNotFoundError(f"No AE mosaic found for {year} {zone}.")


def target_crs() -> CRS:
    return CRS.from_epsg(TARGET_EPSG)


def effective_s2_source_crs(src, source_year: int) -> CRS:
    if source_year == 2018:
        if src.crs is None:
            raise ValueError(f"2018 S2 raster has no CRS: {src.name}")
        return CRS.from_user_input(src.crs)
    if source_year >= 2019:
        return CRS.from_epsg(S2_2019_PLUS_SOURCE_EPSG)
    raise ValueError(f"No S2 source CRS rule for year {source_year}.")


def derive_tile_grid(anchor_src) -> GridDefinition:
    source_crs = effective_s2_source_crs(anchor_src, S2_REFERENCE_YEAR)
    dst_crs = target_crs()
    res = float(TARGET_RESOLUTION_METRES)

    if source_crs == dst_crs:
        left, bottom, right, top = tuple(anchor_src.bounds)
    else:
        left, bottom, right, top = transform_bounds(
            source_crs,
            dst_crs,
            *anchor_src.bounds,
            densify_pts=41,
        )

    left = math.floor(left / res) * res
    bottom = math.floor(bottom / res) * res
    right = math.ceil(right / res) * res
    top = math.ceil(top / res) * res

    width = int(math.ceil((right - left) / res))
    height = int(math.ceil((top - bottom) / res))
    transform = from_origin(left, top, res, res)
    exact_right = left + width * res
    exact_bottom = top - height * res

    return GridDefinition(
        crs_wkt=dst_crs.to_wkt(),
        transform_values=tuple(transform)[:6],
        width=width,
        height=height,
        bounds_values=(left, exact_bottom, exact_right, top),
    )


def build_tile_tasks(
    s2_index: pd.DataFrame,
    ae_mapping: dict[str, str],
) -> list[TileTask]:
    tasks: list[TileTask] = []

    for tile_id in sorted(s2_index["tile_id"].unique()):
        rows = s2_index[s2_index["tile_id"] == tile_id]
        year_to_path = {int(r.year): Path(r.path) for r in rows.itertuples()}

        missing_years = sorted(set(SOURCE_YEARS) - set(year_to_path))
        if missing_years:
            raise RuntimeError(f"Tile {tile_id} is missing S2 years {missing_years}.")
        if tile_id not in ae_mapping:
            raise RuntimeError(f"No training AE source selection for tile {tile_id}.")

        with rasterio.open(year_to_path[S2_REFERENCE_YEAR]) as anchor:
            grid = derive_tile_grid(anchor)

        tasks.append(
            TileTask(
                tile_id=tile_id,
                grid=grid,
                s2_paths=year_to_path,
                ae_source_zone=ae_mapping[tile_id],
            )
        )

    logging.info("Built %d per-tile prediction tasks.", len(tasks))
    return tasks


# National mosaic grid helpers intentionally omitted: outputs are per tile.


def make_warped_vrt(
    src,
    grid: GridDefinition,
    resampling: Resampling,
    src_crs_override: CRS | None = None,
):
    kwargs: dict[str, Any] = {
        "crs": grid.crs,
        "transform": grid.transform,
        "width": grid.width,
        "height": grid.height,
        "resampling": resampling,
        "dtype": "float32",
        "warp_mem_limit": GDAL_WARP_MEMORY_MB,
        "init_dest_nodata": True,
    }

    if src_crs_override is not None:
        kwargs["src_crs"] = src_crs_override
    if src.nodata is not None:
        kwargs["src_nodata"] = src.nodata
        kwargs["nodata"] = src.nodata

    return WarpedVRT(src, **kwargs)


def validate_s2_band_count(src) -> None:
    if src.count < max(S2_BAND_INDEXES):
        raise ValueError(
            f"S2 raster has only {src.count} bands but {S2_BAND_INDEXES} are required: "
            f"{src.name}"
        )


def open_tile_sources(
    stack: ExitStack,
    task: TileTask,
) -> tuple[dict[int, Any], dict[int, Any]]:
    s2_sources: dict[int, Any] = {}
    ae_sources: dict[int, Any] = {}

    for year in SOURCE_YEARS:
        s2_base = stack.enter_context(rasterio.open(task.s2_paths[year]))
        validate_s2_band_count(s2_base)
        s2_sources[year] = stack.enter_context(
            make_warped_vrt(
                src=s2_base,
                grid=task.grid,
                resampling=S2_RESAMPLING,
                src_crs_override=effective_s2_source_crs(s2_base, year),
            )
        )

        ae_path = find_ae_mosaic(year, task.ae_source_zone)
        ae_base = stack.enter_context(rasterio.open(ae_path))
        if ae_base.count != EXPECTED_AE_BANDS:
            raise RuntimeError(
                f"AE raster {ae_path} has {ae_base.count} bands; expected {EXPECTED_AE_BANDS}."
            )
        ae_sources[year] = stack.enter_context(
            make_warped_vrt(
                src=ae_base,
                grid=task.grid,
                resampling=AE_RESAMPLING,
            )
        )

    return s2_sources, ae_sources


# =============================================================================
# WINDOWS / PREFLIGHT
# =============================================================================


def iter_windows(width: int, height: int, block_size: int) -> Iterator[Window]:
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=w, height=h)


def reduced_shape(width: int, height: int, maximum_dimension: int) -> tuple[int, int]:
    scale = max(width / maximum_dimension, height / maximum_dimension, 1.0)
    return (
        max(1, int(math.ceil(height / scale))),
        max(1, int(math.ceil(width / scale))),
    )


def coarse_valid_mask(
    src,
    indexes: list[int] | None,
    out_height: int,
    out_width: int,
) -> np.ndarray:
    if indexes is None:
        masks = src.read_masks(
            out_shape=(src.count, out_height, out_width),
            resampling=Resampling.nearest,
        )
    else:
        masks = src.read_masks(
            indexes=indexes,
            out_shape=(len(indexes), out_height, out_width),
            resampling=Resampling.nearest,
        )
    valid = np.all(masks > 0, axis=0)
    del masks
    return valid


def exact_block_valid_mask(
    src, indexes: list[int] | None, window: Window
) -> np.ndarray:
    if indexes is None:
        masks = src.read_masks(window=window)
    else:
        masks = src.read_masks(indexes=indexes, window=window)
    valid = np.all(masks > 0, axis=0)
    del masks
    return valid


def exact_year_has_combined_valid(
    visible_year: int,
    task: TileTask,
    s2_sources: dict[int, Any],
    ae_sources: dict[int, Any],
) -> bool:
    pre_year = visible_year - 1

    for window in iter_windows(
        task.grid.width,
        task.grid.height,
        PREFLIGHT_EXACT_BLOCK_SIZE,
    ):
        ae_pre = exact_block_valid_mask(ae_sources[pre_year], None, window)
        ae_post = exact_block_valid_mask(ae_sources[visible_year], None, window)
        s2_pre = exact_block_valid_mask(s2_sources[pre_year], S2_BAND_INDEXES, window)
        s2_post = exact_block_valid_mask(
            s2_sources[visible_year], S2_BAND_INDEXES, window
        )
        combined = ae_pre & ae_post & s2_pre & s2_post
        found = bool(np.any(combined))
        del ae_pre, ae_post, s2_pre, s2_post, combined
        if found:
            return True
    return False


def run_tile_preflight(
    task: TileTask,
    s2_sources: dict[int, Any],
    ae_sources: dict[int, Any],
) -> tuple[list[int], list[dict[str, Any]]]:
    if not RUN_TILE_PREFLIGHT:
        return list(VISIBLE_YEARS), []

    out_h, out_w = reduced_shape(
        task.grid.width,
        task.grid.height,
        PREFLIGHT_COARSE_MAX_DIM,
    )

    valid_years: list[int] = []
    rows: list[dict[str, Any]] = []

    for year in VISIBLE_YEARS:
        pre = year - 1
        ae_pre = coarse_valid_mask(ae_sources[pre], None, out_h, out_w)
        ae_post = coarse_valid_mask(ae_sources[year], None, out_h, out_w)
        s2_pre = coarse_valid_mask(s2_sources[pre], S2_BAND_INDEXES, out_h, out_w)
        s2_post = coarse_valid_mask(s2_sources[year], S2_BAND_INDEXES, out_h, out_w)
        combined = ae_pre & ae_post & s2_pre & s2_post

        counts = {
            "ae_pre": int(ae_pre.sum()),
            "ae_post": int(ae_post.sum()),
            "s2_pre": int(s2_pre.sum()),
            "s2_post": int(s2_post.sum()),
            "combined": int(combined.sum()),
        }

        exact_used = counts["combined"] == 0
        exact_valid = False
        if exact_used:
            exact_valid = exact_year_has_combined_valid(
                visible_year=year,
                task=task,
                s2_sources=s2_sources,
                ae_sources=ae_sources,
            )

        accepted = counts["combined"] > 0 or exact_valid
        if accepted:
            valid_years.append(year)

        rows.append(
            {
                "tile_id": task.tile_id,
                "visible_year": year,
                "ae_source_zone": task.ae_source_zone,
                "coarse_height": out_h,
                "coarse_width": out_w,
                **counts,
                "exact_fallback_used": exact_used,
                "exact_combined_valid": exact_valid,
                "accepted_for_prediction": accepted,
            }
        )

        logging.info(
            "PREFLIGHT | tile=%s year=%d | AEpre=%d AEpost=%d "
            "S2pre=%d S2post=%d combined=%d | exact=%s | accepted=%s",
            task.tile_id,
            year,
            counts["ae_pre"],
            counts["ae_post"],
            counts["s2_pre"],
            counts["s2_post"],
            counts["combined"],
            exact_used,
            accepted,
        )

        del ae_pre, ae_post, s2_pre, s2_post, combined

    return valid_years, rows


# =============================================================================
# FEATURES / TEMPORAL MODEL LOGIC
# =============================================================================


def build_valid_mask(arrays_and_nodata: list[tuple[np.ndarray, Any]]) -> np.ndarray:
    first = arrays_and_nodata[0][0]
    valid = np.ones((first.shape[1], first.shape[2]), dtype=bool)

    for array, nodata in arrays_and_nodata:
        valid &= np.all(np.isfinite(array), axis=0)
        if nodata is not None:
            valid &= ~np.any(array == nodata, axis=0)
    return valid


def build_feature_batch(
    ae_pre: np.ndarray,
    ae_post: np.ndarray,
    s2_pre: np.ndarray,
    s2_post: np.ndarray,
    flat_indexes: np.ndarray,
    visible_year: int,
) -> np.ndarray:
    ae_pre_flat = ae_pre.reshape(ae_pre.shape[0], -1)
    ae_post_flat = ae_post.reshape(ae_post.shape[0], -1)
    s2_pre_flat = s2_pre.reshape(s2_pre.shape[0], -1)
    s2_post_flat = s2_post.reshape(s2_post.shape[0], -1)

    a0 = ae_pre_flat[:, flat_indexes].T.astype(np.float32, copy=False)
    a1 = ae_post_flat[:, flat_indexes].T.astype(np.float32, copy=False)
    s0 = s2_pre_flat[:, flat_indexes].T.astype(np.float32, copy=False)
    s1 = s2_post_flat[:, flat_indexes].T.astype(np.float32, copy=False)
    year_column = np.full((len(flat_indexes), 1), float(visible_year), dtype=np.float32)

    X = np.concatenate(
        [a0, a1, a1 - a0, s0, s1, s1 - s0, year_column],
        axis=1,
        dtype=np.float32,
    )

    if X.shape[1] != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Created {X.shape[1]} features; expected {EXPECTED_FEATURES}."
        )
    return X


def update_year_probabilities(
    model,
    visible_year: int,
    ae_pre: np.ndarray,
    ae_post: np.ndarray,
    s2_pre: np.ndarray,
    s2_post: np.ndarray,
    ae_pre_nodata: Any,
    ae_post_nodata: Any,
    s2_pre_nodata: Any,
    s2_post_nodata: Any,
    any_valid: np.ndarray,
    best_change_probability: np.ndarray,
    best_change_class: np.ndarray,
    best_change_year: np.ndarray,
    best_nochange_probability: np.ndarray,
) -> int:
    valid = build_valid_mask(
        [
            (ae_pre, ae_pre_nodata),
            (ae_post, ae_post_nodata),
            (s2_pre, s2_pre_nodata),
            (s2_post, s2_post_nodata),
        ]
    )
    valid_indexes = np.flatnonzero(valid.ravel())
    if len(valid_indexes) == 0:
        return 0

    any_valid_flat = any_valid.ravel()
    best_change_prob_flat = best_change_probability.ravel()
    best_change_class_flat = best_change_class.ravel()
    best_change_year_flat = best_change_year.ravel()
    best_nochange_flat = best_nochange_probability.ravel()
    any_valid_flat[valid_indexes] = True

    predicted_rows = 0

    for start in range(0, len(valid_indexes), PIXEL_PREDICTION_BATCH_SIZE):
        batch_indexes = valid_indexes[start : start + PIXEL_PREDICTION_BATCH_SIZE]
        X = build_feature_batch(
            ae_pre,
            ae_post,
            s2_pre,
            s2_post,
            batch_indexes,
            visible_year,
        )
        probability = model.predict_proba(X).astype(np.float32, copy=False)
        p_clearcut = probability[:, 0]
        p_urban = probability[:, 1]
        p_nochange = probability[:, 2]

        best_nochange_flat[batch_indexes] = np.maximum(
            best_nochange_flat[batch_indexes],
            p_nochange,
        )

        clearcut_ok = (p_clearcut >= CLEARCUT_PROBABILITY_THRESHOLD) & (
            (p_clearcut - p_nochange) >= CHANGE_VS_NOCHANGE_MARGIN
        )
        urban_ok = (p_urban >= URBAN_PROBABILITY_THRESHOLD) & (
            (p_urban - p_nochange) >= CHANGE_VS_NOCHANGE_MARGIN
        )

        candidate_prob = np.zeros(len(batch_indexes), dtype=np.float32)
        candidate_class = np.zeros(len(batch_indexes), dtype=np.uint8)

        choose_clearcut = clearcut_ok & (~urban_ok | (p_clearcut >= p_urban))
        choose_urban = urban_ok & (~clearcut_ok | (p_urban > p_clearcut))

        candidate_prob[choose_clearcut] = p_clearcut[choose_clearcut]
        candidate_class[choose_clearcut] = OUTPUT_CLEARCUT
        candidate_prob[choose_urban] = p_urban[choose_urban]
        candidate_class[choose_urban] = OUTPUT_URBAN

        update = candidate_prob > best_change_prob_flat[batch_indexes]
        if np.any(update):
            target = batch_indexes[update]
            best_change_prob_flat[target] = candidate_prob[update]
            best_change_class_flat[target] = candidate_class[update]
            best_change_year_flat[target] = visible_year

        predicted_rows += len(batch_indexes)
        del X, probability, p_clearcut, p_urban, p_nochange
        del candidate_prob, candidate_class, choose_clearcut, choose_urban, update

    return predicted_rows


def predict_tile_block(
    model,
    window: Window,
    valid_years: list[int],
    s2_sources: dict[int, Any],
    ae_sources: dict[int, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Predict one spatial block.

    Each annual AE and S2 raster is read only once for the block and then
    reused by the adjacent year-pairs. This is substantially faster than
    rereading the same annual raster as both the post-year of one pair and the
    pre-year of the next pair. With a 1024x1024 block this intentionally uses
    a few GB of RAM to reduce disk and warp I/O on the high-memory workstation.
    """
    h = int(window.height)
    w = int(window.width)

    any_valid = np.zeros((h, w), dtype=bool)
    best_change_probability = np.zeros((h, w), dtype=np.float32)
    best_change_class = np.zeros((h, w), dtype=np.uint8)
    best_change_year = np.zeros((h, w), dtype=np.uint16)
    best_nochange_probability = np.zeros((h, w), dtype=np.float32)
    model_rows = 0

    required_source_years = sorted(
        {
            year
            for visible_year in valid_years
            for year in (visible_year - 1, visible_year)
        }
    )

    ae_arrays: dict[int, np.ndarray] = {}
    s2_arrays: dict[int, np.ndarray] = {}

    for source_year in required_source_years:
        ae_arrays[source_year] = ae_sources[source_year].read(
            window=window,
            out_dtype="float32",
        )
        s2_arrays[source_year] = s2_sources[source_year].read(
            indexes=S2_BAND_INDEXES,
            window=window,
            out_dtype="float32",
        )

    for year in valid_years:
        pre = year - 1
        model_rows += update_year_probabilities(
            model=model,
            visible_year=year,
            ae_pre=ae_arrays[pre],
            ae_post=ae_arrays[year],
            s2_pre=s2_arrays[pre],
            s2_post=s2_arrays[year],
            ae_pre_nodata=ae_sources[pre].nodata,
            ae_post_nodata=ae_sources[year].nodata,
            s2_pre_nodata=s2_sources[pre].nodata,
            s2_post_nodata=s2_sources[year].nodata,
            any_valid=any_valid,
            best_change_probability=best_change_probability,
            best_change_class=best_change_class,
            best_change_year=best_change_year,
            best_nochange_probability=best_nochange_probability,
        )

    out_class = np.zeros((h, w), dtype=np.uint8)
    out_prob = np.zeros((h, w), dtype=np.float32)
    out_year = np.zeros((h, w), dtype=np.uint16)

    change = best_change_class > 0
    nochange = any_valid & ~change

    out_class[change] = best_change_class[change]
    out_prob[change] = best_change_probability[change]
    out_year[change] = best_change_year[change]

    out_class[nochange] = OUTPUT_NOCHANGE
    out_prob[nochange] = best_nochange_probability[nochange]

    out_confidence = np.clip(
        np.rint(out_prob * CONFIDENCE_SCALE),
        0,
        CONFIDENCE_SCALE,
    ).astype(np.uint16)

    del ae_arrays, s2_arrays

    return out_class, out_confidence, out_year, model_rows


# =============================================================================
# PER-TILE OUTPUT PATHS
# =============================================================================


def tile_output_paths(
    tile_id: str,
) -> dict[str, Path]:
    """
    Return all output paths for one tile.

    Files are written directly into OUTPUT_ROOT, which is the same folder used
    by the previous national predictor.
    """
    prefix = f"{tile_id}_change_2019_2025_EPSG25833"

    return {
        "class": (OUTPUT_ROOT / f"{tile_id}_change_class_2019_2025_EPSG25833.tif"),
        "confidence": (
            OUTPUT_ROOT / f"{tile_id}_change_confidence_2019_2025_EPSG25833.tif"
        ),
        "year": (OUTPUT_ROOT / f"{tile_id}_change_year_2019_2025_EPSG25833.tif"),
        "success": (OUTPUT_ROOT / f"{tile_id}_prediction_SUCCESS.json"),
    }


def raster_profile(
    grid: GridDefinition,
    dtype: str,
    nodata: int,
) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": OUTPUT_BLOCK_SIZE,
        "blockysize": OUTPUT_BLOCK_SIZE,
        "compress": GTIFF_COMPRESSION,
        "predictor": GTIFF_PREDICTOR,
        "zstd_level": GTIFF_ZSTD_LEVEL,
        "BIGTIFF": "YES",
        "SPARSE_OK": "TRUE",
        "NUM_THREADS": "ALL_CPUS",
    }


def expected_tile_output_paths(
    tile_id: str,
) -> list[Path]:
    paths = tile_output_paths(tile_id)

    expected = [
        paths["class"],
        paths["confidence"],
    ]

    if WRITE_YEAR_RASTER:
        expected.append(paths["year"])

    return expected


def tile_is_complete(
    tile_id: str,
) -> bool:
    """
    A tile is considered complete only when the success JSON AND all expected
    GeoTIFFs exist.
    """
    paths = tile_output_paths(tile_id)

    if not paths["success"].exists():
        return False

    if not all(path.exists() for path in expected_tile_output_paths(tile_id)):
        return False

    try:
        success = json.loads(paths["success"].read_text(encoding="utf-8"))
    except Exception:
        return False

    return success.get("status") == "completed"


def remove_incomplete_tile_outputs(
    tile_id: str,
) -> None:
    paths = tile_output_paths(tile_id)

    for key in [
        "class",
        "confidence",
        "year",
        "success",
    ]:
        path = paths[key]

        if path.exists():
            path.unlink()


def initialise_tile_output_rasters(
    *,
    task: TileTask,
) -> dict[str, Path]:
    """
    Create empty sparse GeoTIFFs for one tile.

    These files stay open only while this tile is predicted. They are closed
    before overviews are built and before the next tile starts.
    """
    paths = tile_output_paths(task.tile_id)

    if (
        any(path.exists() for path in expected_tile_output_paths(task.tile_id))
        or paths["success"].exists()
    ):
        if OVERWRITE_INCOMPLETE_TILE_OUTPUTS:
            remove_incomplete_tile_outputs(task.tile_id)
        else:
            raise FileExistsError(
                "Incomplete output already exists for "
                f"{task.tile_id}. Set "
                "OVERWRITE_INCOMPLETE_TILE_OUTPUTS=True "
                "to rebuild it."
            )

    with rasterio.open(
        paths["class"],
        "w",
        **raster_profile(
            task.grid,
            "uint8",
            OUTPUT_NODATA,
        ),
    ) as dst:
        dst.update_tags(
            TILE_ID=(task.tile_id),
            CLASS_0="nodata_invalid",
            CLASS_1="clearcut",
            CLASS_2="urban",
            CLASS_3="nochange",
            MODEL_PATH=str(MODEL_PATH),
            VISIBLE_YEARS="2019-2025",
            TARGET_CRS=(TARGET_CRS_TEXT),
            RESOLUTION_METRES=str(TARGET_RESOLUTION_METRES),
            S2_BAND_INDEXES="1,2,3,7",
            AE_SOURCE_ZONE=(task.ae_source_zone),
        )

    with rasterio.open(
        paths["confidence"],
        "w",
        **raster_profile(
            task.grid,
            "uint16",
            0,
        ),
    ) as dst:
        dst.update_tags(
            TILE_ID=(task.tile_id),
            DESCRIPTION=("Final selected-class probability scaled by 10000"),
            SCALE_FACTOR=str(CONFIDENCE_SCALE),
            MODEL_PATH=str(MODEL_PATH),
            TARGET_CRS=(TARGET_CRS_TEXT),
        )

    if WRITE_YEAR_RASTER:
        with rasterio.open(
            paths["year"],
            "w",
            **raster_profile(
                task.grid,
                "uint16",
                0,
            ),
        ) as dst:
            dst.update_tags(
                TILE_ID=(task.tile_id),
                DESCRIPTION=(
                    "Selected clearcut/urban change year; 0 for nochange/nodata"
                ),
                MODEL_PATH=str(MODEL_PATH),
                TARGET_CRS=(TARGET_CRS_TEXT),
            )

    return paths


# =============================================================================
# RESUME STATE
# =============================================================================


def load_completed_tiles() -> set[str]:
    completed: set[str] = set()

    if COMPLETED_TILES_JSON.exists():
        try:
            data = json.loads(COMPLETED_TILES_JSON.read_text(encoding="utf-8"))

            completed.update(
                normalise_tile_id(value) for value in data.get("completed_tiles", [])
            )
        except Exception:
            logging.warning(
                "Could not read %s. "
                "Completion will be checked "
                "from per-tile SUCCESS files.",
                COMPLETED_TILES_JSON,
            )

    # The SUCCESS files are the authoritative resume signal.
    for success_path in OUTPUT_ROOT.glob("T*_prediction_SUCCESS.json"):
        tile_id = success_path.name.split("_prediction_SUCCESS.json")[0].upper()

        if tile_is_complete(tile_id):
            completed.add(tile_id)

    return completed


def save_completed_tiles(
    completed: set[str],
) -> None:
    COMPLETED_TILES_JSON.write_text(
        json.dumps(
            {"completed_tiles": (sorted(completed))},
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# PER-TILE OVERVIEWS AND COUNTS
# =============================================================================


def valid_overview_levels(
    grid: GridDefinition,
) -> list[int]:
    minimum_dimension = min(
        grid.width,
        grid.height,
    )

    return [level for level in OVERVIEW_LEVELS if (minimum_dimension // level) >= 1]


def build_tile_overviews(
    *,
    task: TileTask,
    paths: dict[str, Path],
) -> None:
    if not BUILD_OVERVIEWS:
        return

    levels = valid_overview_levels(task.grid)

    if not levels:
        return

    logging.info(
        "OVERVIEWS START | %s | levels=%s",
        task.tile_id,
        levels,
    )

    with rasterio.open(
        paths["class"],
        "r+",
    ) as dst:
        dst.build_overviews(
            levels,
            Resampling.nearest,
        )

        dst.update_tags(
            ns="rio_overview",
            resampling="nearest",
        )

    with rasterio.open(
        paths["confidence"],
        "r+",
    ) as dst:
        dst.build_overviews(
            levels,
            Resampling.average,
        )

        dst.update_tags(
            ns="rio_overview",
            resampling="average",
        )

    if WRITE_YEAR_RASTER and paths["year"].exists():
        with rasterio.open(
            paths["year"],
            "r+",
        ) as dst:
            dst.build_overviews(
                levels,
                Resampling.nearest,
            )

            dst.update_tags(
                ns="rio_overview",
                resampling="nearest",
            )

    logging.info(
        "OVERVIEWS DONE | %s",
        task.tile_id,
    )


def compute_tile_counts(
    class_path: Path,
) -> dict[str, int]:
    counts = {
        "nodata": 0,
        "clearcut": 0,
        "urban": 0,
        "nochange": 0,
    }

    with rasterio.open(class_path) as src:
        for _, window in src.block_windows(1):
            array = src.read(
                1,
                window=window,
            )

            counts["nodata"] += int(np.count_nonzero(array == OUTPUT_NODATA))

            counts["clearcut"] += int(np.count_nonzero(array == OUTPUT_CLEARCUT))

            counts["urban"] += int(np.count_nonzero(array == OUTPUT_URBAN))

            counts["nochange"] += int(np.count_nonzero(array == OUTPUT_NOCHANGE))

    return counts


# =============================================================================
# TILE PREDICTION
# =============================================================================


def predict_tile(
    *,
    task: TileTask,
    model,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Predict, close, finish and validate one tile before returning.

    The next tile is not started until the current tile GeoTIFFs have been
    closed and their overviews have been built.
    """
    start_time = time.time()

    logging.info(
        "TILE START | %s | AE zone=%s | grid=%dx%d | bounds=%s",
        task.tile_id,
        task.ae_source_zone,
        task.grid.width,
        task.grid.height,
        task.grid.bounds_values,
    )

    total_blocks = math.ceil(task.grid.height / PREDICTION_BLOCK_SIZE) * math.ceil(
        task.grid.width / PREDICTION_BLOCK_SIZE
    )

    totals = {
        "model_rows": 0,
        "valid": 0,
        "clearcut": 0,
        "urban": 0,
        "nochange": 0,
        "written": 0,
    }

    with ExitStack() as source_stack:
        (
            s2_sources,
            ae_sources,
        ) = open_tile_sources(
            source_stack,
            task,
        )

        (
            valid_years,
            preflight_rows,
        ) = run_tile_preflight(
            task,
            s2_sources,
            ae_sources,
        )

        invalid_years = sorted(set(VISIBLE_YEARS) - set(valid_years))

        if not valid_years:
            logging.warning(
                "TILE SKIPPED | %s | no valid prediction years.",
                task.tile_id,
            )

            return (
                {
                    "tile_id": (task.tile_id),
                    "ae_source_zone": (task.ae_source_zone),
                    "status": ("skipped_no_valid_years"),
                    "valid_years": "",
                    "invalid_years": (
                        ",".join(
                            map(
                                str,
                                invalid_years,
                            )
                        )
                    ),
                    "blocks": 0,
                    "model_rows": 0,
                    "tile_valid_pixels": 0,
                    "tile_clearcut_pixels": 0,
                    "tile_urban_pixels": 0,
                    "tile_nochange_pixels": 0,
                    "written_pixels": 0,
                    "class_raster": "",
                    "confidence_raster": "",
                    "year_raster": "",
                    "seconds": (time.time() - start_time),
                },
                preflight_rows,
            )

        paths = initialise_tile_output_rasters(task=task)

        # Open this tile's output rasters only for this tile.
        with ExitStack() as output_stack:
            class_ds = output_stack.enter_context(
                rasterio.open(
                    paths["class"],
                    "r+",
                )
            )

            confidence_ds = output_stack.enter_context(
                rasterio.open(
                    paths["confidence"],
                    "r+",
                )
            )

            year_ds = (
                output_stack.enter_context(
                    rasterio.open(
                        paths["year"],
                        "r+",
                    )
                )
                if WRITE_YEAR_RASTER
                else None
            )

            for (
                block_number,
                tile_window,
            ) in enumerate(
                iter_windows(
                    task.grid.width,
                    task.grid.height,
                    PREDICTION_BLOCK_SIZE,
                ),
                start=1,
            ):
                (
                    new_class,
                    new_confidence,
                    new_year,
                    model_rows,
                ) = predict_tile_block(
                    model=model,
                    window=tile_window,
                    valid_years=(valid_years),
                    s2_sources=(s2_sources),
                    ae_sources=(ae_sources),
                )

                totals["model_rows"] += model_rows

                block_valid = int(np.count_nonzero(new_class != OUTPUT_NODATA))

                totals["valid"] += block_valid

                totals["clearcut"] += int(
                    np.count_nonzero(new_class == OUTPUT_CLEARCUT)
                )

                totals["urban"] += int(np.count_nonzero(new_class == OUTPUT_URBAN))

                totals["nochange"] += int(
                    np.count_nonzero(new_class == OUTPUT_NOCHANGE)
                )

                # Per-tile output: write directly to the same tile window.
                class_ds.write(
                    new_class,
                    1,
                    window=(tile_window),
                )

                confidence_ds.write(
                    new_confidence,
                    1,
                    window=(tile_window),
                )

                if WRITE_YEAR_RASTER and year_ds is not None:
                    year_ds.write(
                        new_year,
                        1,
                        window=(tile_window),
                    )

                totals["written"] += block_valid

                if (
                    block_number == 1
                    or (block_number % LOG_EVERY_N_BLOCKS == 0)
                    or block_number == total_blocks
                ):
                    logging.info(
                        "TILE PROGRESS | %s | "
                        "block=%d/%d | "
                        "model_rows=%s | "
                        "valid=%s | "
                        "clearcut=%s urban=%s "
                        "nochange=%s | "
                        "written=%s",
                        task.tile_id,
                        block_number,
                        total_blocks,
                        f"{totals['model_rows']:,}",
                        f"{totals['valid']:,}",
                        f"{totals['clearcut']:,}",
                        f"{totals['urban']:,}",
                        f"{totals['nochange']:,}",
                        f"{totals['written']:,}",
                    )

                del (
                    new_class,
                    new_confidence,
                    new_year,
                )

                gc.collect()

        # IMPORTANT: output_stack has exited here, so all three GeoTIFFs are
        # fully closed before overviews are created or the next tile starts.
        build_tile_overviews(
            task=task,
            paths=paths,
        )

    class_counts = compute_tile_counts(paths["class"]) if COMPUTE_TILE_COUNTS else {}

    seconds = time.time() - start_time

    summary = {
        "tile_id": (task.tile_id),
        "ae_source_zone": (task.ae_source_zone),
        "status": "completed",
        "valid_years": (
            ",".join(
                map(
                    str,
                    valid_years,
                )
            )
        ),
        "invalid_years": (
            ",".join(
                map(
                    str,
                    invalid_years,
                )
            )
        ),
        "blocks": (total_blocks),
        "model_rows": (totals["model_rows"]),
        "tile_valid_pixels": (totals["valid"]),
        "tile_clearcut_pixels": (totals["clearcut"]),
        "tile_urban_pixels": (totals["urban"]),
        "tile_nochange_pixels": (totals["nochange"]),
        "written_pixels": (totals["written"]),
        "class_raster": str(paths["class"]),
        "confidence_raster": str(paths["confidence"]),
        "year_raster": (str(paths["year"]) if WRITE_YEAR_RASTER else ""),
        "final_class_counts": (class_counts),
        "seconds": (seconds),
    }

    # Write this ONLY after rasters are closed and overviews are complete.
    paths["success"].write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    logging.info(
        "TILE DONE | %s | %.1fs | "
        "valid=%s clearcut=%s urban=%s "
        "nochange=%s | class=%s | "
        "confidence=%s",
        task.tile_id,
        seconds,
        f"{totals['valid']:,}",
        f"{totals['clearcut']:,}",
        f"{totals['urban']:,}",
        f"{totals['nochange']:,}",
        paths["class"],
        paths["confidence"],
    )

    return (
        summary,
        preflight_rows,
    )


# =============================================================================
# CSV UPDATE HELPERS
# =============================================================================


def load_existing_table(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        return pd.read_csv(path).to_dict(orient="records")
    except Exception:
        logging.warning(
            "Could not read existing table: %s",
            path,
        )

        return []


def replace_tile_rows(
    rows: list[dict[str, Any]],
    tile_id: str,
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = [
        row
        for row in rows
        if normalise_tile_id(
            row.get(
                "tile_id",
                "",
            )
        )
        != tile_id
    ]

    result.extend(new_rows)

    return result


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    setup()

    start_time = time.time()

    logging.info(
        "TRAINING_RUN_ROOT: %s",
        TRAINING_RUN_ROOT,
    )

    logging.info(
        "MODEL_PATH: %s",
        MODEL_PATH,
    )

    logging.info(
        "OUTPUT_ROOT: %s",
        OUTPUT_ROOT,
    )

    logging.info(
        "Output mode: ONE TILE AT A TIME. "
        "Each tile is closed and finished "
        "before the next tile starts."
    )

    logging.info(
        "Target CRS: %s at %.1f m",
        TARGET_CRS_TEXT,
        TARGET_RESOLUTION_METRES,
    )

    logging.info(
        "S2 fixed bands: %s -> %s",
        S2_BAND_NAMES,
        S2_BAND_INDEXES,
    )

    logging.info(
        "Visible years: %s",
        VISIBLE_YEARS,
    )

    logging.info(
        "Thresholds: clearcut=%.3f urban=%.3f margin=%.3f",
        CLEARCUT_PROBABILITY_THRESHOLD,
        URBAN_PROBABILITY_THRESHOLD,
        CHANGE_VS_NOCHANGE_MARGIN,
    )

    logging.info(
        "Performance: RF_N_JOBS=%d block=%d pixel_batch=%d warp_mem=%d MB",
        RF_N_JOBS,
        PREDICTION_BLOCK_SIZE,
        PIXEL_PREDICTION_BATCH_SIZE,
        GDAL_WARP_MEMORY_MB,
    )

    (
        config,
        schema,
    ) = load_training_metadata()

    validate_training_metadata(
        config,
        schema,
    )

    model = load_model()

    s2_index = read_training_s2_index()

    ae_mapping = read_training_ae_selection()

    tasks = build_tile_tasks(
        s2_index,
        ae_mapping,
    )

    completed = load_completed_tiles() if RESUME_COMPLETED_TILES else set()

    summaries = load_existing_table(TILE_SUMMARY_CSV)

    preflight_rows = load_existing_table(PREFLIGHT_SUMMARY_CSV)

    run_completed = 0
    run_skipped_completed = 0
    run_skipped_no_valid = 0

    for (
        tile_number,
        task,
    ) in enumerate(
        tasks,
        start=1,
    ):
        if (
            RESUME_COMPLETED_TILES
            and task.tile_id in completed
            and tile_is_complete(task.tile_id)
        ):
            run_skipped_completed += 1

            logging.info(
                "SKIP COMPLETED | tile=%d/%d | %s",
                tile_number,
                len(tasks),
                task.tile_id,
            )

            continue

        logging.info(
            "PREDICTION PROGRESS | tile=%d/%d | %s",
            tile_number,
            len(tasks),
            task.tile_id,
        )

        (
            summary,
            tile_preflight,
        ) = predict_tile(
            task=task,
            model=model,
        )

        summaries = replace_tile_rows(
            summaries,
            task.tile_id,
            [summary],
        )

        preflight_rows = replace_tile_rows(
            preflight_rows,
            task.tile_id,
            tile_preflight,
        )

        pd.DataFrame(summaries).to_csv(
            TILE_SUMMARY_CSV,
            index=False,
        )

        pd.DataFrame(preflight_rows).to_csv(
            PREFLIGHT_SUMMARY_CSV,
            index=False,
        )

        if summary["status"] == "completed":
            run_completed += 1

            completed.add(task.tile_id)

            save_completed_tiles(completed)

        elif summary["status"] == "skipped_no_valid_years":
            run_skipped_no_valid += 1

        gc.collect()

    total_seconds = time.time() - start_time

    run_summary = {
        "training_run_root": str(TRAINING_RUN_ROOT),
        "model_path": str(MODEL_PATH),
        "output_root": str(OUTPUT_ROOT),
        "output_mode": ("per_tile_epsg25833"),
        "target_crs": (TARGET_CRS_TEXT),
        "resolution_metres": (TARGET_RESOLUTION_METRES),
        "visible_years": (VISIBLE_YEARS),
        "s2_band_indexes": (S2_BAND_INDEXES),
        "clearcut_threshold": (CLEARCUT_PROBABILITY_THRESHOLD),
        "urban_threshold": (URBAN_PROBABILITY_THRESHOLD),
        "change_vs_nochange_margin": (CHANGE_VS_NOCHANGE_MARGIN),
        "task_count": (len(tasks)),
        "completed_tiles_total": (len(completed)),
        "completed_this_run": (run_completed),
        "skipped_already_completed": (run_skipped_completed),
        "skipped_no_valid_years": (run_skipped_no_valid),
        "total_seconds": (total_seconds),
    }

    RUN_SUMMARY_JSON.write_text(
        json.dumps(
            run_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    logging.info(
        "PER-TILE PREDICTION COMPLETE | "
        "%.1fs | completed_this_run=%d | "
        "completed_total=%d | "
        "skipped_completed=%d | "
        "skipped_no_valid=%d",
        total_seconds,
        run_completed,
        len(completed),
        run_skipped_completed,
        run_skipped_no_valid,
    )


if __name__ == "__main__":
    main()
