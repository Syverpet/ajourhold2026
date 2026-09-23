#!/usr/bin/env python3
"""
Fast EPSG:25833 AlphaEarth + Sentinel-2 Random Forest workflow.

This is a complete replacement for the previous filename-UTM training script.

Coregistration rule
-------------------
Every training tile is processed independently, but every raster, vector mask,
sample coordinate and spatial-CV group uses one common target CRS:

    EPSG:25833 — ETRS89 / UTM zone 33N

Sentinel-2 source handling:

    2018:
        the source raster's stored CRS is used. These files are expected to be
        stored in the UTM zone that matches their MGRS tile.

    2019-2025:
        the source coordinates are interpreted as EPSG:25833 regardless of the
        MGRS zone contained in the filename.

For every Sentinel-2 tile, the 2019 raster defines one fixed, north-up, snapped
10 m target grid in EPSG:25833. The 2018 raster is reprojected to that grid.
All later S2 years are aligned to the same fixed grid.

AlphaEarth handling:

    AlphaEarth source mosaics remain in their source UTM zones. The script does
    not use the S2 filename to select an AE UTM zone. Instead, it builds a
    spatial catalog of the AE mosaics, tests the overlapping candidates for
    each S2 tile, and chooses the source zone that produces valid AE pixels on
    the fixed EPSG:25833 tile grid. The selected AE zone is then used for every
    year of that tile and all AE rasters are warped to the S2 grid.

Sentinel-2 bands
----------------
The S2 stack uses one fixed raster-index mapping for every file:

    B2 -> raster index 1
    B3 -> raster index 2
    B4 -> raster index 3
    B8 -> raster index 7

No band-description or tag detection is performed. Raster index 8 is not used.
The script only checks that each S2 raster contains at least seven bands.

Combined feature vector
-----------------------
For visible year Y:

    [AE Y-1, AE Y, AE Y - AE Y-1]
    + [S2 Y-1 B2/B3/B4/B8,
       S2 Y B2/B3/B4/B8,
       S2 Y - S2 Y-1 for B2/B3/B4/B8]
    + [Y]

For a 64-band AlphaEarth mosaic:

    64 * 3 + 4 * 3 + 1 = 205 features

Training organisation
---------------------
Sampling, masks, diagnostics and cache writing remain per Sentinel-2 tile and
visible year. All cache parts are then pooled into one national Random Forest
model in EPSG:25833.

Sixteen Sentinel-2 tiles confirmed not to overlap the actual AOI are excluded
completely before task construction. Coverage-only diagnostic failures for any
remaining tile/year are logged and skipped rather than terminating the complete
national run. Unexpected programming, file-access and model errors still raise.

Strict diagnostics
------------------
AlphaEarth source-zone selection scans the actual spatial intersection between
each AE candidate and the S2 tile. It does not probe only the centre or fixed
fractions of the complete S2 tile. A coarse scan is followed by an exact
blockwise scan whenever the coarse scan finds no valid AE pixels.

Before sampling each tile/year pair, the script performs a full-tile,
downsampled mask scan for:

    AE pre-year
    AE post-year
    S2 pre-year
    S2 post-year

The scan covers the entire tile grid and uses nearest-neighbour mask resampling
with an exact fallback when a coarse scan reports zero coverage. It checks all
64 AE bands and all four selected S2 bands. Invalid coverage is recorded and the
affected tile/year is skipped instead of terminating the national run.

During actual sample extraction, the script also counts source-specific and
combined valid selected rows. If selected training samples exist but none survive
the raster validity checks, that tile/year is logged and skipped. Per-tile/year
JSON diagnostics and national CSV summaries are written to the run folder.

Class codes
-----------
    0 = clearcut
    1 = urban
    2 = nochange

Negative sources
----------------
    1  = manually annotated nochange
    11 = AR5 built-up
    12 = AR5 built-up
    21 = AR5 agriculture
    22 = AR5 agriculture
    81 = AR5 water
    82 = AR5 water
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import gc
import json
import logging
import math
import random
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold


# =============================================================================
# USER SETTINGS
# =============================================================================

RUN_VECTOR_PREPARATION = True
RUN_SAMPLING = True
RUN_TRAINING = True

REBUILD_ZONE_VECTOR_CACHE = False
REBUILD_SAMPLING_CACHE = False
REBUILD_TILE_MASK_CACHE = False

SKIP_COMPLETED_TILE_YEARS = True

ALPHA_ROOT = Path(r"F:\Data\AE\data\data")
S2_ROOT = Path(r"F:\Data\S2_kartverket")

TRAINING_POLYGON_DIR = Path(r"F:\Data\Ajourhold 2026\treningsdata")

TRAINING_GPKG_FILES = [
    "batches123_zone32_FINAL_june2026.gpkg",
    "batches123_zone33_FINAL_june2026.gpkg",
    "batches123_zone34_FINAL_june2026.gpkg",
    "batches123_zone35_FINAL_june2026.gpkg",
]

AR5_GDB_PATH = Path(
    r"F:\Data\AR5\Basisdata_0000_Norge_25833_FKB-AR5_FGDB"
    r"\Basisdata_0000_Norge_25833_FKB-AR5_FGDB.gdb"
)

AR5_LAYER = "fkb_ar5_omrade"
AR5_AREALTYPE_COLUMN = "arealtype"
AR5_SOURCE_CRS = "EPSG:25833"

OUTPUT_ROOT = Path(r"F:\Ajourhold 2026\ML_model_S2")

RUN_NAME = "rf_alphaearth_s2_national_epsg25833_fixed_s2_bands_aoi_filtered_fast"

RUN_ROOT = OUTPUT_ROOT / RUN_NAME

VECTOR_CACHE_ROOT = RUN_ROOT / "zone_vector_cache"
TILE_MASK_CACHE_ROOT = RUN_ROOT / "tile_mask_cache"
SAMPLING_CACHE_ROOT = RUN_ROOT / "sampling_cache"

MODEL_ROOT = RUN_ROOT / "models"
METRICS_ROOT = RUN_ROOT / "metrics"
LOG_ROOT = RUN_ROOT / "logs"
INPUT_INDEX_ROOT = RUN_ROOT / "input_indexes"

S2_INDEX_CSV = INPUT_INDEX_ROOT / "s2_tile_index.csv"
ZONE_GRID_INDEX_CSV = INPUT_INDEX_ROOT / "zone_grid_index.csv"

SKIPPED_POLYGONS_CSV = METRICS_ROOT / "skipped_training_polygons.csv"
SAMPLING_MANIFEST_CSV = METRICS_ROOT / "sampling_manifest.csv"
SAMPLING_SUMMARY_CSV = METRICS_ROOT / "sampling_summary.csv"

FEATURE_SCHEMA_PATH = RUN_ROOT / "feature_schema.json"
CONFIG_PATH = RUN_ROOT / "training_config.json"

INPUT_DIAGNOSTICS_ROOT = RUN_ROOT / "input_diagnostics"

S2_SOURCE_DIAGNOSTICS_CSV = METRICS_ROOT / "s2_source_crs_band_index_diagnostics.csv"

EXCLUDED_S2_TILES_CSV = METRICS_ROOT / "excluded_s2_tiles_outside_aoi.csv"

AE_CATALOG_CSV = INPUT_INDEX_ROOT / "alphaearth_mosaic_catalog.csv"

AE_TILE_SELECTION_CSV = METRICS_ROOT / "alphaearth_tile_source_selection.csv"

INPUT_PREFLIGHT_SUMMARY_CSV = METRICS_ROOT / "input_preflight_summary.csv"


# =============================================================================
# YEARS
# =============================================================================

S2_AVAILABLE_YEARS = list(range(2018, 2026))
VISIBLE_YEARS = list(range(2019, 2026))

AE_MOSAIC_TEMPLATE = "AlphaEarth_mosaic_{year}_{zone}"
S2_FILE_EXTENSIONS = {".tif", ".tiff"}


# =============================================================================
# SENTINEL-2 BANDS
# =============================================================================

S2_BAND_NAMES = [
    "B2",
    "B3",
    "B4",
    "B8",
]

S2_BAND_FALLBACK_INDEXES = {
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8": 7,
}


# =============================================================================
# TILES EXCLUDED FROM AOI
# =============================================================================

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

EXCLUDED_S2_TILE_REASON = "outside_actual_AOI"


# =============================================================================
# COMMON TARGET GRID
# =============================================================================

TARGET_EPSG = 25833
TARGET_CRS_TEXT = "EPSG:25833"
TARGET_ZONE_LABEL = "33N"

S2_REFERENCE_YEAR = 2019
S2_TARGET_PIXEL_SIZE_METRES = 10.0

S2_RESAMPLING = Resampling.nearest
AE_RESAMPLING = Resampling.nearest

S2_2018_USE_STORED_CRS = True
S2_2019_PLUS_SOURCE_EPSG = 25833

SNAP_TARGET_GRID_TO_PIXEL_SIZE = True
WARN_ON_ANNUAL_TARGET_GRID_DIFFERENCE = True


# =============================================================================
# ALPHAEARTH SOURCE SELECTION
# =============================================================================

AE_SOURCE_ZONES = [
    "31N",
    "32N",
    "33N",
    "34N",
    "35N",
    "36N",
]

AE_SELECTION_REFERENCE_YEAR = 2018
AE_SELECTION_PROBE_BANDS = [1, 32, 64]

AE_SELECTION_COARSE_MAX_DIM = 1024
AE_SELECTION_EXACT_BLOCK_SIZE = 2048


# =============================================================================
# INPUT DIAGNOSTICS
# =============================================================================

RUN_INPUT_PREFLIGHT = True

PREFLIGHT_COARSE_MAX_DIM = 1024
PREFLIGHT_EXACT_BLOCK_SIZE = 512

# Coverage problems do NOT terminate the entire national run.
STRICT_ABORT_IF_SOURCE_ALL_NODATA = False
STRICT_ABORT_IF_COMBINED_ALL_NODATA = False
STRICT_ABORT_IF_SELECTED_SAMPLES_ALL_INVALID = False

# Instead, the affected tile/year is excluded from sampling.
SKIP_TILE_YEAR_IF_PREFLIGHT_INVALID = True
SKIP_TILE_YEAR_IF_SELECTED_SAMPLES_ALL_INVALID = True


# =============================================================================
# POLYGON ATTRIBUTES AND CLASSES
# =============================================================================

CLASS_COLUMN = "chg_type"
YEAR_COLUMN = "year"

CLASS_MAP = {
    "clearcut": "clearcut",
    "clear cut": "clearcut",
    "clear-cut": "clearcut",
    "clear_cut": "clearcut",
    "hogst": "clearcut",
    "høgst": "clearcut",
    "flatehogst": "clearcut",
    "urban": "urban",
    "urban expansion": "urban",
    "urban_expansion": "urban",
    "urban-expansion": "urban",
    "gray area": "urban",
    "gray_area": "urban",
    "grey area": "urban",
    "grey_area": "urban",
    "nedbygging": "urban",
    "utbygging": "urban",
    "land take": "urban",
    "land_take": "urban",
    "neg": "nochange",
    "negative": "nochange",
    "hard negative": "nochange",
    "hard_negative": "nochange",
    "hard-negative": "nochange",
    "nochange": "nochange",
    "no change": "nochange",
    "no_change": "nochange",
    "unchanged": "nochange",
    "stable": "nochange",
}

IGNORE_CLASS_VALUES = {
    "",
    "ignore",
    "ignored",
    "skip",
    "exclude",
    "other",
    "others",
}


# =============================================================================
# AR5 NEGATIVES
# =============================================================================

AR5_LAND_CLASSES = [
    11,
    12,
    21,
    22,
]

AR5_WATER_CLASSES = [
    81,
    82,
]

AR5_TARGET_CLASSES = AR5_LAND_CLASSES + AR5_WATER_CLASSES

AR5_LAND_NEGATIVE_TO_POSITIVE_RATIO = 10.0

MIN_AR5_LAND_NEGATIVES_PER_TILE_YEAR = 5000
MAX_AR5_LAND_NEGATIVES_PER_TILE_YEAR = 250000

MANUAL_NOCHANGE_TO_AR5_LAND_RATIO = 2.0
WATER_TO_MANUAL_NOCHANGE_RATIO = 0.5

USE_ALL_POSITIVE_PIXELS = True

RASTERIZE_ALL_TOUCHED = False
AR5_INTERIOR_BUFFER_METRES = 0.0


# =============================================================================
# PERFORMANCE
# =============================================================================

SAMPLING_WORKERS = 16

GDAL_WARP_THREADS_PER_WORKER = 4
GDAL_CACHE_MB_PER_WORKER = 8192
GDAL_WARP_MEMORY_MB_PER_WORKER = 2048

RASTER_READ_BLOCK_SIZE = 1024
MASK_SELECTION_STRIPE_ROWS = 2048

CACHE_PART_MAX_ROWS = 250000


# =============================================================================
# SPATIAL CROSS-VALIDATION
# =============================================================================

SPATIAL_GROUP_CRS = "EPSG:25833"
SPATIAL_GROUP_SIZE_METRES = 20000

SPATIAL_GROUP_ID_OFFSET = 100000
SPATIAL_GROUP_ID_MULTIPLIER = 1000000

N_CV_FOLDS = 5
RANDOM_SEED = 42


# =============================================================================
# RF TRAINING
# =============================================================================

TRAIN_NATIONAL_MODEL = True

FEATURE_BATCH_MEMORY_MB = 2048
METADATA_SCAN_ROWS = 250000
PREDICTION_ROWS = 250000

POSITIVE_FRACTION_PER_BATCH = 0.1
CLEARCUT_SHARE_OF_POSITIVES = 0.5

RF_N_JOBS = 16

TARGET_TREES = 200
MAX_TREES_PER_BATCH = 16

RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_FEATURES = "sqrt"

RF_BOOTSTRAP = True
RF_CLASS_WEIGHT = None
RF_VERBOSE = 0

RUN_CROSS_VALIDATION = True
TRAIN_FINAL_MODEL = True

SAVE_EACH_FOLD_MODEL = True
SAVE_BEST_FOLD_MODEL = True


# =============================================================================
# NUMERIC CODES
# =============================================================================

CLEARCUT = 0
URBAN = 1
NOCHANGE = 2

CLASS_NAMES = [
    "clearcut",
    "urban",
    "nochange",
]

MANUAL_CHANGE_SOURCE = 0
MANUAL_NOCHANGE_SOURCE = 1

SOURCE_NAMES = {
    0: "manual_change",
    1: "manual_nochange",
    11: "ar5_built_up_11",
    12: "ar5_built_up_12",
    21: "ar5_agriculture_21",
    22: "ar5_agriculture_22",
    81: "ar5_water_81",
    82: "ar5_water_82",
}

FEATURE_DTYPE = np.float32
LABEL_DTYPE = np.uint8
SOURCE_DTYPE = np.uint8
YEAR_DTYPE = np.uint16
GROUP_DTYPE = np.int64

REFERENCE_DTYPE = np.dtype(
    [
        ("part", np.int32),
        ("row", np.int32),
    ]
)


@dataclass(frozen=True)
class GridDefinition:
    crs_wkt: str
    transform_values: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    width: int
    height: int
    bounds_values: tuple[
        float,
        float,
        float,
        float,
    ]

    @property
    def crs(self) -> CRS:
        return CRS.from_wkt(self.crs_wkt)

    @property
    def transform(self):
        return rasterio.Affine(*self.transform_values)

    @property
    def bounds(self):
        return rasterio.coords.BoundingBox(*self.bounds_values)


@dataclass(frozen=True)
class CachePart:
    part_id: int
    x_path: Path
    y_path: Path
    groups_path: Path
    sources_path: Path
    years_path: Path
    tile_id: str
    zone: str
    visible_year: int
    rows: int
    features: int


_WORKER_ZONE_POLYGONS: dict[
    str,
    gpd.GeoDataFrame,
] = {}

_WORKER_GROUP_TRANSFORMERS: dict[
    str,
    Transformer,
] = {}


def setup_directories() -> None:
    folders = [
        OUTPUT_ROOT,
        RUN_ROOT,
        VECTOR_CACHE_ROOT,
        TILE_MASK_CACHE_ROOT,
        SAMPLING_CACHE_ROOT,
        MODEL_ROOT,
        METRICS_ROOT,
        LOG_ROOT,
        INPUT_INDEX_ROOT,
        INPUT_DIAGNOSTICS_ROOT,
    ]

    for folder in folders:
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def setup_logging() -> None:
    log_path = (
        LOG_ROOT / "train_rf_alphaearth_s2_epsg25833_fixed_bands_aoi_filtered.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)s | %(message)s"),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_path,
                encoding="utf-8",
            ),
        ],
    )


def save_configuration() -> None:
    config = {
        "ALPHA_ROOT": str(ALPHA_ROOT),
        "S2_ROOT": str(S2_ROOT),
        "TRAINING_POLYGON_DIR": str(TRAINING_POLYGON_DIR),
        "TRAINING_GPKG_FILES": (TRAINING_GPKG_FILES),
        "AR5_GDB_PATH": str(AR5_GDB_PATH),
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "RUN_NAME": RUN_NAME,
        "RUN_VECTOR_PREPARATION": (RUN_VECTOR_PREPARATION),
        "RUN_SAMPLING": (RUN_SAMPLING),
        "RUN_TRAINING": (RUN_TRAINING),
        "REBUILD_ZONE_VECTOR_CACHE": (REBUILD_ZONE_VECTOR_CACHE),
        "REBUILD_SAMPLING_CACHE": (REBUILD_SAMPLING_CACHE),
        "REBUILD_TILE_MASK_CACHE": (REBUILD_TILE_MASK_CACHE),
        "VISIBLE_YEARS": (VISIBLE_YEARS),
        "S2_BAND_NAMES": (S2_BAND_NAMES),
        "S2_BAND_FALLBACK_INDEXES": (S2_BAND_FALLBACK_INDEXES),
        "EXCLUDED_S2_TILE_IDS": sorted(EXCLUDED_S2_TILE_IDS),
        "EXCLUDED_S2_TILE_REASON": (EXCLUDED_S2_TILE_REASON),
        "S2_BAND_SELECTION": ("fixed raster indexes [1,2,3,7]"),
        "TARGET_CRS": (TARGET_CRS_TEXT),
        "TARGET_ZONE_LABEL": (TARGET_ZONE_LABEL),
        "S2_REFERENCE_YEAR": (S2_REFERENCE_YEAR),
        "S2_2018_SOURCE_RULE": ("use stored CRS"),
        "S2_2019_PLUS_SOURCE_EPSG": (S2_2019_PLUS_SOURCE_EPSG),
        "S2_TARGET_PIXEL_SIZE_METRES": (S2_TARGET_PIXEL_SIZE_METRES),
        "S2_RESAMPLING": str(S2_RESAMPLING),
        "AE_RESAMPLING": str(AE_RESAMPLING),
        "AE_SOURCE_ZONES": (AE_SOURCE_ZONES),
        "AE_SELECTION_REFERENCE_YEAR": (AE_SELECTION_REFERENCE_YEAR),
        "RUN_INPUT_PREFLIGHT": (RUN_INPUT_PREFLIGHT),
        "AE_SELECTION_COARSE_MAX_DIM": (AE_SELECTION_COARSE_MAX_DIM),
        "AE_SELECTION_EXACT_BLOCK_SIZE": (AE_SELECTION_EXACT_BLOCK_SIZE),
        "PREFLIGHT_COARSE_MAX_DIM": (PREFLIGHT_COARSE_MAX_DIM),
        "PREFLIGHT_EXACT_BLOCK_SIZE": (PREFLIGHT_EXACT_BLOCK_SIZE),
        "STRICT_ABORT_IF_SOURCE_ALL_NODATA": (STRICT_ABORT_IF_SOURCE_ALL_NODATA),
        "STRICT_ABORT_IF_COMBINED_ALL_NODATA": (STRICT_ABORT_IF_COMBINED_ALL_NODATA),
        "STRICT_ABORT_IF_SELECTED_SAMPLES_ALL_INVALID": (
            STRICT_ABORT_IF_SELECTED_SAMPLES_ALL_INVALID
        ),
        "SKIP_TILE_YEAR_IF_PREFLIGHT_INVALID": (SKIP_TILE_YEAR_IF_PREFLIGHT_INVALID),
        "SKIP_TILE_YEAR_IF_SELECTED_SAMPLES_ALL_INVALID": (
            SKIP_TILE_YEAR_IF_SELECTED_SAMPLES_ALL_INVALID
        ),
        "AR5_LAND_CLASSES": (AR5_LAND_CLASSES),
        "AR5_WATER_CLASSES": (AR5_WATER_CLASSES),
        "AR5_LAND_NEGATIVE_TO_POSITIVE_RATIO": (AR5_LAND_NEGATIVE_TO_POSITIVE_RATIO),
        "MANUAL_NOCHANGE_TO_AR5_LAND_RATIO": (MANUAL_NOCHANGE_TO_AR5_LAND_RATIO),
        "WATER_TO_MANUAL_NOCHANGE_RATIO": (WATER_TO_MANUAL_NOCHANGE_RATIO),
        "SAMPLING_WORKERS": (SAMPLING_WORKERS),
        "GDAL_WARP_THREADS_PER_WORKER": (GDAL_WARP_THREADS_PER_WORKER),
        "GDAL_CACHE_MB_PER_WORKER": (GDAL_CACHE_MB_PER_WORKER),
        "GDAL_WARP_MEMORY_MB_PER_WORKER": (GDAL_WARP_MEMORY_MB_PER_WORKER),
        "RASTER_READ_BLOCK_SIZE": (RASTER_READ_BLOCK_SIZE),
        "TRAIN_NATIONAL_MODEL": (TRAIN_NATIONAL_MODEL),
        "SPATIAL_GROUP_CRS": (SPATIAL_GROUP_CRS),
        "SPATIAL_GROUP_SIZE_METRES": (SPATIAL_GROUP_SIZE_METRES),
        "FEATURE_BATCH_MEMORY_MB": (FEATURE_BATCH_MEMORY_MB),
        "RF_N_JOBS": (RF_N_JOBS),
        "TARGET_TREES": (TARGET_TREES),
        "RANDOM_SEED": (RANDOM_SEED),
    }

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_feature_schema() -> None:
    schema = {
        "reference_grid": (
            "One fixed 10 m EPSG:25833 grid "
            "per S2 tile, derived from the "
            "2019 Sentinel-2 raster."
        ),
        "s2_source_crs": {
            "2018": ("stored native CRS"),
            "2019_2025": ("forced EPSG:25833"),
        },
        "alphaearth_source_selection": (
            "Spatial overlap plus valid-pixel "
            "probing; no filename-derived "
            "UTM-zone selection."
        ),
        "spatial_cross_validation": (
            "Pixel centres are already in "
            "EPSG:25833 and are assigned "
            "to fixed 20 km cells."
        ),
        "feature_order": [
            "ae_pre_all_bands",
            "ae_post_all_bands",
            "ae_post_minus_pre_all_bands",
            "s2_pre_fixed_raster_indexes_1_2_3_7",
            "s2_post_fixed_raster_indexes_1_2_3_7",
            "s2_post_minus_pre_fixed_raster_indexes_1_2_3_7",
            "visible_year",
        ],
        "s2_band_names": (S2_BAND_NAMES),
        "s2_band_selection": ("fixed raster indexes"),
        "s2_raster_indexes": (S2_BAND_FALLBACK_INDEXES),
        "s2_index_order": [
            1,
            2,
            3,
            7,
        ],
        "expected_features_for_64_band_ae": 205,
        "class_codes": {
            "clearcut": CLEARCUT,
            "urban": URBAN,
            "nochange": NOCHANGE,
        },
        "source_codes": (SOURCE_NAMES),
    }

    FEATURE_SCHEMA_PATH.write_text(
        json.dumps(
            schema,
            indent=2,
        ),
        encoding="utf-8",
    )


def clean_class_value(
    value: object,
) -> str:
    if value is None or pd.isna(value):
        return ""

    cleaned = str(value).strip().lower()

    return re.sub(
        r"\s+",
        " ",
        cleaned,
    )


def class_candidates(
    value: object,
) -> list[str]:
    key = clean_class_value(value)

    return list(
        {
            key,
            key.replace("_", " "),
            key.replace("-", " "),
            key.replace(" ", "_"),
            key.replace(" ", "-"),
        }
    )


def normalise_class(
    value: object,
) -> str | None:
    candidates = class_candidates(value)

    if any(candidate in IGNORE_CLASS_VALUES for candidate in candidates):
        return None

    for candidate in candidates:
        if candidate in CLASS_MAP:
            return CLASS_MAP[candidate]

    return None


def read_training_polygons() -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []

    skipped: list[dict[str, Any]] = []

    for filename in TRAINING_GPKG_FILES:
        path = TRAINING_POLYGON_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing training file: {path}")

        logging.info(
            "Reading training polygons: %s",
            path,
        )

        gdf = gpd.read_file(path)

        if gdf.crs is None:
            raise ValueError(f"Training file has no CRS: {path}")

        if CLASS_COLUMN not in gdf.columns:
            raise ValueError(f"{path.name} does not contain {CLASS_COLUMN!r}.")

        if YEAR_COLUMN not in gdf.columns:
            raise ValueError(f"{path.name} does not contain {YEAR_COLUMN!r}.")

        gdf = gdf.copy()

        gdf["source_file"] = path.name

        gdf["source_row"] = np.arange(
            len(gdf),
            dtype=np.int64,
        )

        gdf["polygon_id"] = [f"{path.stem}_{index}" for index in range(len(gdf))]

        gdf["class_norm"] = gdf[CLASS_COLUMN].apply(normalise_class)

        gdf["visible_year"] = pd.to_numeric(
            gdf[YEAR_COLUMN],
            errors="coerce",
        )

        invalid_geometry = gdf.geometry.isna() | gdf.geometry.is_empty

        unknown_class = gdf["class_norm"].isna()

        event_mask = gdf["class_norm"].isin(
            [
                "clearcut",
                "urban",
            ]
        )

        invalid_event_year = event_mask & (
            gdf["visible_year"].isna() | ~gdf["visible_year"].isin(VISIBLE_YEARS)
        )

        skip_mask = invalid_geometry | unknown_class | invalid_event_year

        for _, row in gdf.loc[skip_mask].iterrows():
            reasons: list[str] = []

            if row.geometry is None or row.geometry.is_empty:
                reasons.append("missing_or_empty_geometry")

            if pd.isna(row["class_norm"]):
                reasons.append("unknown_or_ignored_class")

            if row["class_norm"] in {
                "clearcut",
                "urban",
            } and (
                pd.isna(row["visible_year"])
                or int(row["visible_year"]) not in VISIBLE_YEARS
            ):
                if (
                    not pd.isna(row["visible_year"])
                    and int(row["visible_year"]) == 2018
                ):
                    reasons.append("2018_event_requires_missing_2017_sentinel2")
                else:
                    reasons.append("invalid_event_year")

            skipped.append(
                {
                    "source_file": (row["source_file"]),
                    "source_row": (row["source_row"]),
                    "raw_class": (row[CLASS_COLUMN]),
                    "raw_year": (row[YEAR_COLUMN]),
                    "reason": ";".join(reasons),
                }
            )

        gdf = gdf.loc[~skip_mask].copy()

        invalid = ~gdf.geometry.is_valid

        if invalid.any():
            gdf.loc[
                invalid,
                "geometry",
            ] = gdf.loc[
                invalid,
                "geometry",
            ].buffer(0)

        gdf = gdf[
            gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid
        ].copy()

        gdf["visible_year"] = gdf["visible_year"].astype("Int64")

        frames.append(gdf)

    if skipped:
        pd.DataFrame(skipped).to_csv(
            SKIPPED_POLYGONS_CSV,
            index=False,
        )

    if not frames:
        raise RuntimeError("No training polygons were loaded.")

    polygons = gpd.GeoDataFrame(
        pd.concat(
            [frame.to_crs(AR5_SOURCE_CRS) for frame in frames],
            ignore_index=True,
        ),
        crs=AR5_SOURCE_CRS,
    )

    logging.info(
        "Usable polygons: %d\n%s",
        len(polygons),
        polygons["class_norm"].value_counts(),
    )

    return polygons


def tile_id_from_path(
    path: Path,
) -> str:
    tile_id = path.stem.split("_")[-1].strip().upper()

    if not tile_id:
        raise ValueError(f"Could not derive tile ID from {path}")

    return tile_id


def zone_from_tile_id(
    tile_id: str,
) -> str:
    match = re.match(
        r"^T?(\d{2})",
        tile_id.upper(),
    )

    if match is None:
        raise ValueError(f"Could not derive UTM zone from tile ID {tile_id!r}")

    return f"{match.group(1)}N"


def target_crs() -> CRS:
    return CRS.from_epsg(TARGET_EPSG)


def effective_s2_source_crs(
    src,
    source_year: int,
) -> CRS:
    if source_year == 2018:
        if not S2_2018_USE_STORED_CRS:
            raise RuntimeError("No alternative 2018 S2 source-CRS rule is configured.")

        if src.crs is None:
            raise ValueError(f"2018 Sentinel-2 file has no CRS: {src.name}")

        return CRS.from_user_input(src.crs)

    if source_year >= 2019:
        return CRS.from_epsg(S2_2019_PLUS_SOURCE_EPSG)

    raise ValueError(f"No Sentinel-2 source-CRS rule for year {source_year}.")


def log_s2_source_crs_status(
    *,
    tile_id: str,
    source_year: int,
    src,
    raster_path: Path,
) -> None:
    effective_crs = effective_s2_source_crs(
        src,
        source_year,
    )

    logging.info(
        "S2 CRS | tile=%s year=%d path=%s | "
        "stored_crs=%s stored_epsg=%s | "
        "effective_source_crs=%s | "
        "target_crs=%s | bounds=%s "
        "resolution=%s size=%dx%d nodata=%s",
        tile_id,
        source_year,
        raster_path,
        src.crs,
        (src.crs.to_epsg() if src.crs is not None else None),
        effective_crs.to_string(),
        TARGET_CRS_TEXT,
        tuple(src.bounds),
        tuple(src.res),
        src.width,
        src.height,
        src.nodata,
    )


def derive_epsg25833_grid(
    src,
    source_year: int,
) -> GridDefinition:
    source_crs = effective_s2_source_crs(
        src,
        source_year,
    )

    destination_crs = target_crs()

    resolution = float(S2_TARGET_PIXEL_SIZE_METRES)

    if source_crs == destination_crs:
        (
            left,
            bottom,
            right,
            top,
        ) = tuple(src.bounds)
    else:
        (
            left,
            bottom,
            right,
            top,
        ) = transform_bounds(
            source_crs,
            destination_crs,
            *src.bounds,
            densify_pts=41,
        )

    if SNAP_TARGET_GRID_TO_PIXEL_SIZE:
        left = math.floor(left / resolution) * resolution

        bottom = math.floor(bottom / resolution) * resolution

        right = math.ceil(right / resolution) * resolution

        top = math.ceil(top / resolution) * resolution

    width = int(math.ceil((right - left) / resolution))

    height = int(math.ceil((top - bottom) / resolution))

    transform = from_origin(
        left,
        top,
        resolution,
        resolution,
    )

    exact_right = left + width * resolution

    exact_bottom = top - height * resolution

    return GridDefinition(
        crs_wkt=(destination_crs.to_wkt()),
        transform_values=(tuple(transform)[:6]),
        width=width,
        height=height,
        bounds_values=(
            left,
            exact_bottom,
            exact_right,
            top,
        ),
    )


def intersection_area(
    first: tuple[
        float,
        float,
        float,
        float,
    ],
    second: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    left = max(
        first[0],
        second[0],
    )

    bottom = max(
        first[1],
        second[1],
    )

    right = min(
        first[2],
        second[2],
    )

    top = min(
        first[3],
        second[3],
    )

    if right <= left or top <= bottom:
        return 0.0

    return float((right - left) * (top - bottom))


def build_s2_index() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    diagnostics: list[dict[str, Any]] = []

    excluded_rows: list[dict[str, Any]] = []

    for year in S2_AVAILABLE_YEARS:
        year_folder = S2_ROOT / str(year)

        if not year_folder.exists():
            raise FileNotFoundError(f"Missing Sentinel-2 year folder: {year_folder}")

        paths = sorted(
            path
            for path in year_folder.rglob("*")
            if (path.is_file() and path.suffix.lower() in S2_FILE_EXTENSIONS)
        )

        if not paths:
            raise RuntimeError(f"No Sentinel-2 TIFFs found in {year_folder}")

        seen: dict[
            str,
            Path,
        ] = {}

        excluded_this_year = 0

        for path in paths:
            tile_id = tile_id_from_path(path)

            if tile_id in EXCLUDED_S2_TILE_IDS:
                excluded_this_year += 1

                excluded_rows.append(
                    {
                        "year": year,
                        "tile_id": tile_id,
                        "path": str(path),
                        "reason": (EXCLUDED_S2_TILE_REASON),
                    }
                )

                continue

            if tile_id in seen:
                raise RuntimeError(
                    f"Duplicate S2 tile {tile_id} for {year}:\n{seen[tile_id]}\n{path}"
                )

            seen[tile_id] = path

            with rasterio.open(path) as src:
                selected_indexes = detect_s2_band_indexes(src)

                effective_crs = effective_s2_source_crs(
                    src,
                    year,
                )

                diagnostics.append(
                    {
                        "year": year,
                        "tile_id": tile_id,
                        "path": str(path),
                        "stored_crs": str(src.crs),
                        "stored_epsg": (
                            src.crs.to_epsg() if src.crs is not None else None
                        ),
                        "effective_source_crs": (effective_crs.to_string()),
                        "target_crs": (TARGET_CRS_TEXT),
                        "band_count": (src.count),
                        "selected_indexes": (
                            ",".join(str(value) for value in selected_indexes)
                        ),
                        "nodata": (src.nodata),
                        "dtype": ",".join(sorted(set(src.dtypes))),
                        "bounds": str(tuple(src.bounds)),
                        "resolution": str(tuple(src.res)),
                        "width": (src.width),
                        "height": (src.height),
                    }
                )

            rows.append(
                {
                    "year": year,
                    "tile_id": tile_id,
                    "zone": (TARGET_ZONE_LABEL),
                    "filename_zone": (zone_from_tile_id(tile_id)),
                    "path": str(path),
                }
            )

        logging.info(
            "Indexed %d Sentinel-2 tiles for %d; excluded %d known outside-AOI tiles.",
            len(seen),
            year,
            excluded_this_year,
        )

    index = pd.DataFrame(rows)

    index.to_csv(
        S2_INDEX_CSV,
        index=False,
    )

    pd.DataFrame(diagnostics).to_csv(
        S2_SOURCE_DIAGNOSTICS_CSV,
        index=False,
    )

    logging.info(
        "Wrote S2 CRS/band diagnostics: %s",
        S2_SOURCE_DIAGNOSTICS_CSV,
    )

    if excluded_rows:
        excluded_frame = pd.DataFrame(excluded_rows)

        excluded_frame.to_csv(
            EXCLUDED_S2_TILES_CSV,
            index=False,
        )

        excluded_unique = sorted(excluded_frame["tile_id"].unique())

        logging.info(
            "Excluded %d unique S2 tiles outside AOI: %s",
            len(excluded_unique),
            excluded_unique,
        )

        missing_requested_exclusions = sorted(
            EXCLUDED_S2_TILE_IDS - set(excluded_unique)
        )

        if missing_requested_exclusions:
            logging.warning(
                "Configured excluded tile IDs were not found in the S2 index: %s",
                missing_requested_exclusions,
            )

        logging.info(
            "Wrote excluded-tile table: %s",
            EXCLUDED_S2_TILES_CSV,
        )

    return index


def grid_from_dataset(
    src,
) -> GridDefinition:
    return GridDefinition(
        crs_wkt=(src.crs.to_wkt()),
        transform_values=(tuple(src.transform)[:6]),
        width=src.width,
        height=src.height,
        bounds_values=(tuple(src.bounds)),
    )


def grids_match(
    first: GridDefinition,
    second: GridDefinition,
) -> bool:
    return (
        first.crs_wkt == second.crs_wkt
        and first.transform_values == second.transform_values
        and first.width == second.width
        and first.height == second.height
    )


def build_tile_tasks(
    s2_index: pd.DataFrame,
    ae_catalog: pd.DataFrame,
) -> list[dict[str, Any]]:
    lookup = {
        (
            int(row.year),
            str(row.tile_id),
        ): Path(row.path)
        for row in s2_index.itertuples()
    }

    tasks: list[dict[str, Any]] = []

    ae_selection_rows: list[dict[str, Any]] = []

    for tile_id in sorted(s2_index["tile_id"].unique()):
        zone = TARGET_ZONE_LABEL

        tile_rows = s2_index[s2_index["tile_id"] == tile_id].sort_values("year")

        if tile_rows.empty:
            continue

        anchor_rows = tile_rows[tile_rows["year"] == S2_REFERENCE_YEAR]

        if anchor_rows.empty:
            raise FileNotFoundError(
                f"Tile {tile_id} has no "
                f"{S2_REFERENCE_YEAR} "
                "Sentinel-2 raster for "
                "the fixed EPSG:25833 grid."
            )

        anchor_path = Path(anchor_rows.iloc[0]["path"])

        with rasterio.open(anchor_path) as anchor_src:
            log_s2_source_crs_status(
                tile_id=tile_id,
                source_year=(S2_REFERENCE_YEAR),
                src=anchor_src,
                raster_path=anchor_path,
            )

            reference_grid = derive_epsg25833_grid(
                anchor_src,
                S2_REFERENCE_YEAR,
            )

        (
            ae_source_zone,
            ae_selection_diagnostics,
        ) = select_ae_source_zone_for_tile(
            tile_id=tile_id,
            grid=reference_grid,
            ae_catalog=ae_catalog,
        )

        ae_selection_rows.extend(ae_selection_diagnostics)

        pairs: list[dict[str, Any]] = []

        for visible_year in VISIBLE_YEARS:
            pre_path = lookup.get(
                (
                    visible_year - 1,
                    tile_id,
                )
            )

            post_path = lookup.get(
                (
                    visible_year,
                    tile_id,
                )
            )

            if pre_path is None or post_path is None:
                continue

            for (
                source_year,
                source_path,
            ) in [
                (
                    visible_year - 1,
                    pre_path,
                ),
                (
                    visible_year,
                    post_path,
                ),
            ]:
                with rasterio.open(source_path) as source_src:
                    log_s2_source_crs_status(
                        tile_id=tile_id,
                        source_year=source_year,
                        src=source_src,
                        raster_path=source_path,
                    )

                    if WARN_ON_ANNUAL_TARGET_GRID_DIFFERENCE:
                        candidate_grid = derive_epsg25833_grid(
                            source_src,
                            source_year,
                        )

                        if not grids_match(
                            reference_grid,
                            candidate_grid,
                        ):
                            logging.warning(
                                "Annual S2 footprint differs "
                                "from fixed EPSG:25833 anchor "
                                "grid: tile=%s source_year=%d. "
                                "Source will be warped to the "
                                "anchor grid.",
                                tile_id,
                                source_year,
                            )

            ae_pre_path = find_ae_mosaic(
                visible_year - 1,
                ae_source_zone,
            )

            ae_post_path = find_ae_mosaic(
                visible_year,
                ae_source_zone,
            )

            pairs.append(
                {
                    "visible_year": (visible_year),
                    "s2_pre_year": (visible_year - 1),
                    "s2_post_year": (visible_year),
                    "s2_pre_path": str(pre_path),
                    "s2_post_path": str(post_path),
                    "ae_source_zone": (ae_source_zone),
                    "ae_pre_path": str(ae_pre_path),
                    "ae_post_path": str(ae_post_path),
                }
            )

        if pairs:
            tasks.append(
                {
                    "tile_id": tile_id,
                    "zone": zone,
                    "pairs": pairs,
                    "ae_source_zone": (ae_source_zone),
                    "reference_grid": {
                        "crs_wkt": (reference_grid.crs_wkt),
                        "transform_values": (reference_grid.transform_values),
                        "width": (reference_grid.width),
                        "height": (reference_grid.height),
                        "bounds_values": (reference_grid.bounds_values),
                    },
                    "anchor_s2_path": str(anchor_path),
                }
            )

    pd.DataFrame(ae_selection_rows).to_csv(
        AE_TILE_SELECTION_CSV,
        index=False,
    )

    logging.info(
        "Created %d per-tile tasks on fixed EPSG:25833 grids.",
        len(tasks),
    )

    logging.info(
        "Wrote AlphaEarth tile-source selection diagnostics: %s",
        AE_TILE_SELECTION_CSV,
    )

    return tasks


def build_zone_grid_index(
    tasks: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for zone in sorted({task["zone"] for task in tasks}):
        zone_tasks = [task for task in tasks if task["zone"] == zone]

        crs_wkts = {task["reference_grid"]["crs_wkt"] for task in zone_tasks}

        if len(crs_wkts) != 1:
            raise RuntimeError(f"Zone {zone} contains multiple S2 CRSs.")

        bounds = np.array(
            [task["reference_grid"]["bounds_values"] for task in zone_tasks],
            dtype=np.float64,
        )

        rows.append(
            {
                "zone": zone,
                "crs_wkt": next(iter(crs_wkts)),
                "minx": float(
                    bounds[
                        :,
                        0,
                    ].min()
                ),
                "miny": float(
                    bounds[
                        :,
                        1,
                    ].min()
                ),
                "maxx": float(
                    bounds[
                        :,
                        2,
                    ].max()
                ),
                "maxy": float(
                    bounds[
                        :,
                        3,
                    ].max()
                ),
                "tile_count": (len(zone_tasks)),
            }
        )

    table = pd.DataFrame(rows)

    table.to_csv(
        ZONE_GRID_INDEX_CSV,
        index=False,
    )

    return table


def zone_cache_paths(
    zone: str,
) -> dict[str, Path]:
    root = VECTOR_CACHE_ROOT / zone

    return {
        "root": root,
        "polygons": (root / "training_polygons.joblib"),
        "ar5": (root / "ar5_target_classes.fgb"),
        "metadata": (root / "metadata.json"),
        "empty_ar5": (root / "_EMPTY_AR5.json"),
    }


def read_ar5_from_filegdb(
    bounds_in_ar5_crs: tuple[
        float,
        float,
        float,
        float,
    ],
) -> gpd.GeoDataFrame:
    try:
        ar5 = gpd.read_file(
            AR5_GDB_PATH,
            layer=AR5_LAYER,
            bbox=bounds_in_ar5_crs,
            columns=[AR5_AREALTYPE_COLUMN],
            engine="pyogrio",
            use_arrow=True,
        )

    except Exception:
        ar5 = gpd.read_file(
            AR5_GDB_PATH,
            layer=AR5_LAYER,
            bbox=bounds_in_ar5_crs,
        )

    if AR5_AREALTYPE_COLUMN not in ar5.columns:
        raise ValueError(f"AR5 layer is missing {AR5_AREALTYPE_COLUMN!r}.")

    if ar5.crs is None:
        ar5 = ar5.set_crs(AR5_SOURCE_CRS)

    ar5[AR5_AREALTYPE_COLUMN] = pd.to_numeric(
        ar5[AR5_AREALTYPE_COLUMN],
        errors="coerce",
    )

    ar5 = ar5[
        ar5.geometry.notna()
        & ~ar5.geometry.is_empty
        & ar5[AR5_AREALTYPE_COLUMN].isin(AR5_TARGET_CLASSES)
    ].copy()

    invalid = ~ar5.geometry.is_valid

    if invalid.any():
        ar5.loc[
            invalid,
            "geometry",
        ] = ar5.loc[
            invalid,
            "geometry",
        ].buffer(0)

    ar5 = ar5[
        ar5.geometry.notna() & ~ar5.geometry.is_empty & ar5.geometry.is_valid
    ].copy()

    return ar5


def prepare_zone_vector_caches(
    polygons_ar5_crs: gpd.GeoDataFrame,
    zone_grid_index: pd.DataFrame,
) -> None:
    if not AR5_GDB_PATH.exists():
        raise FileNotFoundError(f"AR5 FileGDB not found: {AR5_GDB_PATH}")

    for row in zone_grid_index.itertuples():
        zone = str(row.zone)

        paths = zone_cache_paths(zone)

        complete = (
            paths["polygons"].exists()
            and paths["metadata"].exists()
            and (paths["ar5"].exists() or paths["empty_ar5"].exists())
        )

        if complete and not REBUILD_ZONE_VECTOR_CACHE:
            logging.info(
                "Reusing vector cache for %s.",
                zone,
            )
            continue

        if paths["root"].exists():
            shutil.rmtree(paths["root"])

        paths["root"].mkdir(
            parents=True,
            exist_ok=True,
        )

        zone_crs = CRS.from_wkt(row.crs_wkt)

        zone_bounds = (
            float(row.minx),
            float(row.miny),
            float(row.maxx),
            float(row.maxy),
        )

        zone_box = box(*zone_bounds)

        polygons_zone = polygons_ar5_crs.to_crs(zone_crs)

        polygon_indices = polygons_zone.sindex.query(
            zone_box,
            predicate="intersects",
        )

        polygons_zone = polygons_zone.iloc[polygon_indices].copy()

        joblib.dump(
            polygons_zone,
            paths["polygons"],
            compress=3,
        )

        bounds_ar5 = transform_bounds(
            zone_crs,
            AR5_SOURCE_CRS,
            *zone_bounds,
            densify_pts=21,
        )

        logging.info(
            "Reading AR5 once for zone %s, bbox=%s",
            zone,
            bounds_ar5,
        )

        ar5 = read_ar5_from_filegdb(bounds_ar5)

        if not ar5.empty:
            ar5 = ar5.to_crs(zone_crs)

            if AR5_INTERIOR_BUFFER_METRES > 0:
                buffered = ar5.geometry.buffer(-AR5_INTERIOR_BUFFER_METRES)

                keep = ~buffered.is_empty & buffered.is_valid

                ar5 = ar5.loc[keep].copy()

                ar5.geometry = buffered.loc[keep]

            if paths["ar5"].exists():
                paths["ar5"].unlink()

            ar5.to_file(
                paths["ar5"],
                driver="FlatGeobuf",
                engine="pyogrio",
            )

        else:
            paths["empty_ar5"].write_text(
                json.dumps(
                    {
                        "zone": zone,
                        "reason": ("No target AR5 features found."),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        metadata = {
            "zone": zone,
            "crs_wkt": (zone_crs.to_wkt()),
            "bounds": (zone_bounds),
            "training_polygon_count": (len(polygons_zone)),
            "ar5_feature_count": (len(ar5)),
        }

        paths["metadata"].write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        logging.info(
            "Prepared vector cache %s: polygons=%d AR5=%d",
            zone,
            len(polygons_zone),
            len(ar5),
        )

        del polygons_zone
        del ar5

        gc.collect()


def find_ae_mosaic(
    year: int,
    zone: str,
) -> Path:
    folder = ALPHA_ROOT / str(year) / zone

    stem = AE_MOSAIC_TEMPLATE.format(
        year=year,
        zone=zone,
    )

    if not folder.exists():
        raise FileNotFoundError(f"Missing AlphaEarth folder: {folder}")

    for suffix in [
        ".tif",
        ".tiff",
    ]:
        candidate = folder / f"{stem}{suffix}"

        if candidate.exists():
            return candidate

    candidates: list[Path] = []

    for pattern in [
        f"{stem}*.tif",
        f"{stem}*.tiff",
    ]:
        candidates.extend(folder.glob(pattern))

    candidates = sorted(set(candidates))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple AE mosaics found "
            f"for {year} {zone}:\n" + "\n".join(str(path) for path in candidates)
        )

    raise FileNotFoundError(f"No AE mosaic found for {year} {zone}.")


def build_ae_catalog() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    destination_crs = target_crs()

    for year in S2_AVAILABLE_YEARS:
        for source_zone in AE_SOURCE_ZONES:
            try:
                path = find_ae_mosaic(
                    year,
                    source_zone,
                )
            except FileNotFoundError:
                continue

            with rasterio.open(path) as src:
                if src.crs is None:
                    raise ValueError(f"AlphaEarth mosaic has no CRS: {path}")

                if src.count != 64:
                    raise ValueError(f"Expected 64 AE bands, found {src.count}: {path}")

                transformed_bounds = transform_bounds(
                    src.crs,
                    destination_crs,
                    *src.bounds,
                    densify_pts=41,
                )

                rows.append(
                    {
                        "year": year,
                        "source_zone": (source_zone),
                        "path": str(path),
                        "stored_crs": str(src.crs),
                        "stored_epsg": (src.crs.to_epsg()),
                        "source_bounds": str(tuple(src.bounds)),
                        "target_bounds_minx": float(transformed_bounds[0]),
                        "target_bounds_miny": float(transformed_bounds[1]),
                        "target_bounds_maxx": float(transformed_bounds[2]),
                        "target_bounds_maxy": float(transformed_bounds[3]),
                        "width": (src.width),
                        "height": (src.height),
                        "band_count": (src.count),
                        "nodata": (src.nodata),
                        "dtype": ",".join(sorted(set(src.dtypes))),
                    }
                )

    catalog = pd.DataFrame(rows)

    required_years = set(S2_AVAILABLE_YEARS)

    catalog_years = set(catalog["year"].unique())

    missing_years = sorted(required_years - catalog_years)

    if missing_years:
        raise RuntimeError(
            f"AlphaEarth catalog is missing all mosaics for years {missing_years}."
        )

    catalog.to_csv(
        AE_CATALOG_CSV,
        index=False,
    )

    logging.info(
        "Indexed %d AlphaEarth mosaics. Catalog: %s",
        len(catalog),
        AE_CATALOG_CSV,
    )

    return catalog


def clipped_grid_window_from_bounds(
    *,
    grid: GridDefinition,
    bounds: tuple[
        float,
        float,
        float,
        float,
    ],
) -> Window:
    raw = from_bounds(
        *bounds,
        transform=grid.transform,
    )

    col_start = max(
        0,
        int(math.floor(raw.col_off)),
    )

    row_start = max(
        0,
        int(math.floor(raw.row_off)),
    )

    col_stop = min(
        grid.width,
        int(math.ceil(raw.col_off + raw.width)),
    )

    row_stop = min(
        grid.height,
        int(math.ceil(raw.row_off + raw.height)),
    )

    if col_stop <= col_start or row_stop <= row_start:
        raise ValueError(
            "Bounds do not produce a "
            "positive target-grid window: "
            f"bounds={bounds}, "
            f"grid_bounds={grid.bounds_values}"
        )

    return Window(
        col_off=col_start,
        row_off=row_start,
        width=(col_stop - col_start),
        height=(row_stop - row_start),
    )


def reduced_shape_for_window(
    *,
    window: Window,
    maximum_dimension: int,
) -> tuple[int, int]:
    source_width = max(
        1,
        int(math.ceil(window.width)),
    )

    source_height = max(
        1,
        int(math.ceil(window.height)),
    )

    scale = max(
        source_width / maximum_dimension,
        source_height / maximum_dimension,
        1.0,
    )

    output_width = max(
        1,
        int(math.ceil(source_width / scale)),
    )

    output_height = max(
        1,
        int(math.ceil(source_height / scale)),
    )

    return (
        output_height,
        output_width,
    )


def iter_subwindows(
    *,
    parent: Window,
    block_size: int,
) -> Iterator[Window]:
    row_start = int(parent.row_off)

    col_start = int(parent.col_off)

    row_stop = int(parent.row_off + parent.height)

    col_stop = int(parent.col_off + parent.width)

    for row_off in range(
        row_start,
        row_stop,
        block_size,
    ):
        height = min(
            block_size,
            row_stop - row_off,
        )

        for col_off in range(
            col_start,
            col_stop,
            block_size,
        ):
            width = min(
                block_size,
                col_stop - col_off,
            )

            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=width,
                height=height,
            )


def make_warped_vrt(
    src,
    grid: GridDefinition,
    resampling: Resampling,
    src_crs_override: CRS | None = None,
):
    vrt_kwargs = {
        "crs": grid.crs,
        "transform": grid.transform,
        "width": grid.width,
        "height": grid.height,
        "resampling": resampling,
        "dtype": "float32",
        "warp_mem_limit": (GDAL_WARP_MEMORY_MB_PER_WORKER),
        "init_dest_nodata": True,
    }

    if src_crs_override is not None:
        vrt_kwargs["src_crs"] = src_crs_override

    if src.nodata is not None:
        vrt_kwargs["src_nodata"] = src.nodata

        vrt_kwargs["nodata"] = src.nodata

    return WarpedVRT(
        src,
        **vrt_kwargs,
    )


def probe_ae_candidate(
    *,
    path: Path,
    grid: GridDefinition,
    overlap_bounds: tuple[
        float,
        float,
        float,
        float,
    ],
) -> dict[str, Any]:
    overlap_window = clipped_grid_window_from_bounds(
        grid=grid,
        bounds=overlap_bounds,
    )

    (
        coarse_height,
        coarse_width,
    ) = reduced_shape_for_window(
        window=overlap_window,
        maximum_dimension=(AE_SELECTION_COARSE_MAX_DIM),
    )

    coarse_tested_pixels = coarse_height * coarse_width

    exact_scan_used = False
    exact_blocks_scanned = 0
    exact_valid_pixels_found = 0

    with rasterio.open(path) as src:
        with make_warped_vrt(
            src=src,
            grid=grid,
            resampling=AE_RESAMPLING,
        ) as vrt:
            coarse_masks = vrt.read_masks(
                indexes=(AE_SELECTION_PROBE_BANDS),
                window=overlap_window,
                out_shape=(
                    len(AE_SELECTION_PROBE_BANDS),
                    coarse_height,
                    coarse_width,
                ),
                resampling=(Resampling.nearest),
            )

            coarse_valid = np.all(
                coarse_masks > 0,
                axis=0,
            )

            coarse_valid_pixels = int(coarse_valid.sum())

            del coarse_masks
            del coarse_valid

            if coarse_valid_pixels == 0:
                exact_scan_used = True

                for block_window in iter_subwindows(
                    parent=overlap_window,
                    block_size=(AE_SELECTION_EXACT_BLOCK_SIZE),
                ):
                    exact_blocks_scanned += 1

                    array = vrt.read(
                        indexes=(AE_SELECTION_PROBE_BANDS),
                        window=block_window,
                        out_dtype="float32",
                    )

                    valid = np.all(
                        np.isfinite(array),
                        axis=0,
                    )

                    if vrt.nodata is not None:
                        valid &= ~np.any(
                            array == vrt.nodata,
                            axis=0,
                        )

                    block_valid = int(valid.sum())

                    del array
                    del valid

                    if block_valid > 0:
                        exact_valid_pixels_found = block_valid
                        break

    has_valid_data = coarse_valid_pixels > 0 or exact_valid_pixels_found > 0

    return {
        "has_valid_data": (has_valid_data),
        "coarse_valid_pixels": (coarse_valid_pixels),
        "coarse_tested_pixels": (coarse_tested_pixels),
        "coarse_height": (coarse_height),
        "coarse_width": (coarse_width),
        "overlap_window_col_off": int(overlap_window.col_off),
        "overlap_window_row_off": int(overlap_window.row_off),
        "overlap_window_width": int(overlap_window.width),
        "overlap_window_height": int(overlap_window.height),
        "exact_scan_used": (exact_scan_used),
        "exact_blocks_scanned": (exact_blocks_scanned),
        "exact_valid_pixels_found": (exact_valid_pixels_found),
    }


def select_ae_source_zone_for_tile(
    *,
    tile_id: str,
    grid: GridDefinition,
    ae_catalog: pd.DataFrame,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    reference_rows = ae_catalog[
        ae_catalog["year"] == AE_SELECTION_REFERENCE_YEAR
    ].copy()

    if reference_rows.empty:
        raise RuntimeError(
            "No AlphaEarth catalog rows for "
            "selection reference year "
            f"{AE_SELECTION_REFERENCE_YEAR}."
        )

    diagnostics: list[dict[str, Any]] = []

    target_area = (grid.bounds_values[2] - grid.bounds_values[0]) * (
        grid.bounds_values[3] - grid.bounds_values[1]
    )

    for row in reference_rows.itertuples():
        transformed_bounds = (
            float(row.target_bounds_minx),
            float(row.target_bounds_miny),
            float(row.target_bounds_maxx),
            float(row.target_bounds_maxy),
        )

        overlap_left = max(
            transformed_bounds[0],
            grid.bounds_values[0],
        )

        overlap_bottom = max(
            transformed_bounds[1],
            grid.bounds_values[1],
        )

        overlap_right = min(
            transformed_bounds[2],
            grid.bounds_values[2],
        )

        overlap_top = min(
            transformed_bounds[3],
            grid.bounds_values[3],
        )

        overlap_bounds = (
            overlap_left,
            overlap_bottom,
            overlap_right,
            overlap_top,
        )

        overlap_area = intersection_area(
            transformed_bounds,
            grid.bounds_values,
        )

        if overlap_area <= 0:
            continue

        probe = probe_ae_candidate(
            path=Path(row.path),
            grid=grid,
            overlap_bounds=(overlap_bounds),
        )

        diagnostics.append(
            {
                "tile_id": tile_id,
                "reference_year": (AE_SELECTION_REFERENCE_YEAR),
                "source_zone": (row.source_zone),
                "path": (row.path),
                "stored_crs": (row.stored_crs),
                "overlap_area_m2": (overlap_area),
                "overlap_fraction_of_tile": (
                    overlap_area / target_area if target_area > 0 else 0.0
                ),
                "overlap_bounds": (overlap_bounds),
                "has_valid_data": (probe["has_valid_data"]),
                "coarse_valid_pixels": (probe["coarse_valid_pixels"]),
                "coarse_tested_pixels": (probe["coarse_tested_pixels"]),
                "coarse_height": (probe["coarse_height"]),
                "coarse_width": (probe["coarse_width"]),
                "overlap_window_col_off": (probe["overlap_window_col_off"]),
                "overlap_window_row_off": (probe["overlap_window_row_off"]),
                "overlap_window_width": (probe["overlap_window_width"]),
                "overlap_window_height": (probe["overlap_window_height"]),
                "exact_scan_used": (probe["exact_scan_used"]),
                "exact_blocks_scanned": (probe["exact_blocks_scanned"]),
                "exact_valid_pixels_found": (probe["exact_valid_pixels_found"]),
                "selected": False,
            }
        )

    if not diagnostics:
        raise RuntimeError(
            "No AlphaEarth mosaic overlaps "
            "the EPSG:25833 target grid "
            f"for tile {tile_id}."
        )

    selected = max(
        diagnostics,
        key=lambda row: (
            int(bool(row["has_valid_data"])),
            int(row["coarse_valid_pixels"]),
            int(row["exact_valid_pixels_found"]),
            float(row["overlap_area_m2"]),
        ),
    )

    selected["selected"] = True

    if STRICT_ABORT_IF_SOURCE_ALL_NODATA and not bool(selected["has_valid_data"]):
        raise RuntimeError(
            "Every overlapping AlphaEarth "
            "candidate produced zero valid "
            "probe pixels for tile "
            f"{tile_id}."
        )

    selected_zone = str(selected["source_zone"])

    logging.info(
        "AE source selected | tile=%s "
        "source_zone=%s path=%s | "
        "has_valid_data=%s "
        "coarse_valid=%d/%d | "
        "exact_scan_used=%s exact_blocks=%d "
        "exact_valid_found=%d | "
        "overlap_fraction=%.6f "
        "overlap_bounds=%s",
        tile_id,
        selected_zone,
        selected["path"],
        selected["has_valid_data"],
        selected["coarse_valid_pixels"],
        selected["coarse_tested_pixels"],
        selected["exact_scan_used"],
        selected["exact_blocks_scanned"],
        selected["exact_valid_pixels_found"],
        selected["overlap_fraction_of_tile"],
        selected["overlap_bounds"],
    )

    return (
        selected_zone,
        diagnostics,
    )


def detect_s2_band_indexes(
    src,
) -> list[int]:
    indexes = [S2_BAND_FALLBACK_INDEXES[band_name] for band_name in S2_BAND_NAMES]

    maximum_index = max(indexes)

    if src.count < maximum_index:
        raise ValueError(
            "Sentinel-2 file has only "
            f"{src.count} bands, but fixed "
            f"raster indexes {indexes} are "
            f"required: {src.name}"
        )

    if len(set(indexes)) != len(indexes):
        raise ValueError(
            "S2_BAND_FALLBACK_INDEXES "
            "contains duplicate indexes: "
            f"{S2_BAND_FALLBACK_INDEXES}"
        )

    return indexes


def initialise_sampling_worker() -> None:
    os.environ["GDAL_CACHEMAX"] = str(GDAL_CACHE_MB_PER_WORKER)

    os.environ["GDAL_NUM_THREADS"] = str(GDAL_WARP_THREADS_PER_WORKER)


def load_zone_polygons(
    zone: str,
) -> gpd.GeoDataFrame:
    if zone not in _WORKER_ZONE_POLYGONS:
        path = zone_cache_paths(zone)["polygons"]

        if not path.exists():
            raise FileNotFoundError(f"Missing zone polygon cache: {path}")

        _WORKER_ZONE_POLYGONS[zone] = joblib.load(path)

    return _WORKER_ZONE_POLYGONS[zone]


def read_tile_ar5(
    zone: str,
    tile_bounds: tuple[
        float,
        float,
        float,
        float,
    ],
) -> gpd.GeoDataFrame:
    paths = zone_cache_paths(zone)

    if paths["empty_ar5"].exists():
        return gpd.GeoDataFrame(
            {AR5_AREALTYPE_COLUMN: []},
            geometry=[],
            crs=CRS.from_wkt(
                json.loads(paths["metadata"].read_text(encoding="utf-8"))["crs_wkt"]
            ),
        )

    if not paths["ar5"].exists():
        raise FileNotFoundError(f"Missing AR5 zone cache: {paths['ar5']}")

    try:
        ar5 = gpd.read_file(
            paths["ar5"],
            bbox=tile_bounds,
            columns=[AR5_AREALTYPE_COLUMN],
            engine="pyogrio",
            use_arrow=True,
        )
    except Exception:
        ar5 = gpd.read_file(
            paths["ar5"],
            bbox=tile_bounds,
        )

    return ar5


def subset_tile_vectors(
    zone: str,
    grid: GridDefinition,
) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    polygons_zone = load_zone_polygons(zone)

    tile_box = box(*grid.bounds_values)

    polygon_indices = polygons_zone.sindex.query(
        tile_box,
        predicate="intersects",
    )

    tile_polygons = polygons_zone.iloc[polygon_indices].copy()

    tile_ar5 = read_tile_ar5(
        zone=zone,
        tile_bounds=(grid.bounds_values),
    )

    return (
        tile_polygons,
        tile_ar5,
    )


def tile_mask_paths(
    zone: str,
    tile_id: str,
    visible_year: int | None = None,
) -> dict[str, Path]:
    root = TILE_MASK_CACHE_ROOT / zone / tile_id

    result = {
        "root": root,
        "metadata": (root / "grid_metadata.json"),
        "manual": (root / "manual_nochange.npy"),
        "ar5": (root / "ar5_arealtype.npy"),
    }

    if visible_year is not None:
        result["positive"] = root / f"positive_{visible_year}.npy"

    return result


def write_grid_metadata(
    path: Path,
    grid: GridDefinition,
) -> None:
    path.write_text(
        json.dumps(
            {
                "crs_wkt": (grid.crs_wkt),
                "transform_values": (grid.transform_values),
                "width": (grid.width),
                "height": (grid.height),
                "bounds_values": (grid.bounds_values),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def validate_mask_grid(
    metadata_path: Path,
    grid: GridDefinition,
) -> bool:
    if not metadata_path.exists():
        return False

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    cached = GridDefinition(
        crs_wkt=(metadata["crs_wkt"]),
        transform_values=tuple(metadata["transform_values"]),
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        bounds_values=tuple(metadata["bounds_values"]),
    )

    return grids_match(
        cached,
        grid,
    )


def build_static_tile_masks(
    zone: str,
    tile_id: str,
    grid: GridDefinition,
    tile_polygons: gpd.GeoDataFrame,
    tile_ar5: gpd.GeoDataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    paths = tile_mask_paths(
        zone,
        tile_id,
    )

    reusable = (
        not REBUILD_TILE_MASK_CACHE
        and paths["manual"].exists()
        and paths["ar5"].exists()
        and validate_mask_grid(
            paths["metadata"],
            grid,
        )
    )

    if reusable:
        return (
            np.load(
                paths["manual"],
                mmap_mode="r",
            ),
            np.load(
                paths["ar5"],
                mmap_mode="r",
            ),
        )

    if paths["root"].exists():
        shutil.rmtree(paths["root"])

    paths["root"].mkdir(
        parents=True,
        exist_ok=True,
    )

    nochange_polygons = tile_polygons[tile_polygons["class_norm"] == "nochange"]

    manual_shapes = [
        (
            geometry,
            1,
        )
        for geometry in nochange_polygons.geometry
        if (geometry is not None and not geometry.is_empty)
    ]

    ar5_shapes = [
        (
            geometry,
            int(arealtype),
        )
        for (
            geometry,
            arealtype,
        ) in zip(
            tile_ar5.geometry,
            tile_ar5[AR5_AREALTYPE_COLUMN],
        )
        if (
            geometry is not None
            and not geometry.is_empty
            and int(arealtype) in AR5_TARGET_CLASSES
        )
    ]

    out_shape = (
        grid.height,
        grid.width,
    )

    manual = rasterize(
        manual_shapes,
        out_shape=out_shape,
        transform=grid.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=(RASTERIZE_ALL_TOUCHED),
    )

    ar5 = rasterize(
        ar5_shapes,
        out_shape=out_shape,
        transform=grid.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=(RASTERIZE_ALL_TOUCHED),
    )

    ar5[manual > 0] = 0

    np.save(
        paths["manual"],
        manual,
        allow_pickle=False,
    )

    np.save(
        paths["ar5"],
        ar5,
        allow_pickle=False,
    )

    write_grid_metadata(
        paths["metadata"],
        grid,
    )

    del manual
    del ar5

    return (
        np.load(
            paths["manual"],
            mmap_mode="r",
        ),
        np.load(
            paths["ar5"],
            mmap_mode="r",
        ),
    )


def build_positive_mask(
    zone: str,
    tile_id: str,
    visible_year: int,
    grid: GridDefinition,
    tile_polygons: gpd.GeoDataFrame,
) -> np.ndarray:
    paths = tile_mask_paths(
        zone,
        tile_id,
        visible_year,
    )

    reusable = (
        not REBUILD_TILE_MASK_CACHE
        and paths["positive"].exists()
        and validate_mask_grid(
            paths["metadata"],
            grid,
        )
    )

    if reusable:
        return np.load(
            paths["positive"],
            mmap_mode="r",
        )

    event_polygons = tile_polygons[
        tile_polygons["class_norm"].isin(
            [
                "clearcut",
                "urban",
            ]
        )
        & (tile_polygons["visible_year"] == visible_year)
    ]

    shapes = []

    for (
        geometry,
        class_name,
    ) in zip(
        event_polygons.geometry,
        event_polygons["class_norm"],
    ):
        if geometry is None or geometry.is_empty:
            continue

        value = 1 if class_name == "clearcut" else 2

        shapes.append(
            (
                geometry,
                value,
            )
        )

    positive = rasterize(
        shapes,
        out_shape=(
            grid.height,
            grid.width,
        ),
        transform=grid.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=(RASTERIZE_ALL_TOUCHED),
    )

    paths["root"].mkdir(
        parents=True,
        exist_ok=True,
    )

    if not paths["metadata"].exists():
        write_grid_metadata(
            paths["metadata"],
            grid,
        )

    np.save(
        paths["positive"],
        positive,
        allow_pickle=False,
    )

    del positive

    return np.load(
        paths["positive"],
        mmap_mode="r",
    )


def select_all_coordinates(
    mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    row_parts: list[np.ndarray] = []

    col_parts: list[np.ndarray] = []

    for row_start in range(
        0,
        mask.shape[0],
        MASK_SELECTION_STRIPE_ROWS,
    ):
        row_end = min(
            row_start + MASK_SELECTION_STRIPE_ROWS,
            mask.shape[0],
        )

        rows, cols = np.where(mask[row_start:row_end])

        if len(rows):
            row_parts.append(rows.astype(np.int32) + row_start)

            col_parts.append(cols.astype(np.int32))

    if not row_parts:
        return (
            np.empty(
                0,
                dtype=np.int32,
            ),
            np.empty(
                0,
                dtype=np.int32,
            ),
        )

    return (
        np.concatenate(row_parts),
        np.concatenate(col_parts),
    )


def select_random_coordinates(
    mask: np.ndarray,
    target: int,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    total_available = int(np.count_nonzero(mask))

    target = min(
        int(target),
        total_available,
    )

    if target <= 0:
        return (
            np.empty(
                0,
                dtype=np.int32,
            ),
            np.empty(
                0,
                dtype=np.int32,
            ),
        )

    if target == total_available:
        return select_all_coordinates(mask)

    row_parts: list[np.ndarray] = []

    col_parts: list[np.ndarray] = []

    remaining_available = total_available

    remaining_target = target

    for row_start in range(
        0,
        mask.shape[0],
        MASK_SELECTION_STRIPE_ROWS,
    ):
        row_end = min(
            row_start + MASK_SELECTION_STRIPE_ROWS,
            mask.shape[0],
        )

        stripe = mask[row_start:row_end]

        stripe_available = int(np.count_nonzero(stripe))

        if stripe_available == 0:
            continue

        if remaining_target <= 0:
            break

        if remaining_target == remaining_available:
            stripe_take = stripe_available
        else:
            stripe_take = int(
                rng.hypergeometric(
                    ngood=(stripe_available),
                    nbad=(remaining_available - stripe_available),
                    nsample=(remaining_target),
                )
            )

        if stripe_take > 0:
            rows, cols = np.where(stripe)

            if stripe_take < len(rows):
                selected = rng.choice(
                    len(rows),
                    size=stripe_take,
                    replace=False,
                )

                rows = rows[selected]

                cols = cols[selected]

            row_parts.append(rows.astype(np.int32) + row_start)

            col_parts.append(cols.astype(np.int32))

        remaining_available -= stripe_available

        remaining_target -= stripe_take

    if remaining_target != 0:
        raise RuntimeError(
            f"Coordinate selector missed {remaining_target} requested rows."
        )

    return (
        np.concatenate(row_parts),
        np.concatenate(col_parts),
    )


def allocate_equal_with_capacity(
    total_target: int,
    capacities: dict[
        int,
        int,
    ],
) -> dict[
    int,
    int,
]:
    allocations = {key: 0 for key in capacities}

    remaining = min(
        int(total_target),
        int(sum(capacities.values())),
    )

    active = {
        key
        for (
            key,
            capacity,
        ) in capacities.items()
        if capacity > 0
    }

    while remaining > 0 and active:
        per_class = max(
            1,
            math.ceil(remaining / len(active)),
        )

        allocated_this_round = 0

        for key in sorted(active):
            capacity_left = capacities[key] - allocations[key]

            if capacity_left <= 0:
                continue

            take = min(
                per_class,
                capacity_left,
                remaining,
            )

            allocations[key] += take

            remaining -= take

            allocated_this_round += take

            if remaining <= 0:
                break

        active = {key for key in active if (allocations[key] < capacities[key])}

        if allocated_this_round == 0:
            break

    return allocations


def s2_dataset_matches_grid(
    src,
    grid: GridDefinition,
    source_year: int,
) -> bool:
    effective_crs = effective_s2_source_crs(
        src,
        source_year,
    )

    effective_grid = GridDefinition(
        crs_wkt=(effective_crs.to_wkt()),
        transform_values=(tuple(src.transform)[:6]),
        width=src.width,
        height=src.height,
        bounds_values=(tuple(src.bounds)),
    )

    return grids_match(
        effective_grid,
        grid,
    )


def make_s2_warped_vrt(
    src,
    grid: GridDefinition,
    source_year: int,
):
    source_crs = effective_s2_source_crs(
        src,
        source_year,
    )

    return make_warped_vrt(
        src=src,
        grid=grid,
        resampling=S2_RESAMPLING,
        src_crs_override=source_crs,
    )


def valid_pixel_rows(
    arrays_and_nodata: list[
        tuple[
            np.ndarray,
            Any,
        ]
    ],
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    valid = np.ones(
        len(rows),
        dtype=bool,
    )

    for (
        array,
        nodata,
    ) in arrays_and_nodata:
        pixels = array[
            :,
            rows,
            cols,
        ].T

        valid &= np.all(
            np.isfinite(pixels),
            axis=1,
        )

        if nodata is not None:
            valid &= ~np.any(
                pixels == nodata,
                axis=1,
            )

    return valid


def build_combined_features(
    ae_pre: np.ndarray,
    ae_post: np.ndarray,
    s2_pre: np.ndarray,
    s2_post: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    visible_year: int,
) -> np.ndarray:
    ae_pre_pixels = ae_pre[
        :,
        rows,
        cols,
    ].T.astype(
        FEATURE_DTYPE,
        copy=False,
    )

    ae_post_pixels = ae_post[
        :,
        rows,
        cols,
    ].T.astype(
        FEATURE_DTYPE,
        copy=False,
    )

    s2_pre_pixels = s2_pre[
        :,
        rows,
        cols,
    ].T.astype(
        FEATURE_DTYPE,
        copy=False,
    )

    s2_post_pixels = s2_post[
        :,
        rows,
        cols,
    ].T.astype(
        FEATURE_DTYPE,
        copy=False,
    )

    year_column = np.full(
        (
            len(rows),
            1,
        ),
        float(visible_year),
        dtype=FEATURE_DTYPE,
    )

    return np.concatenate(
        [
            ae_pre_pixels,
            ae_post_pixels,
            (ae_post_pixels - ae_pre_pixels),
            s2_pre_pixels,
            s2_post_pixels,
            (s2_post_pixels - s2_pre_pixels),
            year_column,
        ],
        axis=1,
        dtype=FEATURE_DTYPE,
    )


def national_spatial_groups(
    grid: GridDefinition,
    rows: np.ndarray,
    cols: np.ndarray,
    zone: str,
) -> np.ndarray:
    row_float = rows.astype(np.float64) + 0.5

    col_float = cols.astype(np.float64) + 0.5

    transform = grid.transform

    native_x = transform.c + transform.a * col_float + transform.b * row_float

    native_y = transform.f + transform.d * col_float + transform.e * row_float

    transformer_key = f"{zone}|{grid.crs_wkt}|{SPATIAL_GROUP_CRS}"

    transformer = _WORKER_GROUP_TRANSFORMERS.get(transformer_key)

    if transformer is None:
        transformer = Transformer.from_crs(
            grid.crs,
            SPATIAL_GROUP_CRS,
            always_xy=True,
        )

        _WORKER_GROUP_TRANSFORMERS[transformer_key] = transformer

    (
        national_x,
        national_y,
    ) = transformer.transform(
        native_x,
        native_y,
    )

    group_x = np.floor(np.asarray(national_x) / SPATIAL_GROUP_SIZE_METRES).astype(
        np.int64
    )

    group_y = np.floor(np.asarray(national_y) / SPATIAL_GROUP_SIZE_METRES).astype(
        np.int64
    )

    encoded_x = group_x + SPATIAL_GROUP_ID_OFFSET

    encoded_y = group_y + SPATIAL_GROUP_ID_OFFSET

    if (
        np.any(encoded_x < 0)
        or np.any(encoded_y < 0)
        or np.any(encoded_y >= SPATIAL_GROUP_ID_MULTIPLIER)
    ):
        raise RuntimeError("Spatial-group encoding range was exceeded.")

    return (encoded_x * np.int64(SPATIAL_GROUP_ID_MULTIPLIER) + encoded_y).astype(
        GROUP_DTYPE
    )


class SamplePartBuffer:
    def __init__(
        self,
        output_dir: Path,
        tile_id: str,
        zone: str,
        visible_year: int,
    ) -> None:
        self.output_dir = output_dir

        self.tile_id = tile_id

        self.zone = zone

        self.visible_year = visible_year

        self.x_parts: list[np.ndarray] = []

        self.y_parts: list[np.ndarray] = []

        self.group_parts: list[np.ndarray] = []

        self.source_parts: list[np.ndarray] = []

        self.year_parts: list[np.ndarray] = []

        self.rows = 0
        self.part_number = 0

        self.manifest_rows: list[dict[str, Any]] = []

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def add(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        sources: np.ndarray,
        years: np.ndarray,
    ) -> None:
        if len(y) == 0:
            return

        self.x_parts.append(X)

        self.y_parts.append(y)

        self.group_parts.append(groups)

        self.source_parts.append(sources)

        self.year_parts.append(years)

        self.rows += len(y)

        if self.rows >= CACHE_PART_MAX_ROWS:
            self.flush()

    def flush(self) -> None:
        if self.rows == 0:
            return

        X = np.concatenate(
            self.x_parts,
            axis=0,
        )

        y = np.concatenate(self.y_parts)

        groups = np.concatenate(self.group_parts)

        sources = np.concatenate(self.source_parts)

        years = np.concatenate(self.year_parts)

        prefix = self.output_dir / f"part_{self.part_number:05d}"

        x_path = Path(f"{prefix}_X.npy")

        y_path = Path(f"{prefix}_y.npy")

        groups_path = Path(f"{prefix}_groups.npy")

        sources_path = Path(f"{prefix}_sources.npy")

        years_path = Path(f"{prefix}_years.npy")

        np.save(
            x_path,
            X,
            allow_pickle=False,
        )

        np.save(
            y_path,
            y,
            allow_pickle=False,
        )

        np.save(
            groups_path,
            groups,
            allow_pickle=False,
        )

        np.save(
            sources_path,
            sources,
            allow_pickle=False,
        )

        np.save(
            years_path,
            years,
            allow_pickle=False,
        )

        self.manifest_rows.append(
            {
                "tile_id": (self.tile_id),
                "zone": (self.zone),
                "visible_year": (self.visible_year),
                "part_number": (self.part_number),
                "rows": (len(y)),
                "features": (X.shape[1]),
                "x_path": (str(x_path)),
                "y_path": (str(y_path)),
                "groups_path": (str(groups_path)),
                "sources_path": (str(sources_path)),
                "years_path": (str(years_path)),
                "clearcut": int(np.count_nonzero(y == CLEARCUT)),
                "urban": int(np.count_nonzero(y == URBAN)),
                "nochange": int(np.count_nonzero(y == NOCHANGE)),
                "manual_nochange": int(
                    np.count_nonzero(sources == MANUAL_NOCHANGE_SOURCE)
                ),
                "ar5_11": int(np.count_nonzero(sources == 11)),
                "ar5_12": int(np.count_nonzero(sources == 12)),
                "ar5_21": int(np.count_nonzero(sources == 21)),
                "ar5_22": int(np.count_nonzero(sources == 22)),
                "ar5_81": int(np.count_nonzero(sources == 81)),
                "ar5_82": int(np.count_nonzero(sources == 82)),
            }
        )

        self.part_number += 1
        self.rows = 0

        self.x_parts.clear()
        self.y_parts.clear()
        self.group_parts.clear()
        self.source_parts.clear()
        self.year_parts.clear()

        del X
        del y
        del groups
        del sources
        del years

    def finish(
        self,
    ) -> list[dict[str, Any]]:
        self.flush()

        return self.manifest_rows


def open_pair_sources(
    *,
    stack: ExitStack,
    grid: GridDefinition,
    pair: dict[str, Any],
) -> dict[str, Any]:
    s2_pre_year = int(pair["s2_pre_year"])

    s2_post_year = int(pair["s2_post_year"])

    s2_pre_path = Path(pair["s2_pre_path"])

    s2_post_path = Path(pair["s2_post_path"])

    ae_pre_path = Path(pair["ae_pre_path"])

    ae_post_path = Path(pair["ae_post_path"])

    s2_pre_base = stack.enter_context(rasterio.open(s2_pre_path))

    s2_post_base = stack.enter_context(rasterio.open(s2_post_path))

    s2_pre_indexes = detect_s2_band_indexes(s2_pre_base)

    s2_post_indexes = detect_s2_band_indexes(s2_post_base)

    if s2_dataset_matches_grid(
        s2_pre_base,
        grid,
        s2_pre_year,
    ):
        s2_pre_src = s2_pre_base
    else:
        s2_pre_src = stack.enter_context(
            make_s2_warped_vrt(
                src=s2_pre_base,
                grid=grid,
                source_year=(s2_pre_year),
            )
        )

    if s2_dataset_matches_grid(
        s2_post_base,
        grid,
        s2_post_year,
    ):
        s2_post_src = s2_post_base
    else:
        s2_post_src = stack.enter_context(
            make_s2_warped_vrt(
                src=s2_post_base,
                grid=grid,
                source_year=(s2_post_year),
            )
        )

    ae_pre_base = stack.enter_context(rasterio.open(ae_pre_path))

    ae_post_base = stack.enter_context(rasterio.open(ae_post_path))

    ae_pre_src = stack.enter_context(
        make_warped_vrt(
            src=ae_pre_base,
            grid=grid,
            resampling=(AE_RESAMPLING),
        )
    )

    ae_post_src = stack.enter_context(
        make_warped_vrt(
            src=ae_post_base,
            grid=grid,
            resampling=(AE_RESAMPLING),
        )
    )

    return {
        "s2_pre_year": (s2_pre_year),
        "s2_post_year": (s2_post_year),
        "s2_pre_path": (s2_pre_path),
        "s2_post_path": (s2_post_path),
        "ae_pre_path": (ae_pre_path),
        "ae_post_path": (ae_post_path),
        "s2_pre_indexes": (s2_pre_indexes),
        "s2_post_indexes": (s2_post_indexes),
        "s2_pre_src": (s2_pre_src),
        "s2_post_src": (s2_post_src),
        "ae_pre_src": (ae_pre_src),
        "ae_post_src": (ae_post_src),
    }


def full_grid_coarse_shape(
    *,
    grid: GridDefinition,
    maximum_dimension: int,
) -> tuple[int, int]:
    full_window = Window(
        col_off=0,
        row_off=0,
        width=grid.width,
        height=grid.height,
    )

    return reduced_shape_for_window(
        window=full_window,
        maximum_dimension=(maximum_dimension),
    )


def coarse_source_mask(
    *,
    src,
    indexes: list[int] | None,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    if indexes is None:
        masks = src.read_masks(
            out_shape=(
                src.count,
                output_height,
                output_width,
            ),
            resampling=(Resampling.nearest),
        )
    else:
        masks = src.read_masks(
            indexes=indexes,
            out_shape=(
                len(indexes),
                output_height,
                output_width,
            ),
            resampling=(Resampling.nearest),
        )

    valid = np.all(
        masks > 0,
        axis=0,
    )

    del masks

    return valid


def exact_block_source_mask(
    *,
    src,
    indexes: list[int] | None,
    window: Window,
) -> np.ndarray:
    if indexes is None:
        masks = src.read_masks(window=window)
    else:
        masks = src.read_masks(
            indexes=indexes,
            window=window,
        )

    valid = np.all(
        masks > 0,
        axis=0,
    )

    del masks

    return valid


def exact_preflight_fallback_scan(
    *,
    sources: dict[str, Any],
    grid: GridDefinition,
) -> dict[str, Any]:
    full_window = Window(
        col_off=0,
        row_off=0,
        width=grid.width,
        height=grid.height,
    )

    source_has_valid_data = {
        "ae_pre": False,
        "ae_post": False,
        "s2_pre": False,
        "s2_post": False,
    }

    source_valid_pixels_found = {key: 0 for key in source_has_valid_data}

    combined_has_valid_data = False
    combined_valid_pixels_found = 0

    blocks_scanned = 0
    pixels_scanned = 0

    for block_window in iter_subwindows(
        parent=full_window,
        block_size=(PREFLIGHT_EXACT_BLOCK_SIZE),
    ):
        blocks_scanned += 1

        pixels_scanned += int(block_window.width * block_window.height)

        masks = {
            "ae_pre": (
                exact_block_source_mask(
                    src=sources["ae_pre_src"],
                    indexes=None,
                    window=block_window,
                )
            ),
            "ae_post": (
                exact_block_source_mask(
                    src=sources["ae_post_src"],
                    indexes=None,
                    window=block_window,
                )
            ),
            "s2_pre": (
                exact_block_source_mask(
                    src=sources["s2_pre_src"],
                    indexes=sources["s2_pre_indexes"],
                    window=block_window,
                )
            ),
            "s2_post": (
                exact_block_source_mask(
                    src=sources["s2_post_src"],
                    indexes=sources["s2_post_indexes"],
                    window=block_window,
                )
            ),
        }

        for (
            key,
            valid_mask,
        ) in masks.items():
            valid_count = int(valid_mask.sum())

            if valid_count > 0:
                source_has_valid_data[key] = True

                if source_valid_pixels_found[key] == 0:
                    source_valid_pixels_found[key] = valid_count

        combined = (
            masks["ae_pre"] & masks["ae_post"] & masks["s2_pre"] & masks["s2_post"]
        )

        combined_count = int(combined.sum())

        if combined_count > 0:
            combined_has_valid_data = True

            if combined_valid_pixels_found == 0:
                combined_valid_pixels_found = combined_count

        del masks
        del combined

        if all(source_has_valid_data.values()) and combined_has_valid_data:
            break

    return {
        "blocks_scanned": (blocks_scanned),
        "pixels_scanned": (pixels_scanned),
        "source_has_valid_data": (source_has_valid_data),
        "source_valid_pixels_found": (source_valid_pixels_found),
        "combined_has_valid_data": (combined_has_valid_data),
        "combined_valid_pixels_found": (combined_valid_pixels_found),
    }


def preflight_tile_year_sources(
    *,
    tile_id: str,
    zone: str,
    grid: GridDefinition,
    pair: dict[str, Any],
) -> dict[str, Any]:
    visible_year = int(pair["visible_year"])

    diagnostic_path = (
        INPUT_DIAGNOSTICS_ROOT / tile_id / (f"{visible_year}_input_preflight.json")
    )

    diagnostic_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not RUN_INPUT_PREFLIGHT:
        return {
            "tile_id": tile_id,
            "zone": zone,
            "visible_year": (visible_year),
            "status": "disabled",
        }

    (
        output_height,
        output_width,
    ) = full_grid_coarse_shape(
        grid=grid,
        maximum_dimension=(PREFLIGHT_COARSE_MAX_DIM),
    )

    with ExitStack() as stack:
        sources = open_pair_sources(
            stack=stack,
            grid=grid,
            pair=pair,
        )

        masks = {
            "ae_pre": (
                coarse_source_mask(
                    src=sources["ae_pre_src"],
                    indexes=None,
                    output_height=(output_height),
                    output_width=(output_width),
                )
            ),
            "ae_post": (
                coarse_source_mask(
                    src=sources["ae_post_src"],
                    indexes=None,
                    output_height=(output_height),
                    output_width=(output_width),
                )
            ),
            "s2_pre": (
                coarse_source_mask(
                    src=sources["s2_pre_src"],
                    indexes=sources["s2_pre_indexes"],
                    output_height=(output_height),
                    output_width=(output_width),
                )
            ),
            "s2_post": (
                coarse_source_mask(
                    src=sources["s2_post_src"],
                    indexes=sources["s2_post_indexes"],
                    output_height=(output_height),
                    output_width=(output_width),
                )
            ),
        }

        combined = (
            masks["ae_pre"] & masks["ae_post"] & masks["s2_pre"] & masks["s2_post"]
        )

        coarse_tested_pixels = int(output_height * output_width)

        coarse_source_valid_pixels = {
            key: int(value.sum())
            for (
                key,
                value,
            ) in masks.items()
        }

        coarse_combined_valid_pixels = int(combined.sum())

        suspicious = (
            any(valid_count == 0 for valid_count in coarse_source_valid_pixels.values())
            or coarse_combined_valid_pixels == 0
        )

        exact_result: dict[str, Any] | None = None

        if suspicious:
            logging.warning(
                "Coarse preflight returned "
                "zero coverage for tile=%s "
                "visible_year=%d. Running "
                "exact fallback with block "
                "size %d.",
                tile_id,
                visible_year,
                PREFLIGHT_EXACT_BLOCK_SIZE,
            )

            exact_result = exact_preflight_fallback_scan(
                sources=sources,
                grid=grid,
            )

            source_has_valid_data = dict(exact_result["source_has_valid_data"])

            combined_has_valid_data = bool(exact_result["combined_has_valid_data"])

        else:
            source_has_valid_data = {
                key: count > 0
                for (
                    key,
                    count,
                ) in coarse_source_valid_pixels.items()
            }

            combined_has_valid_data = coarse_combined_valid_pixels > 0

        summary_source_valid_pixels = {
            key: (
                coarse_source_valid_pixels[key]
                if (coarse_source_valid_pixels[key] > 0)
                else (
                    exact_result["source_valid_pixels_found"][key]
                    if exact_result is not None
                    else 0
                )
            )
            for key in coarse_source_valid_pixels
        }

        summary_combined_valid_pixels = (
            coarse_combined_valid_pixels
            if coarse_combined_valid_pixels > 0
            else (
                exact_result["combined_valid_pixels_found"]
                if exact_result is not None
                else 0
            )
        )

        summary = {
            "tile_id": tile_id,
            "zone": zone,
            "visible_year": (visible_year),
            "diagnostic_method": ("nearest_coarse_scan_with_exact_native_fallback"),
            "coarse_output_height": (output_height),
            "coarse_output_width": (output_width),
            "coarse_tested_pixels": (coarse_tested_pixels),
            "coarse_source_valid_pixels": (coarse_source_valid_pixels),
            "coarse_combined_valid_pixels": (coarse_combined_valid_pixels),
            "exact_fallback_used": (exact_result is not None),
            "exact_fallback": (exact_result),
            "source_has_valid_data": (source_has_valid_data),
            "combined_has_valid_data": (combined_has_valid_data),
            "target_crs": (TARGET_CRS_TEXT),
            "target_bounds": (grid.bounds_values),
            "target_width": (grid.width),
            "target_height": (grid.height),
            "ae_source_zone": (pair["ae_source_zone"]),
            "ae_pre_path": str(sources["ae_pre_path"]),
            "ae_post_path": str(sources["ae_post_path"]),
            "s2_pre_path": str(sources["s2_pre_path"]),
            "s2_post_path": str(sources["s2_post_path"]),
            "s2_pre_indexes": (sources["s2_pre_indexes"]),
            "s2_post_indexes": (sources["s2_post_indexes"]),
            "source_valid_pixels": (summary_source_valid_pixels),
            "source_tested_pixels": {
                key: (coarse_tested_pixels) for key in summary_source_valid_pixels
            },
            "combined_valid_pixels": (summary_combined_valid_pixels),
            "combined_tested_pixels": (coarse_tested_pixels),
            "status": "passed",
        }

        del masks
        del combined

    zero_sources = [
        source
        for (
            source,
            has_valid,
        ) in source_has_valid_data.items()
        if not has_valid
    ]

    if zero_sources:
        summary["status"] = "failed_source_all_nodata"

    elif not combined_has_valid_data:
        summary["status"] = "failed_combined_all_nodata"

    diagnostic_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    logging.info(
        "INPUT PREFLIGHT | tile=%s "
        "visible_year=%d | "
        "coarse_shape=%dx%d | "
        "ae_pre=%d ae_post=%d "
        "s2_pre=%d s2_post=%d "
        "combined=%d | exact_used=%s | "
        "source_has_valid=%s "
        "combined_has_valid=%s | "
        "status=%s",
        tile_id,
        visible_year,
        output_width,
        output_height,
        coarse_source_valid_pixels["ae_pre"],
        coarse_source_valid_pixels["ae_post"],
        coarse_source_valid_pixels["s2_pre"],
        coarse_source_valid_pixels["s2_post"],
        coarse_combined_valid_pixels,
        exact_result is not None,
        source_has_valid_data,
        combined_has_valid_data,
        summary["status"],
    )

    if exact_result is not None:
        logging.info(
            "INPUT PREFLIGHT EXACT | "
            "tile=%s visible_year=%d | "
            "blocks=%d pixels=%d "
            "source_valid_found=%s "
            "combined_valid_found=%d",
            tile_id,
            visible_year,
            exact_result["blocks_scanned"],
            exact_result["pixels_scanned"],
            exact_result["source_valid_pixels_found"],
            exact_result["combined_valid_pixels_found"],
        )

    if STRICT_ABORT_IF_SOURCE_ALL_NODATA and zero_sources:
        raise RuntimeError(
            "Exact preflight confirmed "
            "an all-nodata source for "
            f"tile={tile_id}, "
            f"visible_year={visible_year}: "
            f"{zero_sources}. "
            f"Diagnostic: {diagnostic_path}"
        )

    if STRICT_ABORT_IF_COMBINED_ALL_NODATA and not combined_has_valid_data:
        raise RuntimeError(
            "Exact preflight confirmed zero "
            "combined valid coverage for "
            f"tile={tile_id}, "
            f"visible_year={visible_year}. "
            f"Diagnostic: {diagnostic_path}"
        )

    return summary


def aggregate_input_preflight_diagnostics() -> None:
    rows: list[dict[str, Any]] = []

    for path in sorted(INPUT_DIAGNOSTICS_ROOT.rglob("*_input_preflight.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        source_valid = data.get(
            "source_valid_pixels",
            {},
        )

        source_tested = data.get(
            "source_tested_pixels",
            {},
        )

        rows.append(
            {
                "tile_id": (data.get("tile_id")),
                "zone": (data.get("zone")),
                "visible_year": (data.get("visible_year")),
                "status": (data.get("status")),
                "ae_source_zone": (data.get("ae_source_zone")),
                "ae_pre_valid": (source_valid.get("ae_pre")),
                "ae_pre_tested": (source_tested.get("ae_pre")),
                "ae_post_valid": (source_valid.get("ae_post")),
                "ae_post_tested": (source_tested.get("ae_post")),
                "s2_pre_valid": (source_valid.get("s2_pre")),
                "s2_pre_tested": (source_tested.get("s2_pre")),
                "s2_post_valid": (source_valid.get("s2_post")),
                "s2_post_tested": (source_tested.get("s2_post")),
                "combined_valid": (data.get("combined_valid_pixels")),
                "combined_tested": (data.get("combined_tested_pixels")),
                "diagnostic_path": (str(path)),
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(
            INPUT_PREFLIGHT_SUMMARY_CSV,
            index=False,
        )

        logging.info(
            "Wrote national input preflight summary: %s",
            INPUT_PREFLIGHT_SUMMARY_CSV,
        )


def sample_tile_year(
    tile_id: str,
    zone: str,
    grid: GridDefinition,
    pair: dict[str, Any],
    positive_mask: np.ndarray,
    manual_mask: np.ndarray,
    ar5_mask: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    visible_year = int(pair["visible_year"])

    output_dir = SAMPLING_CACHE_ROOT / zone / tile_id / str(visible_year)

    success_path = output_dir / "_SUCCESS.json"

    if (
        SKIP_COMPLETED_TILE_YEARS
        and success_path.exists()
        and not REBUILD_SAMPLING_CACHE
    ):
        return (
            [],
            {
                "tile_id": tile_id,
                "zone": zone,
                "visible_year": (visible_year),
                "status": ("skipped_completed"),
            },
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
        + visible_year * 100000
        + sum(ord(character) for character in tile_id)
    )

    positive_count = int(np.count_nonzero(positive_mask > 0))

    manual_candidate_mask = (manual_mask > 0) & (positive_mask == 0)

    manual_available = int(np.count_nonzero(manual_candidate_mask))

    land_capacities = {
        code: int(np.count_nonzero((ar5_mask == code) & (positive_mask == 0)))
        for code in AR5_LAND_CLASSES
    }

    water_capacities = {
        code: int(np.count_nonzero((ar5_mask == code) & (positive_mask == 0)))
        for code in AR5_WATER_CLASSES
    }

    land_target_total = max(
        MIN_AR5_LAND_NEGATIVES_PER_TILE_YEAR,
        int(round(positive_count * AR5_LAND_NEGATIVE_TO_POSITIVE_RATIO)),
    )

    land_target_total = min(
        land_target_total,
        MAX_AR5_LAND_NEGATIVES_PER_TILE_YEAR,
        sum(land_capacities.values()),
    )

    land_targets = allocate_equal_with_capacity(
        land_target_total,
        land_capacities,
    )

    actual_land_target = int(sum(land_targets.values()))

    manual_target = min(
        manual_available,
        int(round(actual_land_target * MANUAL_NOCHANGE_TO_AR5_LAND_RATIO)),
    )

    water_target_total = min(
        sum(water_capacities.values()),
        int(round(manual_target * WATER_TO_MANUAL_NOCHANGE_RATIO)),
    )

    water_targets = allocate_equal_with_capacity(
        water_target_total,
        water_capacities,
    )

    selected_rows = []
    selected_cols = []
    selected_labels = []
    selected_sources = []

    (
        positive_rows,
        positive_cols,
    ) = select_all_coordinates(positive_mask > 0)

    if len(positive_rows):
        positive_values = positive_mask[
            positive_rows,
            positive_cols,
        ]

        positive_labels = np.where(
            positive_values == 1,
            CLEARCUT,
            URBAN,
        ).astype(LABEL_DTYPE)

        selected_rows.append(positive_rows)

        selected_cols.append(positive_cols)

        selected_labels.append(positive_labels)

        selected_sources.append(
            np.full(
                len(positive_rows),
                MANUAL_CHANGE_SOURCE,
                dtype=SOURCE_DTYPE,
            )
        )

    (
        manual_rows,
        manual_cols,
    ) = select_random_coordinates(
        manual_candidate_mask,
        manual_target,
        rng,
    )

    if len(manual_rows):
        selected_rows.append(manual_rows)

        selected_cols.append(manual_cols)

        selected_labels.append(
            np.full(
                len(manual_rows),
                NOCHANGE,
                dtype=LABEL_DTYPE,
            )
        )

        selected_sources.append(
            np.full(
                len(manual_rows),
                MANUAL_NOCHANGE_SOURCE,
                dtype=SOURCE_DTYPE,
            )
        )

    del manual_candidate_mask

    combined_targets = {
        **land_targets,
        **water_targets,
    }

    for (
        code,
        target,
    ) in combined_targets.items():
        code_mask = (ar5_mask == code) & (positive_mask == 0)

        (
            rows,
            cols,
        ) = select_random_coordinates(
            code_mask,
            target,
            rng,
        )

        del code_mask

        if len(rows):
            selected_rows.append(rows)

            selected_cols.append(cols)

            selected_labels.append(
                np.full(
                    len(rows),
                    NOCHANGE,
                    dtype=LABEL_DTYPE,
                )
            )

            selected_sources.append(
                np.full(
                    len(rows),
                    code,
                    dtype=SOURCE_DTYPE,
                )
            )

    if not selected_rows:
        no_sample_summary = {
            "tile_id": tile_id,
            "zone": zone,
            "visible_year": (visible_year),
            "status": "no_samples",
        }

        success_path.write_text(
            json.dumps(
                no_sample_summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        return (
            [],
            no_sample_summary,
        )

    rows = np.concatenate(selected_rows)

    cols = np.concatenate(selected_cols)

    labels = np.concatenate(selected_labels)

    sources = np.concatenate(selected_sources)

    block_columns = math.ceil(grid.width / RASTER_READ_BLOCK_SIZE)

    block_keys = (
        rows // RASTER_READ_BLOCK_SIZE * block_columns + cols // RASTER_READ_BLOCK_SIZE
    )

    order = np.argsort(
        block_keys,
        kind="stable",
    )

    rows = rows[order]

    cols = cols[order]

    labels = labels[order]

    sources = sources[order]

    block_keys = block_keys[order]

    (
        unique_keys,
        key_starts,
    ) = np.unique(
        block_keys,
        return_index=True,
    )

    key_ends = np.r_[
        key_starts[1:],
        len(block_keys),
    ]

    buffer = SamplePartBuffer(
        output_dir=output_dir,
        tile_id=tile_id,
        zone=zone,
        visible_year=visible_year,
    )

    invalid_selected_rows = 0

    selected_source_valid_counts = {
        "ae_pre": 0,
        "ae_post": 0,
        "s2_pre": 0,
        "s2_post": 0,
        "combined": 0,
    }

    with ExitStack() as stack:
        opened = open_pair_sources(
            stack=stack,
            grid=grid,
            pair=pair,
        )

        s2_pre_src = opened["s2_pre_src"]

        s2_post_src = opened["s2_post_src"]

        ae_pre_src = opened["ae_pre_src"]

        ae_post_src = opened["ae_post_src"]

        s2_pre_indexes = opened["s2_pre_indexes"]

        s2_post_indexes = opened["s2_post_indexes"]

        for (
            key,
            start,
            end,
        ) in zip(
            unique_keys,
            key_starts,
            key_ends,
        ):
            block_row = int(key) // block_columns

            block_col = int(key) % block_columns

            row_off = block_row * RASTER_READ_BLOCK_SIZE

            col_off = block_col * RASTER_READ_BLOCK_SIZE

            window = Window(
                col_off=col_off,
                row_off=row_off,
                width=min(
                    RASTER_READ_BLOCK_SIZE,
                    grid.width - col_off,
                ),
                height=min(
                    RASTER_READ_BLOCK_SIZE,
                    grid.height - row_off,
                ),
            )

            global_rows = rows[start:end]

            global_cols = cols[start:end]

            local_rows = (global_rows - int(window.row_off)).astype(np.int32)

            local_cols = (global_cols - int(window.col_off)).astype(np.int32)

            block_labels = labels[start:end]

            block_sources = sources[start:end]

            ae_pre = ae_pre_src.read(
                window=window,
                out_dtype="float32",
            )

            ae_post = ae_post_src.read(
                window=window,
                out_dtype="float32",
            )

            s2_pre = s2_pre_src.read(
                indexes=(s2_pre_indexes),
                window=window,
                out_dtype="float32",
            )

            s2_post = s2_post_src.read(
                indexes=(s2_post_indexes),
                window=window,
                out_dtype="float32",
            )

            ae_pre_valid = valid_pixel_rows(
                [
                    (
                        ae_pre,
                        ae_pre_src.nodata,
                    )
                ],
                local_rows,
                local_cols,
            )

            ae_post_valid = valid_pixel_rows(
                [
                    (
                        ae_post,
                        ae_post_src.nodata,
                    )
                ],
                local_rows,
                local_cols,
            )

            s2_pre_valid = valid_pixel_rows(
                [
                    (
                        s2_pre,
                        s2_pre_src.nodata,
                    )
                ],
                local_rows,
                local_cols,
            )

            s2_post_valid = valid_pixel_rows(
                [
                    (
                        s2_post,
                        s2_post_src.nodata,
                    )
                ],
                local_rows,
                local_cols,
            )

            valid = ae_pre_valid & ae_post_valid & s2_pre_valid & s2_post_valid

            selected_source_valid_counts["ae_pre"] += int(ae_pre_valid.sum())

            selected_source_valid_counts["ae_post"] += int(ae_post_valid.sum())

            selected_source_valid_counts["s2_pre"] += int(s2_pre_valid.sum())

            selected_source_valid_counts["s2_post"] += int(s2_post_valid.sum())

            selected_source_valid_counts["combined"] += int(valid.sum())

            invalid_selected_rows += int(np.count_nonzero(~valid))

            if not np.any(valid):
                continue

            valid_local_rows = local_rows[valid]

            valid_local_cols = local_cols[valid]

            valid_global_rows = global_rows[valid]

            valid_global_cols = global_cols[valid]

            X = build_combined_features(
                ae_pre=ae_pre,
                ae_post=ae_post,
                s2_pre=s2_pre,
                s2_post=s2_post,
                rows=valid_local_rows,
                cols=valid_local_cols,
                visible_year=visible_year,
            )

            groups = national_spatial_groups(
                grid=grid,
                rows=valid_global_rows,
                cols=valid_global_cols,
                zone=zone,
            )

            valid_labels = block_labels[valid]

            valid_sources = block_sources[valid]

            years = np.full(
                len(valid_labels),
                visible_year,
                dtype=YEAR_DTYPE,
            )

            buffer.add(
                X=X,
                y=valid_labels,
                groups=groups,
                sources=valid_sources,
                years=years,
            )

            del ae_pre
            del ae_post
            del s2_pre
            del s2_post
            del valid
            del X
            del groups
            del years

    manifest_rows = buffer.finish()

    actual = defaultdict(int)

    for manifest_row in manifest_rows:
        for key in [
            "clearcut",
            "urban",
            "manual_nochange",
            "ar5_11",
            "ar5_12",
            "ar5_21",
            "ar5_22",
            "ar5_81",
            "ar5_82",
        ]:
            actual[key] += int(manifest_row[key])

    valid_output_rows = int(sum(row["rows"] for row in manifest_rows))

    selected_samples_invalid = len(rows) > 0 and (
        selected_source_valid_counts["ae_pre"] == 0
        or selected_source_valid_counts["ae_post"] == 0
        or selected_source_valid_counts["s2_pre"] == 0
        or selected_source_valid_counts["s2_post"] == 0
        or selected_source_valid_counts["combined"] == 0
        or valid_output_rows == 0
    )

    if selected_samples_invalid and STRICT_ABORT_IF_SELECTED_SAMPLES_ALL_INVALID:
        raise RuntimeError(
            "Selected training samples "
            "did not survive source "
            "validity checks for "
            f"tile={tile_id}, "
            f"visible_year={visible_year}. "
            "Selected source-valid counts="
            f"{selected_source_valid_counts}, "
            "valid output rows="
            f"{valid_output_rows}."
        )

    if selected_samples_invalid and SKIP_TILE_YEAR_IF_SELECTED_SAMPLES_ALL_INVALID:
        logging.warning(
            "Skipping invalid sampled "
            "tile/year without aborting "
            "national run: tile=%s "
            "visible_year=%d source_valid=%s "
            "valid_output_rows=%d",
            tile_id,
            visible_year,
            selected_source_valid_counts,
            valid_output_rows,
        )

    summary = {
        "tile_id": tile_id,
        "zone": zone,
        "visible_year": (visible_year),
        "status": (
            "skipped_all_selected_samples_invalid"
            if (
                selected_samples_invalid
                and SKIP_TILE_YEAR_IF_SELECTED_SAMPLES_ALL_INVALID
            )
            else "completed"
        ),
        "selected_before_validity_filter": (len(rows)),
        "invalid_selected_rows": (invalid_selected_rows),
        "valid_output_rows": (valid_output_rows),
        "selected_ae_pre_valid": (selected_source_valid_counts["ae_pre"]),
        "selected_ae_post_valid": (selected_source_valid_counts["ae_post"]),
        "selected_s2_pre_valid": (selected_source_valid_counts["s2_pre"]),
        "selected_s2_post_valid": (selected_source_valid_counts["s2_post"]),
        "selected_combined_valid": (selected_source_valid_counts["combined"]),
        "ae_source_zone": (pair["ae_source_zone"]),
        "s2_pre_band_indexes": (s2_pre_indexes),
        "s2_post_band_indexes": (s2_post_indexes),
        "s2_band_names": (S2_BAND_NAMES),
        "positive_mask_pixels": (positive_count),
        "land_target": (actual_land_target),
        "manual_target": (manual_target),
        "water_target": int(sum(water_targets.values())),
        "parts": (len(manifest_rows)),
        **actual,
    }

    success_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    gc.collect()

    return (
        manifest_rows,
        summary,
    )


def task_grid(
    task: dict[str, Any],
) -> GridDefinition:
    data = task["reference_grid"]

    return GridDefinition(
        crs_wkt=(data["crs_wkt"]),
        transform_values=tuple(data["transform_values"]),
        width=int(data["width"]),
        height=int(data["height"]),
        bounds_values=tuple(data["bounds_values"]),
    )


def sample_tile_worker(
    task: dict[str, Any],
) -> dict[str, Any]:
    tile_id = str(task["tile_id"])

    zone = str(task["zone"])

    grid = task_grid(task)

    (
        tile_polygons,
        tile_ar5,
    ) = subset_tile_vectors(
        zone=zone,
        grid=grid,
    )

    (
        manual_mask,
        ar5_mask,
    ) = build_static_tile_masks(
        zone=zone,
        tile_id=tile_id,
        grid=grid,
        tile_polygons=tile_polygons,
        tile_ar5=tile_ar5,
    )

    manifest_rows = []
    summary_rows = []

    for pair in task["pairs"]:
        visible_year = int(pair["visible_year"])

        preflight_summary = preflight_tile_year_sources(
            tile_id=tile_id,
            zone=zone,
            grid=grid,
            pair=pair,
        )

        if SKIP_TILE_YEAR_IF_PREFLIGHT_INVALID and preflight_summary.get(
            "status"
        ) not in {
            "passed",
            "disabled",
        }:
            logging.warning(
                "Skipping tile/year after "
                "failed input preflight: "
                "tile=%s visible_year=%d "
                "status=%s",
                tile_id,
                visible_year,
                preflight_summary.get("status"),
            )

            summary_rows.append(
                {
                    "tile_id": tile_id,
                    "zone": zone,
                    "visible_year": (visible_year),
                    "status": ("skipped_invalid_preflight"),
                    "preflight_status": (preflight_summary.get("status")),
                    "valid_output_rows": 0,
                    "parts": 0,
                }
            )

            continue

        positive_mask = build_positive_mask(
            zone=zone,
            tile_id=tile_id,
            visible_year=visible_year,
            grid=grid,
            tile_polygons=tile_polygons,
        )

        (
            year_manifest,
            year_summary,
        ) = sample_tile_year(
            tile_id=tile_id,
            zone=zone,
            grid=grid,
            pair=pair,
            positive_mask=positive_mask,
            manual_mask=manual_mask,
            ar5_mask=ar5_mask,
        )

        manifest_rows.extend(year_manifest)

        summary_rows.append(year_summary)

        del positive_mask

    del manual_mask
    del ar5_mask
    del tile_polygons
    del tile_ar5

    gc.collect()

    return {
        "tile_id": tile_id,
        "zone": zone,
        "manifest_rows": (manifest_rows),
        "summary_rows": (summary_rows),
    }


def reconstruct_sampling_tables() -> None:
    rows = []

    for x_path in sorted(SAMPLING_CACHE_ROOT.rglob("part_*_X.npy")):
        prefix = str(x_path)[:-6]

        y_path = Path(prefix + "_y.npy")

        sources_path = Path(prefix + "_sources.npy")

        X = np.load(
            x_path,
            mmap_mode="r",
        )

        y = np.load(
            y_path,
            mmap_mode="r",
        )

        sources = np.load(
            sources_path,
            mmap_mode="r",
        )

        relative = x_path.relative_to(SAMPLING_CACHE_ROOT)

        zone = relative.parts[0]

        tile_id = relative.parts[1]

        visible_year = int(relative.parts[2])

        part_number = int(
            re.search(
                r"part_(\d+)_X",
                x_path.name,
            ).group(1)
        )

        rows.append(
            {
                "tile_id": (tile_id),
                "zone": (zone),
                "visible_year": (visible_year),
                "part_number": (part_number),
                "rows": (len(y)),
                "features": (X.shape[1]),
                "x_path": (str(x_path)),
                "y_path": (str(y_path)),
                "groups_path": (prefix + "_groups.npy"),
                "sources_path": (prefix + "_sources.npy"),
                "years_path": (prefix + "_years.npy"),
                "clearcut": int(np.count_nonzero(y == CLEARCUT)),
                "urban": int(np.count_nonzero(y == URBAN)),
                "nochange": int(np.count_nonzero(y == NOCHANGE)),
                "manual_nochange": int(
                    np.count_nonzero(sources == MANUAL_NOCHANGE_SOURCE)
                ),
                "ar5_11": int(np.count_nonzero(sources == 11)),
                "ar5_12": int(np.count_nonzero(sources == 12)),
                "ar5_21": int(np.count_nonzero(sources == 21)),
                "ar5_22": int(np.count_nonzero(sources == 22)),
                "ar5_81": int(np.count_nonzero(sources == 81)),
                "ar5_82": int(np.count_nonzero(sources == 82)),
            }
        )

        del X
        del y
        del sources

    if rows:
        manifest = pd.DataFrame(rows)

        manifest.to_csv(
            SAMPLING_MANIFEST_CSV,
            index=False,
        )

        summary = manifest.groupby(
            [
                "zone",
                "tile_id",
                "visible_year",
            ],
            as_index=False,
        )[
            [
                "rows",
                "clearcut",
                "urban",
                "nochange",
                "manual_nochange",
                "ar5_11",
                "ar5_12",
                "ar5_21",
                "ar5_22",
                "ar5_81",
                "ar5_82",
            ]
        ].sum()

        summary.to_csv(
            SAMPLING_SUMMARY_CSV,
            index=False,
        )


def run_parallel_sampling(
    tasks: list[dict[str, Any]],
) -> None:
    if REBUILD_SAMPLING_CACHE and SAMPLING_CACHE_ROOT.exists():
        shutil.rmtree(SAMPLING_CACHE_ROOT)

    SAMPLING_CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.info(
        "Starting per-tile EPSG:25833 "
        "sampling: tasks=%d workers=%d "
        "warp_threads_per_worker=%d",
        len(tasks),
        SAMPLING_WORKERS,
        GDAL_WARP_THREADS_PER_WORKER,
    )

    start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=(SAMPLING_WORKERS),
        initializer=(initialise_sampling_worker),
    ) as executor:
        futures = {
            executor.submit(
                sample_tile_worker,
                task,
            ): (
                task["zone"],
                task["tile_id"],
            )
            for task in tasks
        }

        completed = 0

        for future in as_completed(futures):
            (
                zone,
                tile_id,
            ) = futures[future]

            try:
                result = future.result()
            except Exception:
                logging.exception(
                    "Sampling failed for %s %s",
                    zone,
                    tile_id,
                )
                raise

            completed += 1

            logging.info(
                "Sampling tile %d/%d completed: %s %s, years=%d, parts=%d",
                completed,
                len(tasks),
                zone,
                tile_id,
                len(result["summary_rows"]),
                len(result["manifest_rows"]),
            )

    reconstruct_sampling_tables()
    aggregate_input_preflight_diagnostics()

    logging.info(
        "Sampling completed in %.1f seconds.",
        time.time() - start_time,
    )


def discover_cache_parts() -> list[CachePart]:
    x_paths = sorted(SAMPLING_CACHE_ROOT.rglob("part_*_X.npy"))

    if not x_paths:
        raise RuntimeError("No AE+S2 sample cache parts were found.")

    parts = []

    for (
        part_id,
        x_path,
    ) in enumerate(x_paths):
        prefix = str(x_path)[:-6]

        y_path = Path(prefix + "_y.npy")

        groups_path = Path(prefix + "_groups.npy")

        sources_path = Path(prefix + "_sources.npy")

        years_path = Path(prefix + "_years.npy")

        required = [
            y_path,
            groups_path,
            sources_path,
            years_path,
        ]

        if not all(path.exists() for path in required):
            raise RuntimeError(f"Incomplete cache part: {x_path}")

        X = np.load(
            x_path,
            mmap_mode="r",
        )

        y = np.load(
            y_path,
            mmap_mode="r",
        )

        groups = np.load(
            groups_path,
            mmap_mode="r",
        )

        sources = np.load(
            sources_path,
            mmap_mode="r",
        )

        years = np.load(
            years_path,
            mmap_mode="r",
        )

        if not (len(X) == len(y) == len(groups) == len(sources) == len(years)):
            raise RuntimeError(f"Cache array length mismatch: {x_path}")

        relative = x_path.relative_to(SAMPLING_CACHE_ROOT)

        parts.append(
            CachePart(
                part_id=part_id,
                x_path=x_path,
                y_path=y_path,
                groups_path=groups_path,
                sources_path=sources_path,
                years_path=years_path,
                zone=(relative.parts[0]),
                tile_id=(relative.parts[1]),
                visible_year=int(relative.parts[2]),
                rows=len(y),
                features=(X.shape[1]),
            )
        )

        del X
        del y
        del groups
        del sources
        del years

    feature_counts = {part.features for part in parts}

    if len(feature_counts) != 1:
        raise RuntimeError(
            f"Feature counts differ across cache parts: {feature_counts}"
        )

    logging.info(
        "Discovered cache: parts=%d rows=%d features=%d zones=%s",
        len(parts),
        sum(part.rows for part in parts),
        next(iter(feature_counts)),
        sorted({part.zone for part in parts}),
    )

    return parts


def inventory_for_parts(
    parts: list[CachePart],
) -> pd.DataFrame:
    rows = []

    for part in parts:
        y = np.load(
            part.y_path,
            mmap_mode="r",
        )

        sources = np.load(
            part.sources_path,
            mmap_mode="r",
        )

        for class_code in [
            CLEARCUT,
            URBAN,
            NOCHANGE,
        ]:
            for source_code in np.unique(sources):
                count = int(
                    np.count_nonzero((y == class_code) & (sources == source_code))
                )

                if count:
                    rows.append(
                        {
                            "part_id": (part.part_id),
                            "zone": (part.zone),
                            "tile_id": (part.tile_id),
                            "visible_year": (part.visible_year),
                            "class_code": (class_code),
                            "class_name": (CLASS_NAMES[class_code]),
                            "source_code": int(source_code),
                            "source_name": (
                                SOURCE_NAMES.get(
                                    int(source_code),
                                    (f"unknown_{int(source_code)}"),
                                )
                            ),
                            "count": (count),
                        }
                    )

        del y
        del sources

    return pd.DataFrame(rows)


def build_group_inventory(
    parts: list[CachePart],
) -> pd.DataFrame:
    counts: dict[
        int,
        np.ndarray,
    ] = defaultdict(
        lambda: np.zeros(
            3,
            dtype=np.int64,
        )
    )

    for part in parts:
        y = np.load(
            part.y_path,
            mmap_mode="r",
        )

        groups = np.load(
            part.groups_path,
            mmap_mode="r",
        )

        for start in range(
            0,
            len(y),
            METADATA_SCAN_ROWS,
        ):
            end = min(
                start + METADATA_SCAN_ROWS,
                len(y),
            )

            y_chunk = np.asarray(y[start:end])

            group_chunk = np.asarray(groups[start:end])

            for group in np.unique(group_chunk):
                mask = group_chunk == group

                counts[int(group)] += np.bincount(
                    y_chunk[mask].astype(np.int64),
                    minlength=3,
                )[:3]

        del y
        del groups

    rows = []

    for (
        group,
        class_counts,
    ) in counts.items():
        (
            clearcut_count,
            urban_count,
            nochange_count,
        ) = class_counts.tolist()

        if clearcut_count > 0 and urban_count > 0:
            stratum = "both_changes"

        elif urban_count > 0:
            stratum = "urban"

        elif clearcut_count > 0:
            stratum = "clearcut"

        else:
            stratum = "nochange_only"

        rows.append(
            {
                "spatial_group": (group),
                "clearcut_count": (clearcut_count),
                "urban_count": (urban_count),
                "nochange_count": (nochange_count),
                "stratum": (stratum),
            }
        )

    return pd.DataFrame(rows).sort_values("spatial_group").reset_index(drop=True)


def assign_group_folds(
    group_inventory: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if len(group_inventory) < N_CV_FOLDS:
        raise RuntimeError(
            f"Only {len(group_inventory)} spatial groups for {N_CV_FOLDS} folds."
        )

    table = group_inventory.copy()

    table["fold"] = -1

    stratum_counts = table["stratum"].value_counts()

    can_stratify = len(stratum_counts) > 1 and int(stratum_counts.min()) >= N_CV_FOLDS

    dummy = np.zeros(
        (
            len(table),
            1,
        ),
        dtype=np.uint8,
    )

    if can_stratify:
        splitter = StratifiedKFold(
            n_splits=(N_CV_FOLDS),
            shuffle=True,
            random_state=(RANDOM_SEED),
        )

        split_iterator = splitter.split(
            dummy,
            table["stratum"].to_numpy(),
        )

    else:
        logging.warning(
            "Not every spatial stratum "
            "has %d groups. Using shuffled "
            "KFold on spatial groups.",
            N_CV_FOLDS,
        )

        splitter = KFold(
            n_splits=N_CV_FOLDS,
            shuffle=True,
            random_state=(RANDOM_SEED),
        )

        split_iterator = splitter.split(dummy)

    for (
        fold,
        (
            _,
            validation_rows,
        ),
    ) in enumerate(split_iterator):
        table.loc[
            validation_rows,
            "fold",
        ] = fold

    if (table["fold"] < 0).any():
        raise RuntimeError("Some spatial groups were not assigned to a fold.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        output_path,
        index=False,
    )

    return table


def make_references(
    part_id: int,
    rows: np.ndarray,
) -> np.ndarray:
    references = np.empty(
        len(rows),
        dtype=REFERENCE_DTYPE,
    )

    references["part"] = part_id

    references["row"] = rows.astype(
        np.int32,
        copy=False,
    )

    return references


def collect_training_references(
    parts: list[CachePart],
    excluded_groups: set[int],
    seed: int,
) -> dict[str, np.ndarray]:
    clearcut_parts = []
    urban_parts = []
    negative_parts = []

    excluded_array = np.asarray(
        sorted(excluded_groups),
        dtype=GROUP_DTYPE,
    )

    for (
        local_part_id,
        part,
    ) in enumerate(parts):
        y = np.load(
            part.y_path,
            mmap_mode="r",
        )

        groups = np.load(
            part.groups_path,
            mmap_mode="r",
        )

        training_mask = np.ones(
            len(y),
            dtype=bool,
        )

        if len(excluded_array):
            training_mask &= ~np.isin(
                groups,
                excluded_array,
                assume_unique=False,
            )

        clearcut_rows = np.flatnonzero(training_mask & (y == CLEARCUT))

        urban_rows = np.flatnonzero(training_mask & (y == URBAN))

        negative_rows = np.flatnonzero(training_mask & (y == NOCHANGE))

        if len(clearcut_rows):
            clearcut_parts.append(
                make_references(
                    local_part_id,
                    clearcut_rows,
                )
            )

        if len(urban_rows):
            urban_parts.append(
                make_references(
                    local_part_id,
                    urban_rows,
                )
            )

        if len(negative_rows):
            negative_parts.append(
                make_references(
                    local_part_id,
                    negative_rows,
                )
            )

        del y
        del groups

    if not clearcut_parts or not urban_parts or not negative_parts:
        raise RuntimeError("Training scope is missing one or more target classes.")

    references = {
        "clearcut": (np.concatenate(clearcut_parts)),
        "urban": (np.concatenate(urban_parts)),
        "negative": (np.concatenate(negative_parts)),
    }

    rng = np.random.default_rng(seed)

    rng.shuffle(references["clearcut"])

    rng.shuffle(references["urban"])

    rng.shuffle(references["negative"])

    return references


class ReferenceCycler:
    def __init__(
        self,
        references: np.ndarray,
        seed: int,
    ) -> None:
        self.references = references

        self.rng = np.random.default_rng(seed)

        self.position = 0

    def take(
        self,
        count: int,
    ) -> np.ndarray:
        pieces = []
        remaining = count

        while remaining > 0:
            available = len(self.references) - self.position

            if available == 0:
                self.rng.shuffle(self.references)

                self.position = 0

                available = len(self.references)

            take_count = min(
                remaining,
                available,
            )

            pieces.append(
                self.references[self.position : self.position + take_count].copy()
            )

            self.position += take_count

            remaining -= take_count

        return np.concatenate(pieces)


def load_features_from_references(
    references: np.ndarray,
    parts: list[CachePart],
    feature_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X_out = np.empty(
        (
            len(references),
            feature_count,
        ),
        dtype=FEATURE_DTYPE,
    )

    y_out = np.empty(
        len(references),
        dtype=LABEL_DTYPE,
    )

    sources_out = np.empty(
        len(references),
        dtype=SOURCE_DTYPE,
    )

    for part_id in np.unique(references["part"]):
        positions = np.flatnonzero(references["part"] == part_id)

        part = parts[int(part_id)]

        rows = references["row"][positions]

        X = np.load(
            part.x_path,
            mmap_mode="r",
        )

        y = np.load(
            part.y_path,
            mmap_mode="r",
        )

        sources = np.load(
            part.sources_path,
            mmap_mode="r",
        )

        X_out[positions] = X[rows]

        y_out[positions] = y[rows]

        sources_out[positions] = sources[rows]

        del X
        del y
        del sources

    return (
        X_out,
        y_out,
        sources_out,
    )


def rows_for_feature_budget(
    feature_count: int,
) -> int:
    bytes_budget = FEATURE_BATCH_MEMORY_MB * 1024**2

    bytes_per_row = feature_count * np.dtype(FEATURE_DTYPE).itemsize

    rows = int(bytes_budget // bytes_per_row)

    if rows < 50000:
        raise ValueError("FEATURE_BATCH_MEMORY_MB is too small.")

    return rows


def make_batch_plan(
    references: dict[
        str,
        np.ndarray,
    ],
    feature_count: int,
) -> pd.DataFrame:
    total_capacity = rows_for_feature_budget(feature_count)

    positive_capacity = max(
        2,
        int(round(total_capacity * POSITIVE_FRACTION_PER_BATCH)),
    )

    negative_capacity = total_capacity - positive_capacity

    batch_count = max(
        1,
        math.ceil(len(references["negative"]) / negative_capacity),
    )

    base_trees = TARGET_TREES // batch_count

    tree_remainder = TARGET_TREES % batch_count

    rows = []

    negative_remaining = len(references["negative"])

    for batch_index in range(batch_count):
        batches_left = batch_count - batch_index

        negative_count = min(
            negative_capacity,
            math.ceil(negative_remaining / batches_left),
        )

        positive_count = max(
            2,
            int(
                round(
                    negative_count
                    * POSITIVE_FRACTION_PER_BATCH
                    / max(
                        1e-9,
                        (1.0 - POSITIVE_FRACTION_PER_BATCH),
                    )
                )
            ),
        )

        clearcut_count = max(
            1,
            int(round(positive_count * CLEARCUT_SHARE_OF_POSITIVES)),
        )

        urban_count = max(
            1,
            (positive_count - clearcut_count),
        )

        trees_to_add = base_trees + (1 if (batch_index < tree_remainder) else 0)

        trees_to_add = min(
            MAX_TREES_PER_BATCH,
            max(
                1,
                trees_to_add,
            ),
        )

        rows.append(
            {
                "batch": (batch_index + 1),
                "negative": (negative_count),
                "clearcut": (clearcut_count),
                "urban": (urban_count),
                "trees_to_add": (trees_to_add),
            }
        )

        negative_remaining -= negative_count

    plan = pd.DataFrame(rows)

    missing_trees = TARGET_TREES - int(plan["trees_to_add"].sum())

    while missing_trees > 0:
        changed = False

        for index in plan.index:
            if missing_trees <= 0:
                break

            if (
                plan.loc[
                    index,
                    "trees_to_add",
                ]
                < MAX_TREES_PER_BATCH
            ):
                plan.loc[
                    index,
                    "trees_to_add",
                ] += 1

                missing_trees -= 1
                changed = True

        if not changed:
            break

    return plan


def create_forest(
    seed: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=0,
        warm_start=True,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=(RF_MIN_SAMPLES_LEAF),
        max_features=(RF_MAX_FEATURES),
        bootstrap=(RF_BOOTSTRAP),
        class_weight=(RF_CLASS_WEIGHT),
        n_jobs=(RF_N_JOBS),
        random_state=seed,
        verbose=(RF_VERBOSE),
    )


def train_streamed_forest(
    parts: list[CachePart],
    excluded_groups: set[int],
    feature_count: int,
    seed: int,
    fold_name: str,
) -> tuple[
    RandomForestClassifier,
    dict[str, Any],
    list[dict[str, Any]],
]:
    references = collect_training_references(
        parts=parts,
        excluded_groups=(excluded_groups),
        seed=seed,
    )

    plan = make_batch_plan(
        references=references,
        feature_count=feature_count,
    )

    clearcut_cycler = ReferenceCycler(
        references["clearcut"],
        seed + 101,
    )

    urban_cycler = ReferenceCycler(
        references["urban"],
        seed + 202,
    )

    model = create_forest(seed)

    negative_position = 0

    batch_rows = []

    start_time = time.time()

    for _, plan_row in plan.iterrows():
        batch_number = int(plan_row["batch"])

        negative_count = int(plan_row["negative"])

        clearcut_count = int(plan_row["clearcut"])

        urban_count = int(plan_row["urban"])

        trees_to_add = int(plan_row["trees_to_add"])

        negative_refs = references["negative"][
            negative_position : negative_position + negative_count
        ]

        clearcut_refs = clearcut_cycler.take(clearcut_count)

        urban_refs = urban_cycler.take(urban_count)

        batch_refs = np.concatenate(
            [
                negative_refs,
                clearcut_refs,
                urban_refs,
            ]
        )

        rng = np.random.default_rng(seed + batch_number * 1009)

        rng.shuffle(batch_refs)

        (
            X_batch,
            y_batch,
            source_batch,
        ) = load_features_from_references(
            references=batch_refs,
            parts=parts,
            feature_count=(feature_count),
        )

        class_counts = np.bincount(
            y_batch.astype(np.int64),
            minlength=3,
        )

        if np.any(class_counts == 0):
            raise RuntimeError(
                f"{fold_name} batch "
                f"{batch_number} misses "
                "a class: "
                f"{class_counts.tolist()}"
            )

        model.n_estimators += trees_to_add

        batch_start = time.time()

        model.fit(
            X_batch,
            y_batch,
        )

        batch_seconds = time.time() - batch_start

        negative_position += negative_count

        source_counts = {
            source_code: int(np.count_nonzero(source_batch == source_code))
            for source_code in SOURCE_NAMES
        }

        row = {
            "fold": fold_name,
            "batch": batch_number,
            "trees_added": (trees_to_add),
            "total_trees": (model.n_estimators),
            "batch_rows": (len(y_batch)),
            "clearcut_rows": int(class_counts[CLEARCUT]),
            "urban_rows": int(class_counts[URBAN]),
            "nochange_rows": int(class_counts[NOCHANGE]),
            "manual_nochange": (source_counts[1]),
            "ar5_11": (source_counts[11]),
            "ar5_12": (source_counts[12]),
            "ar5_21": (source_counts[21]),
            "ar5_22": (source_counts[22]),
            "ar5_81": (source_counts[81]),
            "ar5_82": (source_counts[82]),
            "batch_seconds": (batch_seconds),
            "elapsed_seconds": (time.time() - start_time),
        }

        batch_rows.append(row)

        logging.info(
            "%s batch %d/%d | "
            "trees +%d => %d | "
            "rows=%d class=[%d,%d,%d] "
            "manual=%d landAR5=%d "
            "water=%d %.1fs",
            fold_name,
            batch_number,
            len(plan),
            trees_to_add,
            model.n_estimators,
            len(y_batch),
            class_counts[0],
            class_counts[1],
            class_counts[2],
            source_counts[1],
            (
                source_counts[11]
                + source_counts[12]
                + source_counts[21]
                + source_counts[22]
            ),
            (source_counts[81] + source_counts[82]),
            batch_seconds,
        )

        del negative_refs
        del clearcut_refs
        del urban_refs
        del batch_refs
        del X_batch
        del y_batch
        del source_batch

        gc.collect()

    if negative_position != len(references["negative"]):
        raise RuntimeError("Not every cached negative was consumed.")

    summary = {
        "batches": (len(plan)),
        "trees": (model.n_estimators),
        "negative_rows_used": (negative_position),
        "clearcut_available": (len(references["clearcut"])),
        "urban_available": (len(references["urban"])),
        "fit_seconds": (time.time() - start_time),
    }

    return (
        model,
        summary,
        batch_rows,
    )


def evaluate_streaming(
    model: RandomForestClassifier,
    parts: list[CachePart],
    validation_groups: set[int],
) -> tuple[
    np.ndarray,
    int,
]:
    total_confusion = np.zeros(
        (
            3,
            3,
        ),
        dtype=np.int64,
    )

    validation_rows = 0

    validation_array = np.asarray(
        sorted(validation_groups),
        dtype=GROUP_DTYPE,
    )

    for part in parts:
        X = np.load(
            part.x_path,
            mmap_mode="r",
        )

        y = np.load(
            part.y_path,
            mmap_mode="r",
        )

        groups = np.load(
            part.groups_path,
            mmap_mode="r",
        )

        for start in range(
            0,
            len(y),
            PREDICTION_ROWS,
        ):
            end = min(
                start + PREDICTION_ROWS,
                len(y),
            )

            group_chunk = np.asarray(groups[start:end])

            mask = np.isin(
                group_chunk,
                validation_array,
                assume_unique=False,
            )

            selected = np.flatnonzero(mask)

            if len(selected) == 0:
                continue

            X_chunk = np.asarray(
                X[start:end][selected],
                dtype=FEATURE_DTYPE,
                order="C",
            )

            y_true = np.asarray(
                y[start:end][selected],
                dtype=LABEL_DTYPE,
            )

            y_pred = model.predict(X_chunk).astype(LABEL_DTYPE)

            total_confusion += confusion_matrix(
                y_true,
                y_pred,
                labels=[
                    CLEARCUT,
                    URBAN,
                    NOCHANGE,
                ],
            )

            validation_rows += len(y_true)

            del X_chunk
            del y_true
            del y_pred

        del X
        del y
        del groups

        gc.collect()

    return (
        total_confusion,
        validation_rows,
    )


def metrics_from_confusion(
    matrix: np.ndarray,
) -> dict[str, float]:
    total = float(matrix.sum())

    result = {"accuracy": (float(np.trace(matrix)) / total if total else 0.0)}

    precisions = []
    recalls = []
    f1_values = []

    for (
        class_code,
        class_name,
    ) in enumerate(CLASS_NAMES):
        tp = float(
            matrix[
                class_code,
                class_code,
            ]
        )

        fp = float(
            matrix[
                :,
                class_code,
            ].sum()
            - tp
        )

        fn = float(
            matrix[
                class_code,
                :,
            ].sum()
            - tp
        )

        precision = tp / (tp + fp) if (tp + fp) else 0.0

        recall = tp / (tp + fn) if (tp + fn) else 0.0

        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        result[f"{class_name}_precision"] = precision

        result[f"{class_name}_recall"] = recall

        result[f"{class_name}_f1"] = f1

        precisions.append(precision)

        recalls.append(recall)

        f1_values.append(f1)

    result["macro_precision"] = float(np.mean(precisions))

    result["macro_recall"] = float(np.mean(recalls))

    result["macro_f1"] = float(np.mean(f1_values))

    return result


def report_from_confusion(
    matrix: np.ndarray,
) -> str:
    metrics = metrics_from_confusion(matrix)

    supports = matrix.sum(axis=1)

    lines = [("class          precision    recall        f1      support")]

    for (
        class_name,
        support,
    ) in zip(
        CLASS_NAMES,
        supports,
    ):
        lines.append(
            f"{class_name:<14s} "
            f"{metrics[f'{class_name}_precision']:9.4f} "
            f"{metrics[f'{class_name}_recall']:9.4f} "
            f"{metrics[f'{class_name}_f1']:9.4f} "
            f"{int(support):12d}"
        )

    return "\n".join(lines)


def scope_paths(
    scope_name: str,
) -> dict[str, Path]:
    root = MODEL_ROOT / scope_name

    metrics = METRICS_ROOT / scope_name

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "model_root": root,
        "metrics_root": metrics,
        "group_folds": (metrics / "spatial_group_fold_assignment.csv"),
        "cv_metrics": (metrics / "spatial_cv_metrics.csv"),
        "cv_confusion": (metrics / "spatial_cv_confusion_matrix.csv"),
        "cv_report": (metrics / "spatial_cv_classification_report.txt"),
        "batch_log": (metrics / "training_batch_composition.csv"),
        "best_model": (root / "rf_best_spatial_fold.joblib"),
        "final_model": (root / "rf_final.joblib"),
        "inventory": (metrics / "source_cache_inventory.csv"),
        "status": (metrics / "status.json"),
    }


def scope_is_trainable(
    scope_name: str,
    parts: list[CachePart],
) -> tuple[
    bool,
    dict[str, Any],
]:
    inventory = inventory_for_parts(parts)

    if not inventory.empty:
        class_counts = inventory.groupby("class_code")["count"].sum()
    else:
        class_counts = pd.Series(dtype=np.int64)

    group_inventory = build_group_inventory(parts)

    status = {
        "scope": (scope_name),
        "parts": (len(parts)),
        "rows": (sum(part.rows for part in parts)),
        "spatial_groups": (len(group_inventory)),
        "clearcut": int(
            class_counts.get(
                CLEARCUT,
                0,
            )
        ),
        "urban": int(
            class_counts.get(
                URBAN,
                0,
            )
        ),
        "nochange": int(
            class_counts.get(
                NOCHANGE,
                0,
            )
        ),
    }

    trainable = (
        status["clearcut"] > 0
        and status["urban"] > 0
        and status["nochange"] > 0
        and status["spatial_groups"] >= N_CV_FOLDS
    )

    return (
        trainable,
        status,
    )


def run_cross_validation_scope(
    scope_name: str,
    parts: list[CachePart],
    feature_count: int,
    group_fold_table: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    metric_rows = []
    all_batch_rows = []

    total_confusion = np.zeros(
        (
            3,
            3,
        ),
        dtype=np.int64,
    )

    best_macro_f1 = -np.inf

    for fold in range(N_CV_FOLDS):
        validation_groups = set(
            group_fold_table.loc[
                group_fold_table["fold"] == fold,
                "spatial_group",
            ].astype(np.int64)
        )

        fold_name = f"{scope_name}_fold_{fold + 1}"

        (
            model,
            training_summary,
            batch_rows,
        ) = train_streamed_forest(
            parts=parts,
            excluded_groups=(validation_groups),
            feature_count=(feature_count),
            seed=(RANDOM_SEED + fold + 1),
            fold_name=(fold_name),
        )

        all_batch_rows.extend(batch_rows)

        (
            fold_confusion,
            validation_rows,
        ) = evaluate_streaming(
            model=model,
            parts=parts,
            validation_groups=(validation_groups),
        )

        metrics = metrics_from_confusion(fold_confusion)

        metric_rows.append(
            {
                "fold": (fold + 1),
                "validation_groups": (len(validation_groups)),
                "validation_rows": (validation_rows),
                **training_summary,
                **metrics,
            }
        )

        total_confusion += fold_confusion

        pd.DataFrame(metric_rows).to_csv(
            paths["cv_metrics"],
            index=False,
        )

        pd.DataFrame(all_batch_rows).to_csv(
            paths["batch_log"],
            index=False,
        )

        print(f"\n{scope_name} spatial fold {fold + 1}/{N_CV_FOLDS}")

        print(report_from_confusion(fold_confusion))

        print("\nConfusion matrix:")

        print(fold_confusion)

        if SAVE_EACH_FOLD_MODEL:
            joblib.dump(
                model,
                (paths["model_root"] / (f"rf_spatial_fold_{fold + 1:02d}.joblib")),
                compress=3,
            )

        if SAVE_BEST_FOLD_MODEL and metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]

            joblib.dump(
                model,
                paths["best_model"],
                compress=3,
            )

        del model

        gc.collect()

    pd.DataFrame(
        total_confusion,
        index=[
            "true_clearcut",
            "true_urban",
            "true_nochange",
        ],
        columns=[
            "pred_clearcut",
            "pred_urban",
            "pred_nochange",
        ],
    ).to_csv(paths["cv_confusion"])

    combined_report = report_from_confusion(total_confusion)

    paths["cv_report"].write_text(
        combined_report,
        encoding="utf-8",
    )


def train_final_scope(
    scope_name: str,
    parts: list[CachePart],
    feature_count: int,
    paths: dict[str, Path],
) -> None:
    (
        model,
        summary,
        batch_rows,
    ) = train_streamed_forest(
        parts=parts,
        excluded_groups=set(),
        feature_count=(feature_count),
        seed=(RANDOM_SEED),
        fold_name=(f"{scope_name}_final"),
    )

    joblib.dump(
        model,
        paths["final_model"],
        compress=3,
    )

    pd.DataFrame(batch_rows).to_csv(
        (paths["metrics_root"] / ("final_training_batch_composition.csv")),
        index=False,
    )

    logging.info(
        "Final %s model saved: %s | %s",
        scope_name,
        paths["final_model"],
        summary,
    )


def train_scope(
    scope_name: str,
    parts: list[CachePart],
) -> None:
    paths = scope_paths(scope_name)

    inventory = inventory_for_parts(parts)

    inventory.to_csv(
        paths["inventory"],
        index=False,
    )

    (
        trainable,
        status,
    ) = scope_is_trainable(
        scope_name,
        parts,
    )

    status["trainable"] = trainable

    paths["status"].write_text(
        json.dumps(
            status,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not trainable:
        logging.warning(
            "Skipping scope %s because it is not trainable: %s",
            scope_name,
            status,
        )
        return

    feature_count = parts[0].features

    group_inventory = build_group_inventory(parts)

    fold_table = assign_group_folds(
        group_inventory,
        paths["group_folds"],
    )

    logging.info(
        "Training scope %s: %s",
        scope_name,
        status,
    )

    if RUN_CROSS_VALIDATION:
        run_cross_validation_scope(
            scope_name=scope_name,
            parts=parts,
            feature_count=(feature_count),
            group_fold_table=(fold_table),
            paths=paths,
        )

    if TRAIN_FINAL_MODEL:
        train_final_scope(
            scope_name=scope_name,
            parts=parts,
            feature_count=(feature_count),
            paths=paths,
        )


def run_training(
    parts: list[CachePart],
) -> None:
    feature_count = parts[0].features

    expected_features = 64 * 3 + len(S2_BAND_NAMES) * 3 + 1

    if feature_count != expected_features:
        raise RuntimeError(
            f"Expected {expected_features} features, found {feature_count}."
        )

    if not TRAIN_NATIONAL_MODEL:
        logging.info("TRAIN_NATIONAL_MODEL is False; training is skipped.")
        return

    train_scope(
        scope_name="national",
        parts=parts,
    )


def main() -> None:
    setup_directories()
    setup_logging()
    save_configuration()
    save_feature_schema()

    random.seed(RANDOM_SEED)

    np.random.seed(RANDOM_SEED)

    logging.info(
        "OUTPUT_ROOT: %s",
        OUTPUT_ROOT,
    )

    logging.info(
        "RUN_ROOT: %s",
        RUN_ROOT,
    )

    logging.info(
        "Common target CRS: %s. "
        "One fixed %g m grid is derived "
        "per Sentinel-2 tile from year %d.",
        TARGET_CRS_TEXT,
        S2_TARGET_PIXEL_SIZE_METRES,
        S2_REFERENCE_YEAR,
    )

    logging.info(
        "S2 source CRS: 2018=stored native CRS; 2019+=forced EPSG:%d.",
        S2_2019_PLUS_SOURCE_EPSG,
    )

    logging.info(
        "S2 fixed band mapping: names=%s raster_indexes=%s.",
        S2_BAND_NAMES,
        S2_BAND_FALLBACK_INDEXES,
    )

    logging.info(
        "Excluding %d known outside-AOI S2 tiles from all years: %s",
        len(EXCLUDED_S2_TILE_IDS),
        sorted(EXCLUDED_S2_TILE_IDS),
    )

    logging.info(
        "AlphaEarth source zone is selected "
        "spatially per tile and warped "
        "to EPSG:25833."
    )

    logging.info(
        "Strict preflight: enabled=%s "
        "coarse max dimension=%d using "
        "nearest-neighbour mask reads; "
        "exact fallback block=%d.",
        RUN_INPUT_PREFLIGHT,
        PREFLIGHT_COARSE_MAX_DIM,
        PREFLIGHT_EXACT_BLOCK_SIZE,
    )

    logging.info(
        "AE selection: scan actual "
        "AE/S2 intersections; "
        "coarse max dimension=%d; "
        "exact fallback block=%d.",
        AE_SELECTION_COARSE_MAX_DIM,
        AE_SELECTION_EXACT_BLOCK_SIZE,
    )

    logging.info(
        "Coverage failure handling: "
        "abort_source=%s "
        "abort_combined=%s "
        "abort_selected=%s "
        "skip_bad_preflight=%s "
        "skip_bad_samples=%s",
        STRICT_ABORT_IF_SOURCE_ALL_NODATA,
        STRICT_ABORT_IF_COMBINED_ALL_NODATA,
        STRICT_ABORT_IF_SELECTED_SAMPLES_ALL_INVALID,
        SKIP_TILE_YEAR_IF_PREFLIGHT_INVALID,
        SKIP_TILE_YEAR_IF_SELECTED_SAMPLES_ALL_INVALID,
    )

    logging.info(
        "Spatial CV: fixed %d m cells in %s.",
        SPATIAL_GROUP_SIZE_METRES,
        SPATIAL_GROUP_CRS,
    )

    logging.info("Training output: exactly one national model.")

    logging.info(
        "Sampling workers=%d, warp threads/worker=%d, read block=%d",
        SAMPLING_WORKERS,
        GDAL_WARP_THREADS_PER_WORKER,
        RASTER_READ_BLOCK_SIZE,
    )

    if not ALPHA_ROOT.exists():
        raise FileNotFoundError(f"AlphaEarth root does not exist: {ALPHA_ROOT}")

    if not S2_ROOT.exists():
        raise FileNotFoundError(f"Sentinel-2 root does not exist: {S2_ROOT}")

    if not TRAINING_POLYGON_DIR.exists():
        raise FileNotFoundError(
            f"Training folder does not exist: {TRAINING_POLYGON_DIR}"
        )

    s2_index = build_s2_index()

    ae_catalog = build_ae_catalog()

    tasks = build_tile_tasks(
        s2_index=s2_index,
        ae_catalog=ae_catalog,
    )

    zone_grid_index = build_zone_grid_index(tasks)

    if RUN_VECTOR_PREPARATION:
        polygons = read_training_polygons()

        prepare_zone_vector_caches(
            polygons_ar5_crs=(polygons),
            zone_grid_index=(zone_grid_index),
        )

        del polygons
        gc.collect()

    else:
        for zone in zone_grid_index["zone"]:
            paths = zone_cache_paths(str(zone))

            if not (
                paths["polygons"].exists()
                and paths["metadata"].exists()
                and (paths["ar5"].exists() or paths["empty_ar5"].exists())
            ):
                raise FileNotFoundError(f"Missing prepared vector cache for {zone}.")

    if RUN_SAMPLING:
        run_parallel_sampling(tasks)

    if RUN_TRAINING:
        parts = discover_cache_parts()

        run_training(parts)

    logging.info("EPSG:25833 AlphaEarth + Sentinel-2 national workflow completed.")


if __name__ == "__main__":
    main()
