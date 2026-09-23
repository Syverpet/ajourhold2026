#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TESSERA-NO-S2: Norwegian teacher adaptation -> ~206M S2-only student -> 128-D embeddings
=========================================================================================

Pipeline
--------
1) Discover Norwegian Sentinel-2 MGRS tiles.
2) Build a sparse annual S2 time-series cache by HTTP range-reading public L2A COGs.
3) Domain-adapt the ~822M-parameter Sentinel-2 branch of the TESSERA v2 2B teacher.
4) Build a compact cache of 128-D targets from the adapted Norwegian teacher.
5) Distil those targets into an S2-only ~206M-parameter student.
6) Use ONLY the student for final 128-D embedding production.
7) Produce a test embedding GeoTIFF for MGRS tile T32VNM.

Pause / resume
--------------
Training is resumable by default.

* Press Ctrl+C during teacher or student training:
    - the script saves model + optimizer + scheduler + epoch/step + RNG state
    - then exits
    - rerunning the same script automatically continues from that checkpoint

* Or create this empty file while training:
      <RUN_DIR>/PAUSE_TRAINING.flag
  The script notices it after the current optimizer step, saves a full resume
  checkpoint, removes the flag, and exits cleanly.

* A full resume checkpoint is also written at the end of every epoch.

The cache builder and teacher-target builder are independently resumable because
completed files are skipped automatically.

Important methodological note
-----------------------------
The released TESSERA v2 teacher checkpoint is encoder-only. The original v2
training projector/optimizer is not public. The Norwegian teacher therefore uses
a Barlow-Twins-style self-supervised adaptation objective over two cloud-free
temporal views of the same S2 pixel-year.

The Norwegian student is then trained by knowledge distillation. For efficiency,
the adapted teacher is NOT run during every student update. Instead, a compact
teacher-target cache is created once. Each cached target is the adapted teacher's
full 4096-D S2-backbone representation for a sampled pixel. The 128-D student
uses a TRAINING-ONLY linear projector (128 -> 4096) to reconstruct that teacher
representation. The projector is discarded after distillation, so final outputs
remain native 128-D student embeddings.

Final inference uses the ~206M student, not the ~822M teacher.
"""

# =============================================================================
# USER SETTINGS -- EDIT THESE FIRST
# =============================================================================

# ----- Run control ------------------------------------------------------------
RUN_NAME = "tessera_no_s2_teacher_student_128"

RUN_STAGES = [
    "preflight",
    "discover_tiles",
    "build_cache",
    "train_teacher",
    "build_teacher_targets",
    "train_student",
    "infer_test_tile",
]

RANDOM_SEED = 42

# Resume behavior.
AUTO_RESUME = True
FORCE_RESTART_TEACHER = False
FORCE_RESTART_STUDENT = False

# Set True before launch if you want the script to finish the current epoch,
# checkpoint, and exit instead of continuing into the next epoch.
STOP_AFTER_CURRENT_EPOCH = False

# While the script is running, creating this file inside RUN_DIR requests a
# clean pause at the next optimizer-step boundary.
PAUSE_FLAG_FILENAME = "PAUSE_TRAINING.flag"

# ----- Storage ---------------------------------------------------------------
# FAST SSD. The hot S2 cache, checkpoints, teacher targets, and test output live here.
SSD_ROOT = r"F:\TESSERA_NO_S2"

# Optional slower HDD. It is not used in the hot training path.
# It can be used manually to archive old checkpoints/cache copies.
HDD_ROOT = r"H:\TESSERA_NO_S2_ARCHIVE"

# Sparse S2 cache cap. The design should normally stay far below 10 TB.
MAX_SSD_CACHE_GB = 2500.0

# With ~500 GB RAM, preload the hot sparse cache when it fits.
PRELOAD_CACHE_TO_RAM = True
RAM_PRELOAD_LIMIT_GB = 280.0

# Student distillation can use more RAM because the teacher is no longer resident.
STUDENT_RAM_PRELOAD_LIMIT_GB = 330.0

# ----- Norway boundary --------------------------------------------------------
# Strongly recommended: point this to your Norwegian kommune FGDB or another
# vector polygon covering Norway. It is used to keep sampled cache windows in Norway.
#
# Example:
NORWAY_BOUNDARY_PATH = (
    r"F:\Data\Administrative grenser\...\Norge_25833_Kommuner_FGDB.gdb"
)
# NORWAY_BOUNDARY_LAYER = "kommune"
#
# If None, the broad fallback bbox also includes parts of Sweden/Finland.
# NORWAY_BOUNDARY_PATH = None
NORWAY_BOUNDARY_LAYER = "kommune"
NORWAY_SEARCH_BBOX = (4.0, 57.5, 31.5, 71.5)

# ----- Sentinel-2 source ------------------------------------------------------
STAC_URL = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-2-l2a"
TRAIN_YEARS = list(range(2018, 2026))
DISCOVERY_DATE_RANGE = "2024-05-01/2024-09-30"
MAX_SCENE_CLOUD_PERCENT = 85.0
MAX_OBSERVATIONS_PER_YEAR = 24

# Official TESSERA S2 order. DO NOT reorder.
S2_ASSET_KEYS = [
    "red",  # B04
    "blue",  # B02
    "green",  # B03
    "nir",  # B08
    "nir08",  # B8A
    "rededge1",  # B05
    "rededge2",  # B06
    "rededge3",  # B07
    "swir16",  # B11
    "swir22",  # B12
]
SCL_ASSET_KEY = "scl"

# Invalid SCL classes. Snow/ice and thin cirrus are deliberately retained to
# stay aligned with the current public TESSERA preprocessing convention.
SCL_INVALID_VALUES = {0, 1, 2, 3, 8, 9}

# ----- Sparse S2 cache --------------------------------------------------------
CACHE_WINDOW_PIXELS = 256
WINDOWS_PER_TILE_YEAR = 4
MIN_WINDOW_NORWAY_FRACTION = 0.70
MAX_WINDOW_SAMPLE_ATTEMPTS = 100
MIN_CLEAR_OBSERVATIONS = 8
MAX_CACHE_WINDOWS = None

COG_READ_WORKERS = 24
STAC_RETRY_COUNT = 6

# ----- Geographic holdout -----------------------------------------------------
VAL_MGRS_FRACTION = 0.10
TEST_MGRS_TILE = "T32VNM"
TEST_YEAR = 2025

# ----- Official TESSERA v2 teacher -------------------------------------------
HF_TEACHER_REPO = "geotessera/TESSERA-V-2.0-2B-Teacher"
HF_TEACHER_CKPT = "ckpt/tessera_v2_2B_teacher.pt"
HF_TEACHER_MODEL_PY = "model.py"

# Final Norwegian embedding dimension.
EMBED_DIM = 128

# Distillation target is the adapted teacher S2 backbone before its 128-D
# self-supervised head. This preserves much more teacher information.
TEACHER_DISTILL_DIM = 4096

# ----- Temporal views ---------------------------------------------------------
TRAIN_SEQUENCE_LENGTH = 16
VIEW_KEEP_FRACTION = 0.70

# ----- GPU / numeric settings -------------------------------------------------
DEVICE = "cuda"
USE_BF16 = True
USE_TF32 = True
USE_TORCH_COMPILE = False
GRADIENT_CHECKPOINTING = True

# ----- Teacher adaptation -----------------------------------------------------
TEACHER_MICRO_BATCH_PIXELS = 192
TEACHER_GRAD_ACCUMULATION_STEPS = 4
TEACHER_BATCH_CHUNKS = 8

TEACHER_STEPS_PER_EPOCH = 800
TEACHER_VAL_STEPS_PER_EPOCH = 120

TEACHER_HEAD_WARMUP_EPOCHS = 3
TEACHER_LR_HEAD_WARMUP = 3e-4

TEACHER_STAGE2_LAST_N_LAYERS = 2
TEACHER_STAGE2_MAX_EPOCHS = 24
TEACHER_STAGE2_PATIENCE = 5
TEACHER_LR_STAGE2_BACKBONE = 2e-6
TEACHER_LR_STAGE2_HEAD = 8e-5

TEACHER_FULL_FINETUNE = True
TEACHER_STAGE3_MAX_EPOCHS = 16
TEACHER_STAGE3_PATIENCE = 4
TEACHER_LR_STAGE3_BACKBONE = 5e-7
TEACHER_LR_STAGE3_HEAD = 2e-5

TEACHER_WEIGHT_DECAY = 0.05
TEACHER_GRAD_CLIP_NORM = 1.0

# Barlow Twins objective for Norwegian teacher adaptation.
BARLOW_LAMBDA = 0.005
BARLOW_EPS = 1e-5

# ----- Teacher target cache for distillation ----------------------------------
# Only a deterministic subset of eligible pixels from each 256x256 S2 cache
# window is run through the expensive adapted teacher. Each target is the
# normalized 4096-D adapted S2-backbone representation. The student can revisit
# these pixels with many different temporal views.
TEACHER_TARGET_PIXELS_PER_WINDOW = 2048
TEACHER_TARGET_BATCH_PIXELS = 384
TEACHER_TARGET_DTYPE = "float16"

# ----- ~206M S2-only student architecture -------------------------------------
# Approx. 206M trainable parameters:
# d_model=1536, 7 QK-normalized Transformer layers, FFN=6144, 128-D output.
STUDENT_D_MODEL = 1536
STUDENT_NUM_LAYERS = 7
STUDENT_NHEAD = 12
STUDENT_FFN_DIM = 6144
STUDENT_HEAD_HIDDEN = 3072
STUDENT_DROPOUT = 0.0

# ----- Student distillation ----------------------------------------------------
STUDENT_MICRO_BATCH_PIXELS = 768
STUDENT_GRAD_ACCUMULATION_STEPS = 2
STUDENT_BATCH_CHUNKS = 8

STUDENT_STEPS_PER_EPOCH = 1000
STUDENT_VAL_STEPS_PER_EPOCH = 160
STUDENT_MAX_EPOCHS = 50
STUDENT_PATIENCE = 7

STUDENT_LR = 2e-4
STUDENT_WEIGHT_DECAY = 0.05
STUDENT_GRAD_CLIP_NORM = 1.0

DISTILL_MSE_WEIGHT = 1.0
DISTILL_COSINE_WEIGHT = 0.25

# ----- Plateau behavior -------------------------------------------------------
PLATEAU_FACTOR = 0.35
PLATEAU_MIN_LR = 1e-8
PLATEAU_THRESHOLD = 1e-4

# ----- Checkpoint / logging frequency -----------------------------------------
LOG_EVERY_N_STEPS = 10
SAVE_EVERY_N_EPOCHS = 1

# Full resume checkpoints contain optimizer states and can be large.
# 0 means no periodic mid-epoch disk checkpoint; Ctrl+C / pause-flag still saves
# immediately, and every completed epoch is checkpointed.
SAVE_RESUME_EVERY_N_STEPS = 0

# ----- Final test-tile inference: STUDENT ONLY --------------------------------
STUDENT_INFER_BATCH_PIXELS = 1024
INFER_WINDOW_PIXELS = 256
INFER_MIN_CLEAR_OBSERVATIONS = 4

TEST_OUTPUT_FLOAT32_GEOTIFF = True
OUTPUT_ZSTD_LEVEL = 6

# =============================================================================
# IMPORTS
# =============================================================================

import contextlib
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window, bounds as window_bounds, from_bounds
    from rasterio.warp import transform as rio_transform
except ImportError as e:
    raise RuntimeError("Install rasterio before running this script.") from e

try:
    import pystac_client
except ImportError as e:
    raise RuntimeError("Install pystac-client before running this script.") from e

try:
    from huggingface_hub import hf_hub_download
except ImportError as e:
    raise RuntimeError("Install huggingface_hub before running this script.") from e

# geopandas/shapely are optional unless a Norway boundary file is supplied.
try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    from shapely.ops import transform as shapely_transform
except Exception:
    gpd = None
    Point = None
    box = None


# =============================================================================
# PATHS
# =============================================================================

SSD_ROOT = Path(SSD_ROOT)
RUN_DIR = SSD_ROOT / RUN_NAME
CACHE_DIR = RUN_DIR / "s2_sparse_cache"
TEACHER_TARGET_DIR = RUN_DIR / "teacher_targets"
MODEL_DIR = RUN_DIR / "model"
OUTPUT_DIR = RUN_DIR / "output"
MANIFEST_DIR = RUN_DIR / "manifests"
LOG_DIR = RUN_DIR / "logs"

for _p in [
    RUN_DIR,
    CACHE_DIR,
    TEACHER_TARGET_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    MANIFEST_DIR,
    LOG_DIR,
]:
    _p.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "tessera_no_run.log"
TEACHER_HISTORY_CSV = LOG_DIR / "teacher_training_history.csv"
STUDENT_HISTORY_CSV = LOG_DIR / "student_training_history.csv"
CONFIG_JSON = LOG_DIR / "resolved_config.json"

TILE_MANIFEST_JSON = MANIFEST_DIR / "mgrs_tiles.json"
CACHE_INDEX_JSON = MANIFEST_DIR / "cache_index.json"
TEACHER_TARGET_INDEX_JSON = MANIFEST_DIR / "teacher_target_index.json"
TEACHER_TARGET_DONE_MARKER = MANIFEST_DIR / "teacher_targets_complete.json"

TEACHER_BEST_PATH = MODEL_DIR / "teacher_best_tessera_no_s2_128.pt"
TEACHER_RESUME_PATH = MODEL_DIR / "teacher_resume_full.pt"
TEACHER_DONE_MARKER = MODEL_DIR / "teacher_training_complete.json"

STUDENT_BEST_PATH = MODEL_DIR / "student_best_tessera_no_s2_128.pt"
STUDENT_RESUME_PATH = MODEL_DIR / "student_resume_full.pt"
STUDENT_DONE_MARKER = MODEL_DIR / "student_training_complete.json"

STUDENT_ARCH_JSON = MODEL_DIR / "student_architecture.json"
PAUSE_FLAG_PATH = RUN_DIR / PAUSE_FLAG_FILENAME

# =============================================================================
# LOGGING
# =============================================================================


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("tessera_no")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


LOGGER = setup_logging()


def log_exception(prefix: str):
    LOGGER.error("%s\n%s", prefix, traceback.format_exc())


def log_system_state(tag: str):
    parts = [tag]
    if psutil is not None:
        vm = psutil.virtual_memory()
        parts.append(
            f"RAM total={vm.total / 1024**3:.1f}GB available={vm.available / 1024**3:.1f}GB"
        )
        parts.append(f"CPU logical={psutil.cpu_count(logical=True)}")
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(dev)
        parts.append(f"GPU={prop.name}")
        parts.append(f"VRAM={prop.total_memory / 1024**3:.1f}GB")
        parts.append(f"alloc={torch.cuda.memory_allocated(dev) / 1024**3:.2f}GB")
        parts.append(f"reserved={torch.cuda.memory_reserved(dev) / 1024**3:.2f}GB")
    LOGGER.info(" | ".join(parts))


# =============================================================================
# REPRODUCIBILITY / PERFORMANCE
# =============================================================================


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(RANDOM_SEED)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = USE_TF32
    torch.backends.cudnn.allow_tf32 = USE_TF32
    torch.set_float32_matmul_precision("high")
    # Good default for large transformer GEMMs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Remote COG optimizations.
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", str(512 * 1024 * 1024))
os.environ.setdefault("GDAL_CACHEMAX", "8192")


# =============================================================================
# CONFIG SNAPSHOT
# =============================================================================


def write_config_snapshot():
    """Write all simple uppercase configuration values for reproducibility."""

    def _jsonable(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, set):
            return sorted(v)
        if isinstance(v, tuple):
            return list(v)
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, list):
            return [_jsonable(x) for x in v]
        return None

    d = {}
    for k, v in sorted(globals().items()):
        if not k.isupper():
            continue
        j = _jsonable(v)
        if j is not None:
            d[k] = j
    CONFIG_JSON.write_text(json.dumps(d, indent=2), encoding="utf-8")


# =============================================================================
# NORWAY BOUNDARY
# =============================================================================


class NorwayBoundary:
    def __init__(self):
        self.geometry_wgs84 = None
        self.enabled = False

        if NORWAY_BOUNDARY_PATH:
            p = Path(NORWAY_BOUNDARY_PATH)
            if p.exists() and gpd is not None:
                LOGGER.info(
                    "Loading Norway boundary: %s layer=%s", p, NORWAY_BOUNDARY_LAYER
                )
                gdf = gpd.read_file(p, layer=NORWAY_BOUNDARY_LAYER)
                if gdf.crs is None:
                    raise ValueError("Norway boundary has no CRS.")
                gdf = gdf.to_crs(4326)
                self.geometry_wgs84 = gdf.geometry.union_all()
                self.enabled = True
                LOGGER.info("Norway boundary loaded and dissolved.")
            else:
                LOGGER.warning(
                    "NORWAY_BOUNDARY_PATH was supplied but unavailable or geopandas "
                    "is missing. Falling back to bbox."
                )
        else:
            LOGGER.warning(
                "NORWAY_BOUNDARY_PATH=None: using broad bbox. Training discovery "
                "may include MGRS tiles in Sweden/Finland. Set your kommune GDB "
                "for a strict Norway-only run."
            )

    def contains_lonlat(self, lon: float, lat: float) -> bool:
        if not self.enabled:
            x0, y0, x1, y1 = NORWAY_SEARCH_BBOX
            return x0 <= lon <= x1 and y0 <= lat <= y1
        return bool(self.geometry_wgs84.contains(Point(lon, lat)))

    def intersects_bbox_lonlat(self, bbox_ll: Sequence[float]) -> bool:
        if not self.enabled:
            a = box(*NORWAY_SEARCH_BBOX) if box is not None else None
            b = box(*bbox_ll) if box is not None else None
            if a is None or b is None:
                return True
            return a.intersects(b)
        return bool(self.geometry_wgs84.intersects(box(*bbox_ll)))


# =============================================================================
# STAC / MGRS HELPERS
# =============================================================================

MGRS_RE = re.compile(r"(?:^|_)(\d{2}[A-Z]{3})(?:_|$)")


def normalize_mgrs(code: str) -> str:
    code = code.upper()
    return code if code.startswith("T") else "T" + code


def mgrs_from_item_id(item_id: str) -> Optional[str]:
    m = MGRS_RE.search(item_id.upper())
    if not m:
        return None
    return "T" + m.group(1)


def stac_client():
    return pystac_client.Client.open(STAC_URL)


def stac_search_with_retry(**kwargs):
    delay = 2.0
    last = None
    for attempt in range(1, STAC_RETRY_COUNT + 1):
        try:
            client = stac_client()
            search = client.search(collections=[STAC_COLLECTION], **kwargs)
            return list(search.items())
        except Exception as e:
            last = e
            LOGGER.warning(
                "STAC attempt %d/%d failed: %s", attempt, STAC_RETRY_COUNT, e
            )
            if attempt < STAC_RETRY_COUNT:
                time.sleep(delay)
                delay = min(delay * 1.8, 30.0)
    raise RuntimeError(f"STAC search failed after retries: {last}")


def discover_mgrs_tiles(boundary: NorwayBoundary) -> Dict[str, dict]:
    LOGGER.info("Discovering MGRS tiles intersecting Norway search region...")
    items = stac_search_with_retry(
        bbox=NORWAY_SEARCH_BBOX,
        datetime=DISCOVERY_DATE_RANGE,
        query={"eo:cloud_cover": {"lt": 95}},
    )
    LOGGER.info("Discovery STAC returned %d scenes.", len(items))

    tiles: Dict[str, dict] = {}
    for item in items:
        mgrs = mgrs_from_item_id(item.id)
        if mgrs is None or mgrs == TEST_MGRS_TILE:
            continue
        if not boundary.intersects_bbox_lonlat(item.bbox):
            continue
        if mgrs not in tiles:
            tiles[mgrs] = {
                "bbox": list(item.bbox),
                "example_item_id": item.id,
            }

    LOGGER.info("Discovered %d training-candidate MGRS tiles.", len(tiles))
    if not tiles:
        raise RuntimeError("No MGRS tiles discovered. Check STAC/network/bbox.")

    TILE_MANIFEST_JSON.write_text(json.dumps(tiles, indent=2), encoding="utf-8")
    return tiles


def load_or_discover_tiles(boundary: NorwayBoundary) -> Dict[str, dict]:
    if TILE_MANIFEST_JSON.exists():
        LOGGER.info("Using existing MGRS manifest: %s", TILE_MANIFEST_JSON)
        return json.loads(TILE_MANIFEST_JSON.read_text(encoding="utf-8"))
    return discover_mgrs_tiles(boundary)


def split_mgrs_tiles(tile_codes: Sequence[str]) -> Tuple[List[str], List[str]]:
    tiles = sorted(t for t in tile_codes if t != TEST_MGRS_TILE)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(tiles)
    n_val = max(1, int(round(len(tiles) * VAL_MGRS_FRACTION)))
    val = sorted(tiles[:n_val])
    train = sorted(tiles[n_val:])
    LOGGER.info(
        "Spatial MGRS split: train=%d val=%d test(excluded)=%s",
        len(train),
        len(val),
        TEST_MGRS_TILE,
    )
    return train, val


def search_tile_year(mgrs: str, bbox_ll: Sequence[float], year: int):
    items = stac_search_with_retry(
        bbox=list(bbox_ll),
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD_PERCENT}},
    )
    # Filter strictly by MGRS code in item ID.
    out = [it for it in items if mgrs_from_item_id(it.id) == mgrs]
    return out


def item_datetime(item) -> datetime:
    s = item.properties.get("datetime")
    if s:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    dt = item.datetime
    if dt is None:
        raise ValueError(f"Item has no datetime: {item.id}")
    return dt


def select_temporally_distributed_items(items: Sequence, max_n: int) -> List:
    if len(items) <= max_n:
        return sorted(items, key=item_datetime)

    # Divide the year into max_n temporal bins and keep the least cloudy scene
    # from each occupied bin, then fill any gaps with the best remaining scenes.
    bins: List[List] = [[] for _ in range(max_n)]
    for it in items:
        dt = item_datetime(it)
        doy = dt.timetuple().tm_yday
        bi = min(max_n - 1, int((doy - 1) / 366.0 * max_n))
        bins[bi].append(it)

    chosen = []
    chosen_ids = set()
    for group in bins:
        if not group:
            continue
        group = sorted(
            group,
            key=lambda x: (
                float(x.properties.get("eo:cloud_cover", 100.0)),
                item_datetime(x),
            ),
        )
        it = group[0]
        chosen.append(it)
        chosen_ids.add(it.id)

    if len(chosen) < max_n:
        remaining = [x for x in items if x.id not in chosen_ids]
        remaining.sort(key=lambda x: float(x.properties.get("eo:cloud_cover", 100.0)))
        chosen.extend(remaining[: max_n - len(chosen)])

    return sorted(chosen[:max_n], key=item_datetime)


# =============================================================================
# REMOTE COG READING
# =============================================================================


def rasterio_remote_env():
    return rasterio.Env(
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        GDAL_CACHEMAX=8192,
    )


def asset_href(item, key: str) -> str:
    if key not in item.assets:
        raise KeyError(
            f"Asset '{key}' missing from {item.id}; available={list(item.assets)}"
        )
    return item.assets[key].href


@dataclass
class BaseGrid:
    crs: str
    transform: object
    width: int
    height: int
    bounds: object


def get_base_grid(item) -> BaseGrid:
    href = asset_href(item, "red")
    with rasterio_remote_env():
        with rasterio.open(href) as src:
            return BaseGrid(
                crs=src.crs.to_string(),
                transform=src.transform,
                width=src.width,
                height=src.height,
                bounds=src.bounds,
            )


def window_center_lonlat(grid: BaseGrid, win: Window) -> Tuple[float, float]:
    x = grid.transform.c + (win.col_off + win.width / 2) * grid.transform.a
    y = grid.transform.f + (win.row_off + win.height / 2) * grid.transform.e
    lon, lat = rio_transform(grid.crs, "EPSG:4326", [x], [y])
    return lon[0], lat[0]


def choose_cache_windows(
    grid: BaseGrid,
    boundary: NorwayBoundary,
    n: int,
    rng: random.Random,
) -> List[Window]:
    p = CACHE_WINDOW_PIXELS
    if grid.width < p or grid.height < p:
        return []

    wins = []
    attempts = 0
    while len(wins) < n and attempts < MAX_WINDOW_SAMPLE_ATTEMPTS:
        attempts += 1
        col = rng.randrange(0, grid.width - p + 1)
        row = rng.randrange(0, grid.height - p + 1)
        win = Window(col, row, p, p)
        lon, lat = window_center_lonlat(grid, win)
        if boundary.contains_lonlat(lon, lat):
            wins.append(win)
    return wins


def read_asset_to_base_window(
    href: str,
    base_grid: BaseGrid,
    win: Window,
    resampling: Resampling,
    out_dtype=None,
) -> np.ndarray:
    b = window_bounds(win, base_grid.transform)
    with rasterio_remote_env():
        with rasterio.open(href) as src:
            src_win = from_bounds(*b, transform=src.transform)
            arr = src.read(
                1,
                window=src_win,
                out_shape=(int(win.height), int(win.width)),
                resampling=resampling,
                boundless=True,
                fill_value=0,
            )
    if out_dtype is not None:
        arr = arr.astype(out_dtype, copy=False)
    return arr


def read_one_observation_window(item, base_grid: BaseGrid, win: Window):
    # SCL first: cheap quality mask.
    scl = read_asset_to_base_window(
        asset_href(item, SCL_ASSET_KEY),
        base_grid,
        win,
        Resampling.nearest,
        np.uint8,
    )
    valid = ~np.isin(scl, list(SCL_INVALID_VALUES))

    bands = np.empty((int(win.height), int(win.width), 10), dtype=np.uint16)
    for j, key in enumerate(S2_ASSET_KEYS):
        # 20 m bands are resampled to the 10 m B04 grid.
        resampling = Resampling.bilinear
        if key in {"red", "blue", "green", "nir"}:
            resampling = Resampling.nearest
        arr = read_asset_to_base_window(
            asset_href(item, key),
            base_grid,
            win,
            resampling,
            np.uint16,
        )
        bands[..., j] = arr

    valid &= np.any(bands != 0, axis=-1)
    dt = item_datetime(item)
    doy = dt.timetuple().tm_yday
    return bands, valid.astype(np.uint8), doy, item.id


def cache_file_path(split: str, mgrs: str, year: int, wi: int) -> Path:
    d = CACHE_DIR / split / mgrs / str(year)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{mgrs}_{year}_w{wi:03d}.npz"


def save_cache_chunk(
    path: Path,
    bands: np.ndarray,
    masks: np.ndarray,
    doys: np.ndarray,
    item_ids: Sequence[str],
    win: Window,
    base_grid: BaseGrid,
):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        bands=bands,
        masks=masks,
        doys=doys.astype(np.int16),
        item_ids=np.array(item_ids, dtype="U120"),
        window=np.array(
            [win.col_off, win.row_off, win.width, win.height], dtype=np.int32
        ),
        crs=np.array([base_grid.crs]),
        transform=np.array(tuple(base_grid.transform)[:6], dtype=np.float64),
    )
    tmp.replace(path)


def directory_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / 1024**3


def build_sparse_cache(
    tiles: Dict[str, dict],
    train_tiles: Sequence[str],
    val_tiles: Sequence[str],
    boundary: NorwayBoundary,
):
    LOGGER.info("=== BUILD SPARSE SENTINEL-2 CACHE ===")
    split_lookup = {t: "train" for t in train_tiles}
    split_lookup.update({t: "val" for t in val_tiles})

    records = []
    created = 0

    for tile_i, mgrs in enumerate(sorted(split_lookup), 1):
        split = split_lookup[mgrs]
        bbox_ll = tiles[mgrs]["bbox"]

        for year in TRAIN_YEARS:
            if MAX_CACHE_WINDOWS is not None and created >= MAX_CACHE_WINDOWS:
                break

            current_gb = directory_size_gb(CACHE_DIR)
            if current_gb >= MAX_SSD_CACHE_GB:
                LOGGER.warning(
                    "Cache reached %.1f GB >= limit %.1f GB. Stopping cache build.",
                    current_gb,
                    MAX_SSD_CACHE_GB,
                )
                CACHE_INDEX_JSON.write_text(
                    json.dumps(records, indent=2), encoding="utf-8"
                )
                return

            # Skip a tile-year if all expected windows already exist.
            expected = [
                cache_file_path(split, mgrs, year, wi)
                for wi in range(WINDOWS_PER_TILE_YEAR)
            ]
            if all(p.exists() for p in expected):
                for p in expected:
                    records.append(
                        {"path": str(p), "split": split, "mgrs": mgrs, "year": year}
                    )
                continue

            LOGGER.info(
                "CACHE %s | tile %d/%d | %s | year=%d | querying STAC",
                split.upper(),
                tile_i,
                len(split_lookup),
                mgrs,
                year,
            )

            try:
                items = search_tile_year(mgrs, bbox_ll, year)
            except Exception:
                log_exception(f"STAC failed for {mgrs} {year}")
                continue

            if not items:
                LOGGER.warning("No S2 items: %s %d", mgrs, year)
                continue

            items = select_temporally_distributed_items(
                items, MAX_OBSERVATIONS_PER_YEAR
            )
            LOGGER.info(
                "CACHE %s %d | retained observations=%d",
                mgrs,
                year,
                len(items),
            )

            try:
                base_grid = get_base_grid(items[0])
            except Exception:
                log_exception(f"Could not read base grid for {mgrs} {year}")
                continue

            rng = random.Random(
                RANDOM_SEED
                + year * 100000
                + int(hashlib.md5(mgrs.encode()).hexdigest()[:6], 16)
            )
            windows = choose_cache_windows(
                base_grid, boundary, WINDOWS_PER_TILE_YEAR, rng
            )
            if not windows:
                LOGGER.warning(
                    "Could not sample Norway-centred windows: %s %d", mgrs, year
                )
                continue

            for wi, win in enumerate(windows):
                out = cache_file_path(split, mgrs, year, wi)
                if out.exists():
                    records.append(
                        {"path": str(out), "split": split, "mgrs": mgrs, "year": year}
                    )
                    continue

                t0 = time.time()
                obs_results = [None] * len(items)

                def _worker(k_it):
                    k, it = k_it
                    return k, read_one_observation_window(it, base_grid, win)

                with ThreadPoolExecutor(
                    max_workers=min(COG_READ_WORKERS, len(items))
                ) as ex:
                    futs = [ex.submit(_worker, pair) for pair in enumerate(items)]
                    for fut in as_completed(futs):
                        try:
                            k, result = fut.result()
                            obs_results[k] = result
                        except Exception as e:
                            LOGGER.warning(
                                "COG observation read failed %s %d window=%d: %s",
                                mgrs,
                                year,
                                wi,
                                e,
                            )

                obs_results = [x for x in obs_results if x is not None]
                if len(obs_results) < MIN_CLEAR_OBSERVATIONS:
                    LOGGER.warning(
                        "Too few readable observations %s %d w%d: %d",
                        mgrs,
                        year,
                        wi,
                        len(obs_results),
                    )
                    continue

                bands = np.stack([x[0] for x in obs_results], axis=0)  # T,H,W,10
                masks = np.stack([x[1] for x in obs_results], axis=0)  # T,H,W
                doys = np.array([x[2] for x in obs_results], dtype=np.int16)
                item_ids = [x[3] for x in obs_results]

                eligible_fraction = float(
                    (masks.sum(axis=0) >= MIN_CLEAR_OBSERVATIONS).mean()
                )
                if eligible_fraction < 0.05:
                    LOGGER.warning(
                        "Skipping low-clear-data chunk %s %d w%d eligible=%.1f%%",
                        mgrs,
                        year,
                        wi,
                        100 * eligible_fraction,
                    )
                    continue

                save_cache_chunk(out, bands, masks, doys, item_ids, win, base_grid)
                created += 1
                records.append(
                    {"path": str(out), "split": split, "mgrs": mgrs, "year": year}
                )

                LOGGER.info(
                    "CACHE DONE %s %d w%d | T=%d | eligible=%.1f%% | %.1f MB | %.1fs",
                    mgrs,
                    year,
                    wi,
                    bands.shape[0],
                    100 * eligible_fraction,
                    out.stat().st_size / 1024**2,
                    time.time() - t0,
                )

                del bands, masks, obs_results
                gc.collect()

    CACHE_INDEX_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    LOGGER.info(
        "Sparse cache build complete. files=%d size=%.1fGB",
        len(records),
        directory_size_gb(CACHE_DIR),
    )


# =============================================================================
# CACHE STORE
# =============================================================================


@dataclass
class CacheChunk:
    bands: np.ndarray  # T,H,W,10 uint16
    masks: np.ndarray  # T,H,W uint8
    doys: np.ndarray  # T int16
    path: str


class ChunkStore:
    def __init__(self, files: Sequence[Path], preload: bool, ram_limit_gb: float):
        self.files = list(files)
        self.preloaded: Optional[List[CacheChunk]] = None
        self.lru = OrderedDict()
        self.lru_max = 24

        if not self.files:
            raise RuntimeError("No cache files found.")

        # Conservative raw-size estimate.
        p = CACHE_WINDOW_PIXELS
        approx_raw_per_file = MAX_OBSERVATIONS_PER_YEAR * p * p * (10 * 2 + 1)
        est_raw_gb = approx_raw_per_file * len(self.files) / 1024**3

        LOGGER.info(
            "ChunkStore files=%d estimated_raw=%.1fGB preload=%s limit=%.1fGB",
            len(self.files),
            est_raw_gb,
            preload,
            ram_limit_gb,
        )

        can_preload = preload and est_raw_gb <= ram_limit_gb
        if psutil is not None:
            avail_gb = psutil.virtual_memory().available / 1024**3
            can_preload = can_preload and est_raw_gb < avail_gb * 0.70

        if can_preload:
            self.preloaded = []
            t0 = time.time()
            for i, f in enumerate(self.files, 1):
                self.preloaded.append(self._load_file(f))
                if i % 100 == 0:
                    LOGGER.info("RAM preload %d/%d", i, len(self.files))
            LOGGER.info("RAM preload complete in %.1f min", (time.time() - t0) / 60.0)
        else:
            LOGGER.info("Using on-demand SSD cache loading with small RAM LRU.")

    @staticmethod
    def _load_file(path: Path) -> CacheChunk:
        with np.load(path, allow_pickle=False) as z:
            return CacheChunk(
                bands=z["bands"],
                masks=z["masks"],
                doys=z["doys"],
                path=str(path),
            )

    def __len__(self):
        return len(self.files)

    def get(self, idx: int) -> CacheChunk:
        if self.preloaded is not None:
            return self.preloaded[idx]

        path = self.files[idx]
        key = str(path)
        if key in self.lru:
            chunk = self.lru.pop(key)
            self.lru[key] = chunk
            return chunk

        chunk = self._load_file(path)
        self.lru[key] = chunk
        while len(self.lru) > self.lru_max:
            self.lru.popitem(last=False)
        return chunk


# =============================================================================
# TESSERA S2 TEACHER + ~206M S2-ONLY STUDENT
# =============================================================================

S2_BAND_MEAN = np.array(
    [
        1633.0042,
        1341.1090,
        1539.5536,
        3054.8269,
        3117.4658,
        2004.1648,
        2694.7275,
        2945.1504,
        2266.6079,
        1657.3094,
    ],
    dtype=np.float32,
)
S2_BAND_STD = np.array(
    [
        1999.4603,
        2014.7549,
        1929.2201,
        1754.2493,
        1649.9807,
        1936.8988,
        1748.6041,
        1708.6991,
        1207.5250,
        1108.6046,
    ],
    dtype=np.float32,
)


class TrainingPaused(Exception):
    """Clean user-requested pause after a checkpoint has been written."""


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def atomic_torch_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def pause_flag_requested() -> bool:
    if not PAUSE_FLAG_PATH.exists():
        return False
    LOGGER.warning("Pause flag detected: %s", PAUSE_FLAG_PATH)
    try:
        PAUSE_FLAG_PATH.unlink()
    except OSError:
        pass
    return True


def import_module_from_file(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pretrained_s2_backbone_cpu():
    """
    Load ONLY the official TESSERA v2 teacher S2 branch.

    We still read the official checkpoint payload, but avoid constructing the
    S1 backbone, modality-fusion transformer, and full 1024-D reducer.
    """
    LOGGER.info("Loading official TESSERA v2 2B teacher checkpoint...")
    model_py = hf_hub_download(HF_TEACHER_REPO, HF_TEACHER_MODEL_PY)
    ckpt_path = hf_hub_download(HF_TEACHER_REPO, HF_TEACHER_CKPT)
    mod = import_module_from_file(model_py, "tessera_teacher_official")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = payload.get("arch", {}) or {}
    state = payload["encoder_state_dict"]

    s2_backbone = mod.TransformerEncoder(
        band_num=10,
        latent_dim=int(arch.get("latent_dim", 1024)),
        nhead=int(arch.get("nhead", 4)),
        num_encoder_layers=int(arch.get("num_layers", 4)),
        dim_feedforward=int(arch.get("dim_feedforward", 16384)),
        dropout=0.0,
    )

    prefix = "s2_backbone."
    s2_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not s2_state:
        raise RuntimeError(
            "No s2_backbone.* keys found in official teacher checkpoint."
        )
    s2_backbone.load_state_dict(s2_state, strict=True)

    del s2_state, state, payload
    gc.collect()

    n = sum(p.numel() for p in s2_backbone.parameters())
    LOGGER.info("Official S2 teacher branch loaded: %.3f M parameters", n / 1e6)
    return s2_backbone


class NorwayS2Teacher(nn.Module):
    """Norway-adapted official S2 teacher branch + new 128-D head."""

    def __init__(self, s2_backbone: nn.Module, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = s2_backbone
        self.head = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim, elementwise_affine=False),
        )

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        bands, doy = x[:, :, :-1], x[:, :, -1]
        h = self.backbone.embedding(bands)
        h = h + self.backbone.temporal_encoder(doy).to(h.dtype)

        for layer in self.backbone.transformer_encoder.layers:
            if (
                self.training
                and GRADIENT_CHECKPOINTING
                and any(p.requires_grad for p in layer.parameters())
            ):
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return self.backbone.attn_pool(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_backbone(x))


class TemporalPositionalEncoderStudent(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(
                0,
                self.d_model,
                2,
                dtype=torch.float32,
                device=doy.device,
            )
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(
            doy.shape[0],
            doy.shape[1],
            self.d_model,
            dtype=torch.float32,
            device=doy.device,
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class AttentionPoolingStudent(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x[:, 0]
        weights = torch.softmax(self.query(x), dim=1)
        return (weights * x).sum(dim=1)


class QKNormStudentLayer(nn.Module):
    """Pre-LN QK-normalized Transformer block, matching the v2 teacher style."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("STUDENT_D_MODEL must be divisible by STUDENT_NHEAD.")

        self.nhead = int(nhead)
        self.head_dim = d_model // nhead
        self.attn_dropout = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h, hd = self.nhead, self.head_dim

        q = self.q_proj(x).view(b, t, h, hd)
        k = self.k_proj(x).view(b, t, h, hd)
        v = self.v_proj(x).view(b, t, h, hd)

        # Match the public teacher implementation: Q/K norm in fp32.
        q = self.q_norm(q.float()).to(v.dtype).transpose(1, 2)
        k = self.k_norm(k.float()).to(v.dtype).transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).reshape(b, t, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self._attention(self.norm1(x)))
        h = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x)))))
        return x + self.dropout2(h)


class NorwayS2Student(nn.Module):
    """
    S2-only student.

    Default configuration is ~206M parameters, comfortably inside the requested
    100-250M range while being far cheaper for national inference than the
    ~822M teacher branch.
    """

    def __init__(self):
        super().__init__()
        d = STUDENT_D_MODEL

        self.embedding = nn.Sequential(
            nn.Linear(10, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.temporal_encoder = TemporalPositionalEncoderStudent(d)
        self.layers = nn.ModuleList(
            [
                QKNormStudentLayer(
                    d_model=d,
                    nhead=STUDENT_NHEAD,
                    ffn_dim=STUDENT_FFN_DIM,
                    dropout=STUDENT_DROPOUT,
                )
                for _ in range(STUDENT_NUM_LAYERS)
            ]
        )
        self.attn_pool = AttentionPoolingStudent(d)
        self.head = nn.Sequential(
            nn.Linear(d, STUDENT_HEAD_HIDDEN),
            nn.LayerNorm(STUDENT_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(STUDENT_HEAD_HIDDEN, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM, elementwise_affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands, doy = x[:, :, :-1], x[:, :, -1]
        h = self.embedding(bands)
        h = h + self.temporal_encoder(doy).to(h.dtype)

        for layer in self.layers:
            if self.training and GRADIENT_CHECKPOINTING:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)

        return self.head(self.attn_pool(h))


class StudentDistillWrapper(nn.Module):
    """
    Training-only wrapper.

    The final student emits 128-D. During distillation a linear projector maps
    128 -> 4096 so the student can reconstruct the full adapted S2-teacher
    representation. This is analogous to the released TESSERA v2 distillation
    idea, where training-only heads reconstruct a larger teacher representation.
    """

    def __init__(self, student: NorwayS2Student):
        super().__init__()
        self.student = student
        self.projector = nn.Sequential(
            nn.Linear(EMBED_DIM, TEACHER_DISTILL_DIM),
            nn.LayerNorm(TEACHER_DISTILL_DIM, elementwise_affine=False),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z128 = self.student(x)
        projected = self.projector(z128)
        return z128, projected


def write_student_architecture(model: NorwayS2Student):
    total = sum(p.numel() for p in model.parameters())
    atomic_json_save(
        {
            "parameters": int(total),
            "parameters_millions": total / 1e6,
            "d_model": STUDENT_D_MODEL,
            "layers": STUDENT_NUM_LAYERS,
            "heads": STUDENT_NHEAD,
            "ffn_dim": STUDENT_FFN_DIM,
            "head_hidden": STUDENT_HEAD_HIDDEN,
            "embedding_dim": EMBED_DIM,
            "distillation_target_dim": TEACHER_DISTILL_DIM,
            "training_only_projector": "128 -> 4096",
            "input": "Sentinel-2 annual per-pixel time series: 10 bands + raw DOY",
            "s2_band_order": S2_ASSET_KEYS,
            "teacher": "Norway-adapted TESSERA v2 2B teacher S2 branch",
        },
        STUDENT_ARCH_JSON,
    )
    LOGGER.info(
        "Student architecture: %.3f M params | d_model=%d layers=%d FFN=%d",
        total / 1e6,
        STUDENT_D_MODEL,
        STUDENT_NUM_LAYERS,
        STUDENT_FFN_DIM,
    )


# =============================================================================
# TEMPORAL VIEW SAMPLING
# =============================================================================


def sample_indices_fixed_length(
    valid_idx: np.ndarray,
    target_len: int,
    keep_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    if valid_idx.size == 0:
        return np.zeros(target_len, dtype=np.int64)

    keep_n = max(2, int(math.ceil(valid_idx.size * keep_fraction)))
    keep_n = min(keep_n, valid_idx.size)

    if keep_n < valid_idx.size:
        base = rng.choice(valid_idx, size=keep_n, replace=False)
    else:
        base = valid_idx.copy()

    if base.size >= target_len:
        selected = rng.choice(base, size=target_len, replace=False)
    else:
        extra = rng.choice(base, size=target_len - base.size, replace=True)
        selected = np.concatenate([base, extra])

    return np.sort(selected)


def make_one_temporal_view(
    chunk: CacheChunk,
    row: int,
    col: int,
    sequence_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_idx = np.flatnonzero(chunk.masks[:, row, col].astype(bool))
    idx = sample_indices_fixed_length(
        valid_idx, sequence_length, VIEW_KEEP_FRACTION, rng
    )

    vals = chunk.bands[idx, row, col, :].astype(np.float32)
    vals = (vals - S2_BAND_MEAN) / (S2_BAND_STD + 1e-9)

    out = np.empty((sequence_length, 11), dtype=np.float32)
    out[:, :10] = vals
    out[:, 10] = chunk.doys[idx].astype(np.float32)
    return out


def make_teacher_batch_from_chunk(
    chunk: CacheChunk,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    eligible = np.argwhere(
        chunk.masks.astype(bool).sum(axis=0) >= MIN_CLEAR_OBSERVATIONS
    )
    if len(eligible) < batch_size:
        raise RuntimeError(f"Too few eligible pixels in {chunk.path}")

    ids = rng.choice(len(eligible), size=batch_size, replace=False)
    coords = eligible[ids]

    view_a = np.empty((batch_size, TRAIN_SEQUENCE_LENGTH, 11), dtype=np.float32)
    view_b = np.empty_like(view_a)

    for i, (row, col) in enumerate(coords):
        view_a[i] = make_one_temporal_view(
            chunk, int(row), int(col), TRAIN_SEQUENCE_LENGTH, rng
        )
        view_b[i] = make_one_temporal_view(
            chunk, int(row), int(col), TRAIN_SEQUENCE_LENGTH, rng
        )
    return view_a, view_b


# =============================================================================
# LOSSES
# =============================================================================


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1n = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + BARLOW_EPS)
    z2n = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + BARLOW_EPS)
    c = (z1n.T @ z2n) / z1.shape[0]
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    off_diag = off_diagonal(c).pow(2).sum()
    return on_diag + BARLOW_LAMBDA * off_diag


def distillation_loss(student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(student, target)
    cosine = 1.0 - F.cosine_similarity(student, target, dim=-1).mean()
    return DISTILL_MSE_WEIGHT * mse + DISTILL_COSINE_WEIGHT * cosine


# =============================================================================
# COMMON TRAINING HELPERS
# =============================================================================


def list_cache_files(split: str) -> List[Path]:
    return sorted((CACHE_DIR / split).rglob("*.npz"))


def autocast_ctx():
    if DEVICE.startswith("cuda") and USE_BF16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def append_history(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def capture_rng_state(local_rng: np.random.Generator) -> dict:
    out = {
        "python": random.getstate(),
        "numpy_generator": local_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng_state(state: dict, local_rng: np.random.Generator):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy_generator" in state:
        local_rng.bit_generator.state = state["numpy_generator"]
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def checkpoint_model_state(model: nn.Module) -> dict:
    return unwrap_model(model).state_dict()


def save_best_model_only(path: Path, model: nn.Module, metadata: dict):
    payload = dict(metadata)
    payload.update(
        {
            "model_state_dict": checkpoint_model_state(model),
            "embed_dim": EMBED_DIM,
            "s2_band_order": S2_ASSET_KEYS,
            "s2_mean": S2_BAND_MEAN,
            "s2_std": S2_BAND_STD,
        }
    )
    atomic_torch_save(payload, path)


def random_teacher_batch(
    store: ChunkStore,
    rng: np.random.Generator,
    total_batch: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_chunks = min(TEACHER_BATCH_CHUNKS, total_batch)
    per = total_batch // n_chunks
    rem = total_batch % n_chunks

    aa, bb = [], []
    for i in range(n_chunks):
        want = per + (1 if i < rem else 0)

        for _ in range(30):
            chunk = store.get(int(rng.integers(0, len(store))))
            try:
                a, b = make_teacher_batch_from_chunk(chunk, want, rng)
                aa.append(a)
                bb.append(b)
                break
            except RuntimeError:
                pass
        else:
            raise RuntimeError(
                "Could not find a teacher cache chunk with enough valid pixels."
            )

    return np.concatenate(aa, axis=0), np.concatenate(bb, axis=0)


@torch.no_grad()
def evaluate_teacher(
    model: NorwayS2Teacher,
    val_store: ChunkStore,
    seed: int,
) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []

    for _ in range(TEACHER_VAL_STEPS_PER_EPOCH):
        a, b = random_teacher_batch(val_store, rng, TEACHER_MICRO_BATCH_PIXELS)
        ta = torch.from_numpy(a).to(DEVICE, non_blocking=True)
        tb = torch.from_numpy(b).to(DEVICE, non_blocking=True)

        with autocast_ctx():
            za = model(ta)
            zb = model(tb)
            loss = barlow_twins_loss(za.float(), zb.float())
        losses.append(float(loss.item()))

    model.train()
    return float(np.mean(losses))


def set_teacher_trainable(model: NorwayS2Teacher, stage: str):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    if stage == "head":
        pass
    elif stage == "last":
        for p in model.backbone.embedding.parameters():
            p.requires_grad = True
        for p in model.backbone.attn_pool.parameters():
            p.requires_grad = True
        layers = list(model.backbone.transformer_encoder.layers)
        for layer in layers[-TEACHER_STAGE2_LAST_N_LAYERS:]:
            for p in layer.parameters():
                p.requires_grad = True
    elif stage == "full":
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        raise ValueError(stage)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(
        "Teacher stage=%s | trainable %.3fM / %.3fM (%.1f%%)",
        stage,
        trainable / 1e6,
        total / 1e6,
        100.0 * trainable / total,
    )


def build_teacher_optimizer(model: NorwayS2Teacher, stage: str):
    if stage == "head":
        groups = [
            {
                "params": [p for p in model.head.parameters() if p.requires_grad],
                "lr": TEACHER_LR_HEAD_WARMUP,
            }
        ]
    elif stage == "last":
        groups = [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": TEACHER_LR_STAGE2_BACKBONE,
            },
            {
                "params": [p for p in model.head.parameters() if p.requires_grad],
                "lr": TEACHER_LR_STAGE2_HEAD,
            },
        ]
    elif stage == "full":
        groups = [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": TEACHER_LR_STAGE3_BACKBONE,
            },
            {
                "params": [p for p in model.head.parameters() if p.requires_grad],
                "lr": TEACHER_LR_STAGE3_HEAD,
            },
        ]
    else:
        raise ValueError(stage)

    return torch.optim.AdamW(
        groups,
        weight_decay=TEACHER_WEIGHT_DECAY,
        fused=bool(torch.cuda.is_available()),
    )


def teacher_phases() -> List[dict]:
    phases = [
        {
            "name": "head",
            "max_epochs": TEACHER_HEAD_WARMUP_EPOCHS,
            "patience": TEACHER_HEAD_WARMUP_EPOCHS + 1,
        },
        {
            "name": "last",
            "max_epochs": TEACHER_STAGE2_MAX_EPOCHS,
            "patience": TEACHER_STAGE2_PATIENCE,
        },
    ]
    if TEACHER_FULL_FINETUNE:
        phases.append(
            {
                "name": "full",
                "max_epochs": TEACHER_STAGE3_MAX_EPOCHS,
                "patience": TEACHER_STAGE3_PATIENCE,
            }
        )
    return phases


def make_teacher_resume_payload(
    model,
    optimizer,
    scheduler,
    phase_index,
    local_epoch,
    global_epoch,
    next_step,
    phase_best,
    bad_epochs,
    global_best,
    rng,
    running_sum,
    running_count,
    training_complete=False,
) -> dict:
    phases = teacher_phases()
    return {
        "kind": "teacher_resume",
        "version": 2,
        "model_state_dict": checkpoint_model_state(model),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "phase_index": int(phase_index),
        "phase_name": (
            phases[phase_index]["name"] if phase_index < len(phases) else "complete"
        ),
        "local_epoch": int(local_epoch),
        "global_epoch": int(global_epoch),
        "next_step": int(next_step),
        "phase_best": float(phase_best),
        "bad_epochs": int(bad_epochs),
        "global_best": float(global_best),
        "rng_state": capture_rng_state(rng),
        "running_sum": float(running_sum),
        "running_count": int(running_count),
        "training_complete": bool(training_complete),
    }


def train_teacher() -> NorwayS2Teacher:
    LOGGER.info("=== DOMAIN-ADAPT NORWEGIAN TESSERA S2 TEACHER ===")

    if FORCE_RESTART_TEACHER:
        for p in [
            TEACHER_RESUME_PATH,
            TEACHER_BEST_PATH,
            TEACHER_DONE_MARKER,
            TEACHER_TARGET_INDEX_JSON,
            TEACHER_TARGET_DONE_MARKER,
            STUDENT_RESUME_PATH,
            STUDENT_BEST_PATH,
            STUDENT_DONE_MARKER,
        ]:
            if p.exists():
                p.unlink()
        if TEACHER_TARGET_DIR.exists():
            shutil.rmtree(TEACHER_TARGET_DIR, ignore_errors=True)
        TEACHER_TARGET_DIR.mkdir(parents=True, exist_ok=True)
        LOGGER.warning(
            "FORCE_RESTART_TEACHER=True: teacher state, teacher targets, "
            "and dependent student state were invalidated."
        )

    train_files = list_cache_files("train")
    val_files = list_cache_files("val")
    if not train_files or not val_files:
        raise RuntimeError("Train and validation S2 caches are required.")

    train_store = ChunkStore(train_files, PRELOAD_CACHE_TO_RAM, RAM_PRELOAD_LIMIT_GB)
    val_store = ChunkStore(val_files, PRELOAD_CACHE_TO_RAM, RAM_PRELOAD_LIMIT_GB * 0.20)

    backbone = load_pretrained_s2_backbone_cpu()
    model = NorwayS2Teacher(backbone, EMBED_DIM)

    resume = None
    if AUTO_RESUME and TEACHER_RESUME_PATH.exists():
        LOGGER.info("Teacher resume found: %s", TEACHER_RESUME_PATH)
        resume = torch.load(TEACHER_RESUME_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state_dict"], strict=True)

    if resume and resume.get("training_complete", False):
        LOGGER.info("Teacher adaptation already complete; using best checkpoint.")
        if TEACHER_BEST_PATH.exists():
            best_ck = torch.load(
                TEACHER_BEST_PATH, map_location="cpu", weights_only=False
            )
            model.load_state_dict(best_ck["model_state_dict"], strict=True)
        return model.to(DEVICE).eval()

    model = model.to(DEVICE)
    if USE_TORCH_COMPILE:
        LOGGER.info("torch.compile teacher enabled.")
        model = torch.compile(model, mode="max-autotune")

    log_system_state("Teacher model ready")

    phases = teacher_phases()
    phase_index = int(resume.get("phase_index", 0)) if resume else 0
    global_epoch = int(resume.get("global_epoch", 0)) if resume else 0
    global_best = float(resume.get("global_best", math.inf)) if resume else math.inf

    while phase_index < len(phases):
        phase = phases[phase_index]
        stage = phase["name"]

        set_teacher_trainable(unwrap_model(model), stage)
        optimizer = build_teacher_optimizer(unwrap_model(model), stage)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=PLATEAU_FACTOR,
            patience=max(1, phase["patience"] // 2),
            threshold=PLATEAU_THRESHOLD,
            min_lr=PLATEAU_MIN_LR,
        )

        if resume and int(resume.get("phase_index", -1)) == phase_index:
            if resume.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(resume["optimizer_state_dict"])
            if resume.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(resume["scheduler_state_dict"])

            local_epoch = int(resume.get("local_epoch", 1))
            next_step = int(resume.get("next_step", 1))
            phase_best = float(resume.get("phase_best", math.inf))
            bad_epochs = int(resume.get("bad_epochs", 0))
            running_sum = float(resume.get("running_sum", 0.0))
            running_count = int(resume.get("running_count", 0))

            rng = np.random.default_rng()
            restore_rng_state(resume.get("rng_state", {}), rng)

            LOGGER.info(
                "RESUME TEACHER | stage=%s local_epoch=%d next_step=%d global_epoch=%d",
                stage,
                local_epoch,
                next_step,
                global_epoch,
            )
            resume = None
        else:
            local_epoch = 1
            next_step = 1
            phase_best = math.inf
            bad_epochs = 0
            running_sum = 0.0
            running_count = 0
            rng = np.random.default_rng(RANDOM_SEED + 1_000_000 * (phase_index + 1))

        while local_epoch <= phase["max_epochs"]:
            if next_step == 1:
                global_epoch += 1
                rng = np.random.default_rng(RANDOM_SEED + global_epoch * 100003)
                running_sum = 0.0
                running_count = 0

            epoch_t0 = time.time()
            unwrap_model(model).train()
            optimizer.zero_grad(set_to_none=True)

            try:
                for step in range(next_step, TEACHER_STEPS_PER_EPOCH + 1):
                    step_t0 = time.time()
                    loss_sum = 0.0

                    for _ in range(TEACHER_GRAD_ACCUMULATION_STEPS):
                        a, b = random_teacher_batch(
                            train_store,
                            rng,
                            TEACHER_MICRO_BATCH_PIXELS,
                        )
                        ta = torch.from_numpy(a).to(DEVICE, non_blocking=True)
                        tb = torch.from_numpy(b).to(DEVICE, non_blocking=True)

                        with autocast_ctx():
                            za = model(ta)
                            zb = model(tb)
                            loss = barlow_twins_loss(za.float(), zb.float())
                            scaled_loss = loss / TEACHER_GRAD_ACCUMULATION_STEPS

                        scaled_loss.backward()
                        loss_sum += float(loss.item())

                    torch.nn.utils.clip_grad_norm_(
                        [
                            p
                            for p in unwrap_model(model).parameters()
                            if p.requires_grad
                        ],
                        TEACHER_GRAD_CLIP_NORM,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    step_loss = loss_sum / TEACHER_GRAD_ACCUMULATION_STEPS
                    running_sum += step_loss
                    running_count += 1
                    next_step = step + 1

                    if step % LOG_EVERY_N_STEPS == 0 or step == 1:
                        lr_text = ",".join(
                            f"{g['lr']:.2e}" for g in optimizer.param_groups
                        )
                        LOGGER.info(
                            "TEACHER stage=%s epoch=%d step=%d/%d "
                            "loss=%.5f mean=%.5f lr=%s step_s=%.2f | "
                            "GPU %.1f/%.1fGB",
                            stage,
                            global_epoch,
                            step,
                            TEACHER_STEPS_PER_EPOCH,
                            step_loss,
                            running_sum / max(1, running_count),
                            lr_text,
                            time.time() - step_t0,
                            torch.cuda.memory_allocated() / 1024**3,
                            torch.cuda.memory_reserved() / 1024**3,
                        )

                    periodic = (
                        SAVE_RESUME_EVERY_N_STEPS > 0
                        and step % SAVE_RESUME_EVERY_N_STEPS == 0
                    )
                    pause = pause_flag_requested()

                    if periodic or pause:
                        LOGGER.info("Saving full teacher resume checkpoint...")
                        atomic_torch_save(
                            make_teacher_resume_payload(
                                model,
                                optimizer,
                                scheduler,
                                phase_index,
                                local_epoch,
                                global_epoch,
                                next_step,
                                phase_best,
                                bad_epochs,
                                global_best,
                                rng,
                                running_sum,
                                running_count,
                            ),
                            TEACHER_RESUME_PATH,
                        )
                        if pause:
                            raise TrainingPaused(
                                "Teacher paused after safe checkpoint."
                            )

            except KeyboardInterrupt:
                LOGGER.warning(
                    "Ctrl+C received during teacher training. "
                    "Saving full resume checkpoint..."
                )
                atomic_torch_save(
                    make_teacher_resume_payload(
                        model,
                        optimizer,
                        scheduler,
                        phase_index,
                        local_epoch,
                        global_epoch,
                        next_step,
                        phase_best,
                        bad_epochs,
                        global_best,
                        rng,
                        running_sum,
                        running_count,
                    ),
                    TEACHER_RESUME_PATH,
                )
                LOGGER.warning("Teacher resume checkpoint saved. Rerun to continue.")
                raise

            train_loss = running_sum / max(1, running_count)
            val_loss = evaluate_teacher(
                unwrap_model(model),
                val_store,
                RANDOM_SEED + 900000 + global_epoch,
            )
            scheduler.step(val_loss)

            append_history(
                TEACHER_HISTORY_CSV,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "stage": stage,
                    "global_epoch": global_epoch,
                    "local_epoch": local_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_minutes": (time.time() - epoch_t0) / 60.0,
                    "lr0": optimizer.param_groups[0]["lr"],
                    "lr1": optimizer.param_groups[-1]["lr"],
                },
            )

            LOGGER.info(
                "TEACHER EPOCH DONE stage=%s global=%d local=%d "
                "train=%.6f val=%.6f time=%.1fmin",
                stage,
                global_epoch,
                local_epoch,
                train_loss,
                val_loss,
                (time.time() - epoch_t0) / 60.0,
            )

            if val_loss < phase_best - PLATEAU_THRESHOLD:
                phase_best = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if val_loss < global_best - PLATEAU_THRESHOLD:
                global_best = val_loss
                LOGGER.info("Saving NEW BEST TEACHER val=%.6f", global_best)
                save_best_model_only(
                    TEACHER_BEST_PATH,
                    model,
                    {
                        "kind": "teacher_best",
                        "stage": stage,
                        "global_epoch": global_epoch,
                        "val_loss": val_loss,
                    },
                )

            local_epoch += 1
            next_step = 1
            running_sum = 0.0
            running_count = 0

            LOGGER.info("Saving end-of-epoch teacher resume checkpoint...")
            atomic_torch_save(
                make_teacher_resume_payload(
                    model,
                    optimizer,
                    scheduler,
                    phase_index,
                    local_epoch,
                    global_epoch,
                    next_step,
                    phase_best,
                    bad_epochs,
                    global_best,
                    rng,
                    running_sum,
                    running_count,
                ),
                TEACHER_RESUME_PATH,
            )

            if STOP_AFTER_CURRENT_EPOCH:
                raise TrainingPaused(
                    "STOP_AFTER_CURRENT_EPOCH=True after teacher epoch."
                )

            if bad_epochs >= phase["patience"]:
                LOGGER.info(
                    "Teacher stage=%s plateau after %d non-improving epochs.",
                    stage,
                    bad_epochs,
                )
                break

        # Move to next unfreezing phase.
        phase_index += 1
        if phase_index < len(phases):
            rng = np.random.default_rng(RANDOM_SEED + 2_000_000 + phase_index)
            atomic_torch_save(
                make_teacher_resume_payload(
                    model,
                    None,
                    None,
                    phase_index,
                    1,
                    global_epoch,
                    1,
                    math.inf,
                    0,
                    global_best,
                    rng,
                    0.0,
                    0,
                ),
                TEACHER_RESUME_PATH,
            )
            resume = torch.load(
                TEACHER_RESUME_PATH,
                map_location="cpu",
                weights_only=False,
            )

    rng = np.random.default_rng(RANDOM_SEED)
    atomic_torch_save(
        make_teacher_resume_payload(
            model,
            None,
            None,
            len(phases),
            1,
            global_epoch,
            1,
            global_best,
            0,
            global_best,
            rng,
            0.0,
            0,
            training_complete=True,
        ),
        TEACHER_RESUME_PATH,
    )
    atomic_json_save(
        {
            "complete": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "best_val_loss": global_best,
            "best_checkpoint": str(TEACHER_BEST_PATH),
        },
        TEACHER_DONE_MARKER,
    )

    if TEACHER_BEST_PATH.exists():
        best_ck = torch.load(
            TEACHER_BEST_PATH,
            map_location="cpu",
            weights_only=False,
        )
        unwrap_model(model).load_state_dict(best_ck["model_state_dict"], strict=True)

    LOGGER.info("Teacher adaptation COMPLETE.")
    return unwrap_model(model).eval()


# =============================================================================
# TEACHER TARGET CACHE
# =============================================================================


def get_bin_size(n_obs: int) -> int:
    if n_obs <= 0:
        return 0
    return min(256, int(math.ceil(n_obs / 8.0) * 8))


def pad_pattern(n: int, bucket: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(bucket, dtype=np.int64)
    if n >= bucket:
        return np.linspace(0, n - 1, bucket, dtype=np.int64)

    remain = bucket - n
    if remain <= n:
        groups = np.array_split(np.arange(n), remain)
        fill = np.array(
            [g[len(g) // 2] for g in groups],
            dtype=np.int64,
        )
    else:
        fill = (np.arange(remain) % n).astype(np.int64)

    return np.concatenate(
        [
            np.arange(n, dtype=np.int64),
            fill,
        ]
    )


def teacher_target_path(source_path: Path) -> Path:
    rel = source_path.relative_to(CACHE_DIR)
    return (TEACHER_TARGET_DIR / rel).with_suffix(".targets.npz")


def deterministic_target_coords(
    chunk: CacheChunk,
    source_path: Path,
) -> np.ndarray:
    eligible = np.argwhere(
        chunk.masks.astype(bool).sum(axis=0) >= MIN_CLEAR_OBSERVATIONS
    )
    if len(eligible) == 0:
        return np.empty((0, 2), dtype=np.int16)

    n = min(TEACHER_TARGET_PIXELS_PER_WINDOW, len(eligible))
    seed = RANDOM_SEED + int(
        hashlib.md5(str(source_path).encode("utf-8")).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(eligible), size=n, replace=False)
    return eligible[ids].astype(np.int16)


@torch.no_grad()
def encode_teacher_coords(
    model: NorwayS2Teacher,
    chunk: CacheChunk,
    coords: np.ndarray,
) -> np.ndarray:
    model.eval()
    targets = np.empty(
        (len(coords), TEACHER_DISTILL_DIM),
        dtype=np.float32,
    )

    grouped: Dict[int, List[int]] = {}
    for i, (row, col) in enumerate(coords):
        n = int(chunk.masks[:, int(row), int(col)].sum())
        bucket = get_bin_size(n)
        grouped.setdefault(bucket, []).append(i)

    for bucket, members in sorted(grouped.items()):
        for start in range(0, len(members), TEACHER_TARGET_BATCH_PIXELS):
            mids = members[start : start + TEACHER_TARGET_BATCH_PIXELS]
            inp = np.empty(
                (len(mids), bucket, 11),
                dtype=np.float32,
            )

            for gi, member_idx in enumerate(mids):
                row, col = coords[member_idx]
                row, col = int(row), int(col)

                valid_idx = np.flatnonzero(chunk.masks[:, row, col].astype(bool))
                valid_idx = np.sort(valid_idx)
                src_idx = valid_idx[pad_pattern(len(valid_idx), bucket)]

                vals = chunk.bands[src_idx, row, col, :].astype(np.float32)
                vals = (vals - S2_BAND_MEAN) / (S2_BAND_STD + 1e-9)

                inp[gi, :, :10] = vals
                inp[gi, :, 10] = chunk.doys[src_idx].astype(np.float32)

            tin = torch.from_numpy(inp).to(DEVICE, non_blocking=True)
            with autocast_ctx():
                backbone_repr = model.encode_backbone(tin)
            # Normalize the full teacher S2 representation per pixel so the
            # distillation target has a stable scale.
            backbone_repr = F.layer_norm(
                backbone_repr.float(),
                (TEACHER_DISTILL_DIM,),
            )
            targets[np.asarray(mids)] = backbone_repr.cpu().numpy()

    return targets


def load_best_teacher_for_targets() -> NorwayS2Teacher:
    if not TEACHER_BEST_PATH.exists():
        raise FileNotFoundError(f"Missing adapted teacher: {TEACHER_BEST_PATH}")
    backbone = load_pretrained_s2_backbone_cpu()
    model = NorwayS2Teacher(backbone, EMBED_DIM)
    ck = torch.load(
        TEACHER_BEST_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model.to(DEVICE).eval()


def build_teacher_targets(
    model: Optional[NorwayS2Teacher] = None,
):
    LOGGER.info("=== BUILD COMPACT TEACHER TARGET CACHE ===")

    if (
        AUTO_RESUME
        and TEACHER_TARGET_DONE_MARKER.exists()
        and not FORCE_RESTART_TEACHER
    ):
        LOGGER.info("Teacher-target cache already marked complete; skipping rebuild.")
        return

    if model is None:
        model = load_best_teacher_for_targets()

    all_sources = list_cache_files("train") + list_cache_files("val")
    records = []
    t_all = time.time()

    LOGGER.info(
        "Teacher-target source chunks=%d | max targets/chunk=%d | target_dim=%d",
        len(all_sources),
        TEACHER_TARGET_PIXELS_PER_WINDOW,
        TEACHER_DISTILL_DIM,
    )

    for i, source in enumerate(all_sources, 1):
        target = teacher_target_path(source)
        split = "train" if "train" in source.parts else "val"

        if target.exists():
            records.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "split": split,
                }
            )
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        chunk = ChunkStore._load_file(source)
        coords = deterministic_target_coords(chunk, source)
        if len(coords) == 0:
            LOGGER.warning(
                "No eligible teacher target pixels: %s",
                source,
            )
            continue

        target_values = encode_teacher_coords(model, chunk, coords)
        if TEACHER_TARGET_DTYPE == "float16":
            target_values = target_values.astype(np.float16)
        else:
            target_values = target_values.astype(np.float32)

        tmp = target.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            coords=coords,
            targets=target_values,
            source=np.array([str(source)]),
        )
        tmp.replace(target)

        records.append(
            {
                "source": str(source),
                "target": str(target),
                "split": split,
            }
        )

        LOGGER.info(
            "TEACHER TARGET %d/%d | %s | pixels=%d | %.2fMB | %.1fs | total %.2fh",
            i,
            len(all_sources),
            source.name,
            len(coords),
            target.stat().st_size / 1024**2,
            time.time() - t0,
            (time.time() - t_all) / 3600.0,
        )

    atomic_json_save(
        {"records": records},
        TEACHER_TARGET_INDEX_JSON,
    )
    atomic_json_save(
        {
            "complete": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "files": len(records),
            "target_dim": TEACHER_DISTILL_DIM,
            "targets_per_window": TEACHER_TARGET_PIXELS_PER_WINDOW,
        },
        TEACHER_TARGET_DONE_MARKER,
    )
    LOGGER.info(
        "Teacher target cache COMPLETE | files=%d | %.2fh",
        len(records),
        (time.time() - t_all) / 3600.0,
    )


# =============================================================================
# STUDENT DISTILLATION STORE
# =============================================================================


@dataclass
class DistillChunk:
    source: CacheChunk
    coords: np.ndarray
    targets: np.ndarray


class DistillStore:
    def __init__(
        self,
        source_files: Sequence[Path],
        preload: bool,
        ram_limit_gb: float,
    ):
        pairs = [
            (p, teacher_target_path(p))
            for p in source_files
            if teacher_target_path(p).exists()
        ]
        self.source_files = [p[0] for p in pairs]
        self.target_files = [p[1] for p in pairs]

        if not self.source_files:
            raise RuntimeError("No S2-cache / teacher-target pairs found.")

        self.preloaded = None
        self.lru = OrderedDict()
        self.lru_max = 24

        disk_gb = (
            sum(p.stat().st_size for p in self.source_files + self.target_files)
            / 1024**3
        )
        # Conservative enough for this cache layout on a 500 GB RAM machine.
        est_raw_gb = max(disk_gb * 3.0, 1.0)

        can_preload = preload and est_raw_gb <= ram_limit_gb
        if psutil is not None:
            avail_gb = psutil.virtual_memory().available / 1024**3
            can_preload = can_preload and est_raw_gb < avail_gb * 0.70

        LOGGER.info(
            "DistillStore pairs=%d disk=%.1fGB estimated_raw=%.1fGB preload=%s",
            len(self.source_files),
            disk_gb,
            est_raw_gb,
            can_preload,
        )

        if can_preload:
            self.preloaded = []
            t0 = time.time()
            for i in range(len(self.source_files)):
                self.preloaded.append(self._load(i))
                if (i + 1) % 100 == 0:
                    LOGGER.info(
                        "Distill RAM preload %d/%d",
                        i + 1,
                        len(self.source_files),
                    )
            LOGGER.info(
                "Distill RAM preload complete in %.1f min",
                (time.time() - t0) / 60.0,
            )

    def __len__(self):
        return len(self.source_files)

    def _load(self, idx: int) -> DistillChunk:
        source = ChunkStore._load_file(self.source_files[idx])
        with np.load(
            self.target_files[idx],
            allow_pickle=False,
        ) as z:
            coords = z["coords"].astype(np.int16, copy=False)
            targets = z["targets"]
        return DistillChunk(
            source=source,
            coords=coords,
            targets=targets,
        )

    def get(self, idx: int) -> DistillChunk:
        if self.preloaded is not None:
            return self.preloaded[idx]

        key = str(self.source_files[idx])
        if key in self.lru:
            out = self.lru.pop(key)
            self.lru[key] = out
            return out

        out = self._load(idx)
        self.lru[key] = out
        while len(self.lru) > self.lru_max:
            self.lru.popitem(last=False)
        return out


def make_student_batch_from_chunk(
    dchunk: DistillChunk,
    n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(dchunk.coords) == 0:
        raise RuntimeError("Empty distillation chunk.")

    ids = rng.choice(
        len(dchunk.coords),
        size=n,
        replace=(len(dchunk.coords) < n),
    )

    x = np.empty(
        (n, TRAIN_SEQUENCE_LENGTH, 11),
        dtype=np.float32,
    )
    y = dchunk.targets[ids].astype(np.float32, copy=False)

    for i, target_idx in enumerate(ids):
        row, col = dchunk.coords[target_idx]
        x[i] = make_one_temporal_view(
            dchunk.source,
            int(row),
            int(col),
            TRAIN_SEQUENCE_LENGTH,
            rng,
        )

    return x, y


def random_student_batch(
    store: DistillStore,
    rng: np.random.Generator,
    total_batch: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_chunks = min(
        STUDENT_BATCH_CHUNKS,
        total_batch,
    )
    per = total_batch // n_chunks
    rem = total_batch % n_chunks

    xs, ys = [], []
    for i in range(n_chunks):
        want = per + (1 if i < rem else 0)
        dchunk = store.get(int(rng.integers(0, len(store))))
        x, y = make_student_batch_from_chunk(dchunk, want, rng)
        xs.append(x)
        ys.append(y)

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
    )


@torch.no_grad()
def evaluate_student(
    model: StudentDistillWrapper,
    val_store: DistillStore,
    seed: int,
) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []

    for _ in range(STUDENT_VAL_STEPS_PER_EPOCH):
        x, y = random_student_batch(
            val_store,
            rng,
            STUDENT_MICRO_BATCH_PIXELS,
        )
        tx = torch.from_numpy(x).to(DEVICE, non_blocking=True)
        ty = torch.from_numpy(y).to(DEVICE, non_blocking=True)

        with autocast_ctx():
            _, projected = model(tx)
            loss = distillation_loss(projected.float(), ty.float())
        losses.append(float(loss.item()))

    model.train()
    return float(np.mean(losses))


def make_student_resume_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    next_step,
    best,
    bad_epochs,
    rng,
    running_sum,
    running_count,
    training_complete=False,
) -> dict:
    return {
        "kind": "student_resume",
        "version": 2,
        "distill_model_state_dict": checkpoint_model_state(model),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "epoch": int(epoch),
        "next_step": int(next_step),
        "best": float(best),
        "bad_epochs": int(bad_epochs),
        "rng_state": capture_rng_state(rng),
        "running_sum": float(running_sum),
        "running_count": int(running_count),
        "training_complete": bool(training_complete),
        "student_arch": {
            "d_model": STUDENT_D_MODEL,
            "layers": STUDENT_NUM_LAYERS,
            "nhead": STUDENT_NHEAD,
            "ffn": STUDENT_FFN_DIM,
            "head_hidden": STUDENT_HEAD_HIDDEN,
            "embed_dim": EMBED_DIM,
        },
    }


def train_student() -> NorwayS2Student:
    LOGGER.info("=== DISTIL ~206M NORWEGIAN S2 STUDENT ===")

    if (
        AUTO_RESUME
        and STUDENT_DONE_MARKER.exists()
        and STUDENT_BEST_PATH.exists()
        and not FORCE_RESTART_STUDENT
    ):
        LOGGER.info(
            "Student training already complete; loading final 128-D student directly."
        )
        return load_best_student_for_inference()

    if FORCE_RESTART_STUDENT:
        for p in [
            STUDENT_RESUME_PATH,
            STUDENT_BEST_PATH,
            STUDENT_DONE_MARKER,
        ]:
            if p.exists():
                p.unlink()
        LOGGER.warning("FORCE_RESTART_STUDENT=True: previous student state removed.")

    train_store = DistillStore(
        list_cache_files("train"),
        PRELOAD_CACHE_TO_RAM,
        STUDENT_RAM_PRELOAD_LIMIT_GB,
    )
    val_store = DistillStore(
        list_cache_files("val"),
        PRELOAD_CACHE_TO_RAM,
        STUDENT_RAM_PRELOAD_LIMIT_GB * 0.20,
    )

    student = NorwayS2Student()
    write_student_architecture(student)
    model = StudentDistillWrapper(student)

    resume = None
    if AUTO_RESUME and STUDENT_RESUME_PATH.exists():
        LOGGER.info(
            "Student resume found: %s",
            STUDENT_RESUME_PATH,
        )
        resume = torch.load(
            STUDENT_RESUME_PATH,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(
            resume["distill_model_state_dict"],
            strict=True,
        )

    if resume and resume.get("training_complete", False):
        LOGGER.info("Student training already complete; using best checkpoint.")
        if STUDENT_BEST_PATH.exists():
            best_ck = torch.load(
                STUDENT_BEST_PATH,
                map_location="cpu",
                weights_only=False,
            )
            model.student.load_state_dict(
                best_ck["model_state_dict"],
                strict=True,
            )
        return model.student.to(DEVICE).eval()

    model = model.to(DEVICE)
    if USE_TORCH_COMPILE:
        LOGGER.info("torch.compile student enabled.")
        model = torch.compile(model, mode="max-autotune")

    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(),
        lr=STUDENT_LR,
        weight_decay=STUDENT_WEIGHT_DECAY,
        fused=bool(torch.cuda.is_available()),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=PLATEAU_FACTOR,
        patience=max(1, STUDENT_PATIENCE // 2),
        threshold=PLATEAU_THRESHOLD,
        min_lr=PLATEAU_MIN_LR,
    )

    if resume:
        if resume.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        if resume.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume["scheduler_state_dict"])

        epoch = int(resume.get("epoch", 1))
        next_step = int(resume.get("next_step", 1))
        best = float(resume.get("best", math.inf))
        bad_epochs = int(resume.get("bad_epochs", 0))
        running_sum = float(resume.get("running_sum", 0.0))
        running_count = int(resume.get("running_count", 0))
        rng = np.random.default_rng()
        restore_rng_state(
            resume.get("rng_state", {}),
            rng,
        )

        LOGGER.info(
            "RESUME STUDENT | epoch=%d next_step=%d best=%.6f bad=%d",
            epoch,
            next_step,
            best,
            bad_epochs,
        )
    else:
        epoch = 1
        next_step = 1
        best = math.inf
        bad_epochs = 0
        running_sum = 0.0
        running_count = 0
        rng = np.random.default_rng(RANDOM_SEED + 7_000_000)

    log_system_state("Student model ready")

    while epoch <= STUDENT_MAX_EPOCHS:
        if next_step == 1:
            rng = np.random.default_rng(RANDOM_SEED + epoch * 300007)
            running_sum = 0.0
            running_count = 0

        epoch_t0 = time.time()
        unwrap_model(model).train()
        optimizer.zero_grad(set_to_none=True)

        try:
            for step in range(
                next_step,
                STUDENT_STEPS_PER_EPOCH + 1,
            ):
                step_t0 = time.time()
                loss_sum = 0.0

                for _ in range(STUDENT_GRAD_ACCUMULATION_STEPS):
                    x, y = random_student_batch(
                        train_store,
                        rng,
                        STUDENT_MICRO_BATCH_PIXELS,
                    )
                    tx = torch.from_numpy(x).to(DEVICE, non_blocking=True)
                    ty = torch.from_numpy(y).to(DEVICE, non_blocking=True)

                    with autocast_ctx():
                        _, projected = model(tx)
                        loss = distillation_loss(
                            projected.float(),
                            ty.float(),
                        )
                        scaled_loss = loss / STUDENT_GRAD_ACCUMULATION_STEPS

                    scaled_loss.backward()
                    loss_sum += float(loss.item())

                torch.nn.utils.clip_grad_norm_(
                    unwrap_model(model).parameters(),
                    STUDENT_GRAD_CLIP_NORM,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                step_loss = loss_sum / STUDENT_GRAD_ACCUMULATION_STEPS
                running_sum += step_loss
                running_count += 1
                next_step = step + 1

                if step % LOG_EVERY_N_STEPS == 0 or step == 1:
                    LOGGER.info(
                        "STUDENT epoch=%d step=%d/%d "
                        "loss=%.6f mean=%.6f lr=%.2e "
                        "step_s=%.2f | GPU %.1f/%.1fGB",
                        epoch,
                        step,
                        STUDENT_STEPS_PER_EPOCH,
                        step_loss,
                        running_sum / max(1, running_count),
                        optimizer.param_groups[0]["lr"],
                        time.time() - step_t0,
                        torch.cuda.memory_allocated() / 1024**3,
                        torch.cuda.memory_reserved() / 1024**3,
                    )

                periodic = (
                    SAVE_RESUME_EVERY_N_STEPS > 0
                    and step % SAVE_RESUME_EVERY_N_STEPS == 0
                )
                pause = pause_flag_requested()

                if periodic or pause:
                    LOGGER.info("Saving full student resume checkpoint...")
                    atomic_torch_save(
                        make_student_resume_payload(
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            next_step,
                            best,
                            bad_epochs,
                            rng,
                            running_sum,
                            running_count,
                        ),
                        STUDENT_RESUME_PATH,
                    )
                    if pause:
                        raise TrainingPaused("Student paused after safe checkpoint.")

        except KeyboardInterrupt:
            LOGGER.warning(
                "Ctrl+C received during student training. "
                "Saving full resume checkpoint..."
            )
            atomic_torch_save(
                make_student_resume_payload(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    next_step,
                    best,
                    bad_epochs,
                    rng,
                    running_sum,
                    running_count,
                ),
                STUDENT_RESUME_PATH,
            )
            LOGGER.warning("Student resume checkpoint saved. Rerun to continue.")
            raise

        train_loss = running_sum / max(1, running_count)
        val_loss = evaluate_student(
            unwrap_model(model),
            val_store,
            RANDOM_SEED + 8_000_000 + epoch,
        )
        scheduler.step(val_loss)

        append_history(
            STUDENT_HISTORY_CSV,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_minutes": (time.time() - epoch_t0) / 60.0,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

        LOGGER.info(
            "STUDENT EPOCH DONE epoch=%d train=%.6f val=%.6f time=%.1fmin",
            epoch,
            train_loss,
            val_loss,
            (time.time() - epoch_t0) / 60.0,
        )

        if val_loss < best - PLATEAU_THRESHOLD:
            best = val_loss
            bad_epochs = 0
            LOGGER.info(
                "Saving NEW BEST STUDENT val=%.6f",
                best,
            )
            save_best_model_only(
                STUDENT_BEST_PATH,
                unwrap_model(model).student,
                {
                    "kind": "student_best",
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "distillation_target_dim": TEACHER_DISTILL_DIM,
                    "student_arch": {
                        "d_model": STUDENT_D_MODEL,
                        "layers": STUDENT_NUM_LAYERS,
                        "nhead": STUDENT_NHEAD,
                        "ffn": STUDENT_FFN_DIM,
                        "head_hidden": STUDENT_HEAD_HIDDEN,
                    },
                },
            )
        else:
            bad_epochs += 1

        epoch += 1
        next_step = 1
        running_sum = 0.0
        running_count = 0

        LOGGER.info("Saving end-of-epoch student resume checkpoint...")
        atomic_torch_save(
            make_student_resume_payload(
                model,
                optimizer,
                scheduler,
                epoch,
                next_step,
                best,
                bad_epochs,
                rng,
                running_sum,
                running_count,
            ),
            STUDENT_RESUME_PATH,
        )

        if STOP_AFTER_CURRENT_EPOCH:
            raise TrainingPaused("STOP_AFTER_CURRENT_EPOCH=True after student epoch.")

        if bad_epochs >= STUDENT_PATIENCE:
            LOGGER.info(
                "Student validation plateau reached after %d non-improving epochs.",
                bad_epochs,
            )
            break

    atomic_torch_save(
        make_student_resume_payload(
            model,
            None,
            None,
            epoch,
            1,
            best,
            bad_epochs,
            np.random.default_rng(RANDOM_SEED),
            0.0,
            0,
            training_complete=True,
        ),
        STUDENT_RESUME_PATH,
    )
    atomic_json_save(
        {
            "complete": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "best_val_loss": best,
            "best_checkpoint": str(STUDENT_BEST_PATH),
        },
        STUDENT_DONE_MARKER,
    )

    final_wrapper = unwrap_model(model)
    if STUDENT_BEST_PATH.exists():
        best_ck = torch.load(
            STUDENT_BEST_PATH,
            map_location="cpu",
            weights_only=False,
        )
        final_wrapper.student.load_state_dict(
            best_ck["model_state_dict"],
            strict=True,
        )

    LOGGER.info("Student distillation COMPLETE.")
    return final_wrapper.student.eval()


# =============================================================================
# TEST TILE INFERENCE -- STUDENT ONLY
# =============================================================================


def get_test_tile_items(mgrs: str, year: int):
    broad = (5.0, 58.0, 13.0, 64.0)
    items = stac_search_with_retry(
        bbox=broad,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD_PERCENT}},
    )
    items = [item for item in items if mgrs_from_item_id(item.id) == mgrs]
    if not items:
        raise RuntimeError(f"No STAC scenes found for {mgrs} {year}")
    return select_temporally_distributed_items(
        items,
        MAX_OBSERVATIONS_PER_YEAR,
    )


def read_test_block(
    items,
    base_grid: BaseGrid,
    win: Window,
):
    obs = []
    for item in items:
        try:
            obs.append(read_one_observation_window(item, base_grid, win))
        except Exception as e:
            LOGGER.warning(
                "Test block observation read failed %s: %s",
                item.id,
                e,
            )
    if not obs:
        raise RuntimeError("No observations readable for test block.")

    return (
        np.stack([x[0] for x in obs], axis=0),
        np.stack([x[1] for x in obs], axis=0),
        np.array(
            [x[2] for x in obs],
            dtype=np.int16,
        ),
    )


@torch.no_grad()
def encode_block_with_student(
    model: NorwayS2Student,
    bands: np.ndarray,
    masks: np.ndarray,
    doys: np.ndarray,
) -> np.ndarray:
    t, h, w, _ = bands.shape
    n_pix = h * w

    flat_bands = bands.transpose(1, 2, 0, 3).reshape(n_pix, t, 10)
    flat_masks = masks.transpose(1, 2, 0).reshape(n_pix, t).astype(bool)

    counts = flat_masks.sum(axis=1)
    out = np.zeros(
        (n_pix, EMBED_DIM),
        dtype=np.float32,
    )
    buckets = np.array(
        [
            get_bin_size(int(n)) if n >= INFER_MIN_CLEAR_OBSERVATIONS else 0
            for n in counts
        ],
        dtype=np.int16,
    )

    model.eval()

    for bucket in sorted(set(int(x) for x in buckets if x > 0)):
        pixels = np.flatnonzero(buckets == bucket)

        for start in range(
            0,
            len(pixels),
            STUDENT_INFER_BATCH_PIXELS,
        ):
            ids = pixels[start : start + STUDENT_INFER_BATCH_PIXELS]
            inp = np.empty(
                (len(ids), bucket, 11),
                dtype=np.float32,
            )

            for i, pixel_id in enumerate(ids):
                valid_idx = np.flatnonzero(flat_masks[pixel_id])
                valid_idx = np.sort(valid_idx)
                src_idx = valid_idx[
                    pad_pattern(
                        len(valid_idx),
                        bucket,
                    )
                ]

                vals = flat_bands[pixel_id, src_idx].astype(np.float32)
                vals = (vals - S2_BAND_MEAN) / (S2_BAND_STD + 1e-9)

                inp[i, :, :10] = vals
                inp[i, :, 10] = doys[src_idx].astype(np.float32)

            tin = torch.from_numpy(inp).to(DEVICE, non_blocking=True)
            with autocast_ctx():
                emb = model(tin)
            out[ids] = emb.float().cpu().numpy()

    return out.reshape(h, w, EMBED_DIM)


def load_best_student_for_inference() -> NorwayS2Student:
    if not STUDENT_BEST_PATH.exists():
        raise FileNotFoundError(f"Missing student checkpoint: {STUDENT_BEST_PATH}")

    model = NorwayS2Student()
    ck = torch.load(
        STUDENT_BEST_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(
        ck["model_state_dict"],
        strict=True,
    )
    write_student_architecture(model)
    return model.to(DEVICE).eval()


def infer_test_tile(
    model: Optional[NorwayS2Student] = None,
):
    LOGGER.info(
        "=== STUDENT TEST TILE INFERENCE %s year=%d ===",
        TEST_MGRS_TILE,
        TEST_YEAR,
    )

    if model is None:
        model = load_best_student_for_inference()

    items = get_test_tile_items(
        TEST_MGRS_TILE,
        TEST_YEAR,
    )
    LOGGER.info(
        "Test tile retained observations=%d",
        len(items),
    )

    for item in items:
        LOGGER.info(
            "TEST OBS %s | %s | cloud=%.1f",
            item.id,
            item_datetime(item).date().isoformat(),
            float(item.properties.get("eo:cloud_cover", -1)),
        )

    base_grid = get_base_grid(items[0])
    out_tif = OUTPUT_DIR / (
        f"{TEST_MGRS_TILE}_{TEST_YEAR}_TESSERA_NO_S2_student206M_emb128.tif"
    )

    if not TEST_OUTPUT_FLOAT32_GEOTIFF:
        raise NotImplementedError("Only float32 GeoTIFF output is implemented.")

    profile = {
        "driver": "GTiff",
        "width": base_grid.width,
        "height": base_grid.height,
        "count": EMBED_DIM,
        "dtype": "float32",
        "crs": base_grid.crs,
        "transform": base_grid.transform,
        "tiled": True,
        "blockxsize": INFER_WINDOW_PIXELS,
        "blockysize": INFER_WINDOW_PIXELS,
        "compress": "ZSTD",
        "zstd_level": OUTPUT_ZSTD_LEVEL,
        "predictor": 3,
        "BIGTIFF": "YES",
        "interleave": "band",
        "nodata": 0.0,
        "NUM_THREADS": "ALL_CPUS",
    }

    total_windows = math.ceil(base_grid.width / INFER_WINDOW_PIXELS) * math.ceil(
        base_grid.height / INFER_WINDOW_PIXELS
    )

    LOGGER.info(
        "Writing %s | %dx%d x %d bands | windows=%d",
        out_tif,
        base_grid.width,
        base_grid.height,
        EMBED_DIM,
        total_windows,
    )

    t_all = time.time()
    done = 0

    with rasterio.open(out_tif, "w", **profile) as dst:
        for row in range(
            0,
            base_grid.height,
            INFER_WINDOW_PIXELS,
        ):
            h = min(
                INFER_WINDOW_PIXELS,
                base_grid.height - row,
            )
            for col in range(
                0,
                base_grid.width,
                INFER_WINDOW_PIXELS,
            ):
                w = min(
                    INFER_WINDOW_PIXELS,
                    base_grid.width - col,
                )
                win = Window(col, row, w, h)
                t0 = time.time()

                bands, masks, doys = read_test_block(
                    items,
                    base_grid,
                    win,
                )
                emb = encode_block_with_student(
                    model,
                    bands,
                    masks,
                    doys,
                )

                dst.write(
                    emb.transpose(2, 0, 1).astype(
                        np.float32,
                        copy=False,
                    ),
                    window=win,
                )

                done += 1
                elapsed = time.time() - t_all
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total_windows - done) / rate if rate > 0 else math.inf

                LOGGER.info(
                    "STUDENT INFER window=%d/%d "
                    "row=%d col=%d size=%dx%d "
                    "window_s=%.1f elapsed=%.2fh ETA=%.2fh",
                    done,
                    total_windows,
                    row,
                    col,
                    h,
                    w,
                    time.time() - t0,
                    elapsed / 3600.0,
                    eta / 3600.0,
                )

                del bands, masks, emb
                gc.collect()

    LOGGER.info(
        "STUDENT TEST TILE COMPLETE: %s | size=%.1fGB | total=%.2fh",
        out_tif,
        out_tif.stat().st_size / 1024**3,
        (time.time() - t_all) / 3600.0,
    )


# =============================================================================
# PREFLIGHT
# =============================================================================


def preflight():
    LOGGER.info("=== PREFLIGHT ===")
    write_config_snapshot()
    log_system_state("Preflight")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")
    if USE_BF16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "USE_BF16=True but this GPU/PyTorch reports no BF16 support."
        )

    LOGGER.info("Python: %s", sys.version.replace("\n", " "))
    LOGGER.info("PyTorch: %s | CUDA runtime: %s", torch.__version__, torch.version.cuda)
    LOGGER.info("Rasterio: %s", rasterio.__version__)
    LOGGER.info("Run dir: %s", RUN_DIR)

    if psutil is not None:
        d = psutil.disk_usage(str(SSD_ROOT.anchor if SSD_ROOT.anchor else SSD_ROOT))
        LOGGER.info(
            "SSD filesystem total=%.1fTB free=%.1fTB",
            d.total / 1024**4,
            d.free / 1024**4,
        )

    if HDD_ROOT:
        hdd = Path(HDD_ROOT)
        try:
            hdd.mkdir(parents=True, exist_ok=True)
            LOGGER.info("HDD archive root available: %s", hdd)
        except Exception as e:
            LOGGER.warning("HDD root unavailable: %s", e)

    # Quick HF metadata/checkpoint availability check without putting model on GPU.
    LOGGER.info("Checking Hugging Face teacher files and configured student...")
    model_py = hf_hub_download(HF_TEACHER_REPO, HF_TEACHER_MODEL_PY)
    LOGGER.info("Teacher model.py: %s", model_py)

    probe_student = NorwayS2Student()
    probe_wrapper = StudentDistillWrapper(probe_student)
    n_student = sum(p.numel() for p in probe_student.parameters())
    n_wrapper = sum(p.numel() for p in probe_wrapper.parameters())
    LOGGER.info(
        "Configured final S2-only student: %.3f M parameters "
        "(%.3f M including training-only projector)",
        n_student / 1e6,
        n_wrapper / 1e6,
    )
    if not (100e6 <= n_student <= 250e6):
        raise RuntimeError(
            f"Configured student has {n_student / 1e6:.1f}M parameters; "
            "expected 100-250M."
        )
    write_student_architecture(probe_student)
    del probe_wrapper, probe_student

    LOGGER.info("Resume checkpoints:")
    LOGGER.info("  teacher: %s", TEACHER_RESUME_PATH)
    LOGGER.info("  student: %s", STUDENT_RESUME_PATH)
    LOGGER.info("Pause flag: %s", PAUSE_FLAG_PATH)
    LOGGER.info("Preflight complete.")


# =============================================================================
# MAIN
# =============================================================================


def main():
    LOGGER.info("=" * 88)
    LOGGER.info("START %s", RUN_NAME)
    LOGGER.info("Stages: %s", RUN_STAGES)
    LOGGER.info("AUTO_RESUME=%s", AUTO_RESUME)
    LOGGER.info("Pause flag: %s", PAUSE_FLAG_PATH)
    LOGGER.info("Log file: %s", LOG_FILE)
    LOGGER.info("=" * 88)

    boundary = NorwayBoundary()
    tiles = None
    train_tiles = val_tiles = None
    teacher_model = None
    student_model = None

    if "preflight" in RUN_STAGES:
        preflight()

    if "discover_tiles" in RUN_STAGES:
        tiles = load_or_discover_tiles(boundary)
        train_tiles, val_tiles = split_mgrs_tiles(list(tiles))
        atomic_json_save(
            {
                "train": train_tiles,
                "val": val_tiles,
                "excluded_test": TEST_MGRS_TILE,
            },
            MANIFEST_DIR / "split.json",
        )

    if "build_cache" in RUN_STAGES:
        if tiles is None:
            tiles = load_or_discover_tiles(boundary)
        if train_tiles is None or val_tiles is None:
            train_tiles, val_tiles = split_mgrs_tiles(list(tiles))
        build_sparse_cache(
            tiles,
            train_tiles,
            val_tiles,
            boundary,
        )

    if "train_teacher" in RUN_STAGES:
        targets_already_complete = (
            AUTO_RESUME
            and TEACHER_TARGET_DONE_MARKER.exists()
            and not FORCE_RESTART_TEACHER
        )
        if targets_already_complete:
            LOGGER.info(
                "Teacher adaptation + teacher-target cache are already available; "
                "skipping expensive teacher reload on this resume run."
            )
        else:
            teacher_model = train_teacher()

    if "build_teacher_targets" in RUN_STAGES:
        build_teacher_targets(teacher_model)

        # The ~822M teacher is no longer needed once its target cache exists.
        if teacher_model is not None:
            del teacher_model
            teacher_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_system_state("Teacher released/skipped after target-cache stage")

    if "train_student" in RUN_STAGES:
        student_model = train_student()

    if "infer_test_tile" in RUN_STAGES:
        infer_test_tile(student_model)

    LOGGER.info("=" * 88)
    LOGGER.info("ALL REQUESTED STAGES COMPLETE")
    LOGGER.info("Main log: %s", LOG_FILE)
    LOGGER.info("Teacher history: %s", TEACHER_HISTORY_CSV)
    LOGGER.info("Student history: %s", STUDENT_HISTORY_CSV)
    LOGGER.info("Best teacher: %s", TEACHER_BEST_PATH)
    LOGGER.info("Best student: %s", STUDENT_BEST_PATH)
    LOGGER.info("Student architecture: %s", STUDENT_ARCH_JSON)
    LOGGER.info("=" * 88)


if __name__ == "__main__":
    try:
        main()
    except TrainingPaused as e:
        LOGGER.warning("%s", e)
        LOGGER.warning(
            "Training is safely paused. Rerun the same script; "
            "AUTO_RESUME will continue automatically."
        )
        sys.exit(0)
    except KeyboardInterrupt:
        LOGGER.warning(
            "Interrupted. If this occurred during teacher/student training, "
            "a resume checkpoint was saved before exit."
        )
        raise
    except Exception:
        log_exception("FATAL ERROR")
        raise
