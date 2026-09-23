"""
Merge AlphaEarth tiles per folder.

This version:
    - Uses gdalwarp directly, not gdalbuildvrt.
    - Handles AlphaEarth south-up source rasters.
    - Writes one mosaic into each input folder.
    - Aligns all years within the same UTM zone to a common grid.
    - Uses explicit GDAL memory units, e.g. -wm 48G.
"""

from __future__ import annotations

# =============================================================================
# USER VARIABLES
# =============================================================================


BASE_FOLDER = "F:/Data/AE/data/data/2025"

YEARS = [
    "2025",
]

ZONES = [
    # "31N",
    "32N",
    "33N",
    # "34N",
    # "35N",
    # "36N",
]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
INPUT_FOLDERS = [f"{BASE_FOLDER}/{year}/{zone}" for year in YEARS for zone in ZONES]

RASTER_EXTENSIONS = [".tif", ".tiff"]

OUTPUT_FILENAME_TEMPLATE = "AlphaEarth_mosaic_{year}_{zone}.tif"

TEMP_SUBFOLDER_NAME = "_mosaic_tmp"
GRID_METADATA_FOLDER_NAME = "_mosaic_grids"

# Set True now because earlier runs may have created partial outputs.
OVERWRITE_OUTPUTS = True

SCAN_ONLY = False

# For your 100-core machine:
# 6 x 16 = 96 GDAL worker threads.
PARALLEL_JOBS = 6
GDAL_THREADS_PER_JOB = 16

# GDAL cache and warp memory.
# Use explicit units for gdalwarp -wm.
GDAL_CACHE_GB_PER_JOB = 48
GDAL_WARP_MEMORY_STRING = "48G"

COMPRESS = "ZSTD"
PREDICTOR = 2
BLOCK_SIZE = 1024
BIGTIFF = "YES"
SPARSE_OK = "TRUE"

RESAMPLING = "near"
FORCE_NODATA_VALUE = -128

BUILD_OVERVIEWS = True
OVERVIEW_LEVELS = ["2", "4", "8", "16", "32", "64"]

VALIDATE_ALIGNMENT = True

STRICT_ALPHAEARTH_CHECKS = True
EXPECTED_BAND_COUNT = 64
EXPECTED_DTYPE = "int8"
EXPECTED_RESOLUTION = 10.0
EXPECTED_RESOLUTION_TOLERANCE = 1e-6

EXCLUDE_FILENAME_PREFIXES = [
    "AlphaEarth_mosaic_",
]

EXCLUDE_FOLDER_NAMES = [
    TEMP_SUBFOLDER_NAME,
    GRID_METADATA_FOLDER_NAME,
]

VERBOSE = True

# =============================================================================
# CODE
# =============================================================================

import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm


@dataclass(frozen=True)
class FolderJob:
    folder: str
    year: str
    zone: str
    output_path: str
    temp_folder: str


@dataclass(frozen=True)
class ZoneGrid:
    zone: str
    crs_string: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    xres: float
    yres: float
    width: int
    height: int
    band_count: int
    dtype: str
    nodata: float | int | None


def log(message: str) -> None:
    if VERBOSE:
        print(message)


def run_command(command: list[str], env: dict[str, str]) -> None:
    log("\n$ " + " ".join(command))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(result.stdout)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )

    if VERBOSE and result.stdout.strip():
        print(result.stdout)


def make_gdal_environment() -> dict[str, str]:
    env = os.environ.copy()

    # GDAL_CACHEMAX is usually interpreted as MB when numeric.
    # Here we keep it numeric for compatibility.
    env["GDAL_CACHEMAX"] = str(GDAL_CACHE_GB_PER_JOB * 1024)

    env["GDAL_NUM_THREADS"] = str(GDAL_THREADS_PER_JOB)
    env["OMP_NUM_THREADS"] = str(GDAL_THREADS_PER_JOB)
    env["OPENBLAS_NUM_THREADS"] = str(GDAL_THREADS_PER_JOB)
    env["MKL_NUM_THREADS"] = str(GDAL_THREADS_PER_JOB)
    env["NUMEXPR_NUM_THREADS"] = str(GDAL_THREADS_PER_JOB)

    env["VSI_CACHE"] = "TRUE"
    env["VSI_CACHE_SIZE"] = str(1024 * 1024 * 1024)

    return env


def is_excluded_file(path: Path) -> bool:
    return any(path.name.startswith(prefix) for prefix in EXCLUDE_FILENAME_PREFIXES)


def is_inside_excluded_folder(path: Path) -> bool:
    parts = set(path.parts)
    return any(folder_name in parts for folder_name in EXCLUDE_FOLDER_NAMES)


def find_rasters(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    allowed_extensions = {ext.lower() for ext in RASTER_EXTENSIONS}
    rasters: list[Path] = []

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if is_inside_excluded_folder(path):
            continue

        if path.suffix.lower() not in allowed_extensions:
            continue

        if is_excluded_file(path):
            continue

        rasters.append(path)

    return sorted(rasters)


def build_folder_jobs() -> list[FolderJob]:
    jobs: list[FolderJob] = []

    for folder_string in INPUT_FOLDERS:
        folder = Path(folder_string)
        year = folder.parent.name
        zone = folder.name

        output_name = OUTPUT_FILENAME_TEMPLATE.format(
            year=year,
            zone=zone,
            folder_name=folder.name,
        )

        jobs.append(
            FolderJob(
                folder=str(folder),
                year=year,
                zone=zone,
                output_path=str(folder / output_name),
                temp_folder=str(folder / TEMP_SUBFOLDER_NAME),
            )
        )

    return jobs


def group_jobs_by_zone(jobs: list[FolderJob]) -> dict[str, list[FolderJob]]:
    grouped: dict[str, list[FolderJob]] = {}

    for job in jobs:
        grouped.setdefault(job.zone, []).append(job)

    return grouped


def crs_to_stable_string(src: rasterio.DatasetReader) -> str:
    epsg = src.crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    return src.crs.to_wkt()


def get_positive_pixel_size(src: rasterio.DatasetReader) -> tuple[float, float]:
    return abs(float(src.transform.a)), abs(float(src.transform.e))


def get_robust_bounds(src: rasterio.DatasetReader) -> tuple[float, float, float, float]:
    bounds = src.bounds

    xmin = min(bounds.left, bounds.right)
    xmax = max(bounds.left, bounds.right)
    ymin = min(bounds.bottom, bounds.top)
    ymax = max(bounds.bottom, bounds.top)

    return xmin, ymin, xmax, ymax


def snap_down(value: float, resolution: float) -> float:
    return math.floor(value / resolution) * resolution


def snap_up(value: float, resolution: float) -> float:
    return math.ceil(value / resolution) * resolution


def check_alphaearth_format(
    raster_path: Path,
    band_count: int,
    dtype: str,
    xres: float,
    yres: float,
) -> None:
    if not STRICT_ALPHAEARTH_CHECKS:
        return

    if band_count != EXPECTED_BAND_COUNT:
        raise ValueError(
            f"Unexpected band count in {raster_path}: "
            f"{band_count}, expected {EXPECTED_BAND_COUNT}"
        )

    if dtype != EXPECTED_DTYPE:
        raise ValueError(
            f"Unexpected dtype in {raster_path}: {dtype}, expected {EXPECTED_DTYPE}"
        )

    if not math.isclose(
        xres,
        EXPECTED_RESOLUTION,
        rel_tol=0,
        abs_tol=EXPECTED_RESOLUTION_TOLERANCE,
    ):
        raise ValueError(
            f"Unexpected x resolution in {raster_path}: "
            f"{xres}, expected {EXPECTED_RESOLUTION}"
        )

    if not math.isclose(
        yres,
        EXPECTED_RESOLUTION,
        rel_tol=0,
        abs_tol=EXPECTED_RESOLUTION_TOLERANCE,
    ):
        raise ValueError(
            f"Unexpected y resolution in {raster_path}: "
            f"{yres}, expected {EXPECTED_RESOLUTION}"
        )


def scan_zone_grid(zone: str, jobs_for_zone: list[FolderJob]) -> ZoneGrid:
    xmin = ymin = xmax = ymax = None
    crs_string = None
    xres = yres = None
    band_count = None
    dtype = None
    nodata = None

    file_count = 0

    for job in jobs_for_zone:
        folder = Path(job.folder)
        rasters = find_rasters(folder)

        if not rasters:
            print(f"[WARNING] No rasters found while scanning: {folder}")
            continue

        for raster_path in rasters:
            try:
                with rasterio.open(raster_path) as src:
                    file_count += 1

                    current_crs_string = crs_to_stable_string(src)
                    current_xres, current_yres = get_positive_pixel_size(src)
                    current_band_count = src.count
                    current_dtype = src.dtypes[0]

                    check_alphaearth_format(
                        raster_path=raster_path,
                        band_count=current_band_count,
                        dtype=current_dtype,
                        xres=current_xres,
                        yres=current_yres,
                    )

                    if crs_string is None:
                        crs_string = current_crs_string
                    elif current_crs_string != crs_string:
                        raise ValueError(
                            "CRS mismatch inside zone group.\n"
                            f"Zone: {zone}\n"
                            f"Expected: {crs_string}\n"
                            f"Found:    {current_crs_string}\n"
                            f"File:     {raster_path}"
                        )

                    if xres is None:
                        xres = current_xres
                        yres = current_yres
                    else:
                        if not math.isclose(
                            current_xres, xres, rel_tol=0, abs_tol=1e-9
                        ):
                            raise ValueError(
                                f"X resolution mismatch in {raster_path}: "
                                f"{current_xres} vs {xres}"
                            )

                        if not math.isclose(
                            current_yres, yres, rel_tol=0, abs_tol=1e-9
                        ):
                            raise ValueError(
                                f"Y resolution mismatch in {raster_path}: "
                                f"{current_yres} vs {yres}"
                            )

                    if band_count is None:
                        band_count = current_band_count
                    elif current_band_count != band_count:
                        raise ValueError(
                            f"Band-count mismatch in {raster_path}: "
                            f"{current_band_count} vs {band_count}"
                        )

                    if dtype is None:
                        dtype = current_dtype
                    elif current_dtype != dtype:
                        raise ValueError(
                            f"Dtype mismatch in {raster_path}: "
                            f"{current_dtype} vs {dtype}"
                        )

                    if FORCE_NODATA_VALUE is not None:
                        nodata = FORCE_NODATA_VALUE
                    elif nodata is None:
                        nodata = src.nodata

                    bxmin, bymin, bxmax, bymax = get_robust_bounds(src)

                    xmin = bxmin if xmin is None else min(xmin, bxmin)
                    ymin = bymin if ymin is None else min(ymin, bymin)
                    xmax = bxmax if xmax is None else max(xmax, bxmax)
                    ymax = bymax if ymax is None else max(ymax, bymax)

            except RasterioIOError as exc:
                raise RuntimeError(f"Could not open raster: {raster_path}") from exc

    if file_count == 0:
        raise FileNotFoundError(f"No raster files found for zone: {zone}")

    assert xmin is not None
    assert ymin is not None
    assert xmax is not None
    assert ymax is not None
    assert xres is not None
    assert yres is not None
    assert crs_string is not None
    assert band_count is not None
    assert dtype is not None

    snapped_xmin = snap_down(xmin, xres)
    snapped_ymin = snap_down(ymin, yres)
    snapped_xmax = snap_up(xmax, xres)
    snapped_ymax = snap_up(ymax, yres)

    width = int(round((snapped_xmax - snapped_xmin) / xres))
    height = int(round((snapped_ymax - snapped_ymin) / yres))

    return ZoneGrid(
        zone=zone,
        crs_string=crs_string,
        xmin=snapped_xmin,
        ymin=snapped_ymin,
        xmax=snapped_xmax,
        ymax=snapped_ymax,
        xres=xres,
        yres=yres,
        width=width,
        height=height,
        band_count=band_count,
        dtype=dtype,
        nodata=nodata,
    )


def write_zone_grid_files(
    jobs_by_zone: dict[str, list[FolderJob]],
    grid_metadata_folder: Path,
) -> dict[str, Path]:
    grid_metadata_folder.mkdir(parents=True, exist_ok=True)

    grid_paths: dict[str, Path] = {}

    print("\nScanning common grid per UTM zone...")

    for zone, zone_jobs in tqdm(jobs_by_zone.items(), desc="Zones"):
        grid = scan_zone_grid(zone, zone_jobs)
        grid_path = grid_metadata_folder / f"grid_{zone}.json"

        with open(grid_path, "w", encoding="utf-8") as file:
            json.dump(asdict(grid), file, indent=2)

        grid_paths[zone] = grid_path

        print(
            f"\nZone {zone}: "
            f"{grid.width} x {grid.height}, "
            f"{grid.band_count} bands, "
            f"{grid.dtype}, "
            f"nodata={grid.nodata}, "
            f"res=({grid.xres}, {grid.yres}), "
            f"crs={grid.crs_string}"
        )

    return grid_paths


def load_zone_grid(grid_path: Path) -> ZoneGrid:
    with open(grid_path, "r", encoding="utf-8") as file:
        return ZoneGrid(**json.load(file))


def build_mosaic_for_job(job: FolderJob, grid_path_string: str) -> str | None:
    folder = Path(job.folder)
    output_path = Path(job.output_path)
    temp_folder = Path(job.temp_folder)
    grid = load_zone_grid(Path(grid_path_string))

    if not folder.exists():
        print(f"[SKIP] Folder does not exist: {folder}")
        return None

    rasters = find_rasters(folder)

    if not rasters:
        print(f"[SKIP] No rasters in: {folder}")
        return None

    if output_path.exists() and not OVERWRITE_OUTPUTS:
        print(f"[SKIP] Output exists: {output_path}")
        return str(output_path)

    temp_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n[START] {job.year}/{job.zone}: {len(rasters)} rasters")
    print(f"[OUTPUT] {output_path}")

    env = make_gdal_environment()

    creation_options = [
        "-co",
        "TILED=YES",
        "-co",
        f"BLOCKXSIZE={BLOCK_SIZE}",
        "-co",
        f"BLOCKYSIZE={BLOCK_SIZE}",
        "-co",
        f"BIGTIFF={BIGTIFF}",
        "-co",
        f"SPARSE_OK={SPARSE_OK}",
        "-co",
        f"COMPRESS={COMPRESS}",
        "-co",
        f"PREDICTOR={PREDICTOR}",
        "-co",
        "NUM_THREADS=ALL_CPUS",
    ]

    command = [
        "gdalwarp",
        "-overwrite",
        "-multi",
        "-wo",
        f"NUM_THREADS={GDAL_THREADS_PER_JOB}",
        # Important: explicit unit. Do NOT pass only 49152.
        "-wm",
        GDAL_WARP_MEMORY_STRING,
        "-r",
        RESAMPLING,
        "-t_srs",
        grid.crs_string,
        "-tr",
        str(grid.xres),
        str(grid.yres),
        "-te",
        str(grid.xmin),
        str(grid.ymin),
        str(grid.xmax),
        str(grid.ymax),
        "-tap",
        "-srcnodata",
        str(grid.nodata),
        "-dstnodata",
        str(grid.nodata),
    ]

    command += creation_options
    command += [str(raster) for raster in rasters]
    command += [str(output_path)]

    run_command(command, env)

    if BUILD_OVERVIEWS:
        overview_command = [
            "gdaladdo",
            "-r",
            "nearest",
            "--config",
            "COMPRESS_OVERVIEW",
            COMPRESS,
            "--config",
            "BIGTIFF_OVERVIEW",
            "YES",
            str(output_path),
        ] + OVERVIEW_LEVELS

        run_command(overview_command, env)

    shutil.rmtree(temp_folder, ignore_errors=True)

    print(f"[DONE] {output_path}")
    return str(output_path)


def read_output_signature(path: Path) -> tuple:
    with rasterio.open(path) as src:
        return (
            crs_to_stable_string(src),
            src.transform,
            src.width,
            src.height,
            src.count,
            src.dtypes,
            src.nodata,
        )


def validate_alignment(jobs_by_zone: dict[str, list[FolderJob]]) -> None:
    print("\nValidating output alignment within each zone...")

    for zone, zone_jobs in jobs_by_zone.items():
        reference_signature = None
        reference_path = None

        for job in zone_jobs:
            output_path = Path(job.output_path)

            if not output_path.exists():
                print(f"[WARNING] Missing output during validation: {output_path}")
                continue

            signature = read_output_signature(output_path)

            if reference_signature is None:
                reference_signature = signature
                reference_path = output_path
                continue

            if signature != reference_signature:
                raise ValueError(
                    "Alignment mismatch detected.\n"
                    f"Zone:      {zone}\n"
                    f"Reference: {reference_path}\n"
                    f"Different: {output_path}\n"
                )

        print(f"[OK] Zone {zone} outputs are aligned.")


def check_gdal_tools_available() -> None:
    tools = ["gdalwarp", "gdaladdo"]

    for tool in tools:
        result = subprocess.run(
            [tool, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"GDAL tool not available: {tool}. "
                "Install GDAL and make sure it is available in PATH."
            )

        log(f"[GDAL] {tool}: {result.stdout.strip()}")


def print_configuration_summary(
    jobs: list[FolderJob], grid_metadata_folder: Path
) -> None:
    print("\nConfiguration:")
    print(f"  Number of folders:          {len(jobs)}")
    print(f"  Parallel jobs:              {PARALLEL_JOBS}")
    print(f"  GDAL threads per job:       {GDAL_THREADS_PER_JOB}")
    print(f"  Approx GDAL threads total:  {PARALLEL_JOBS * GDAL_THREADS_PER_JOB}")
    print(f"  GDAL cache per job:         {GDAL_CACHE_GB_PER_JOB} GB")
    print(f"  GDAL warp memory:           {GDAL_WARP_MEMORY_STRING}")
    print(f"  Approx cache total:         {PARALLEL_JOBS * GDAL_CACHE_GB_PER_JOB} GB")
    print(f"  Compression:                {COMPRESS}")
    print(f"  Predictor:                  {PREDICTOR}")
    print(f"  Block size:                 {BLOCK_SIZE}")
    print(f"  Resampling:                 {RESAMPLING}")
    print(f"  Force NoData:               {FORCE_NODATA_VALUE}")
    print(f"  Build overviews:            {BUILD_OVERVIEWS}")
    print(f"  Strict AE checks:           {STRICT_ALPHAEARTH_CHECKS}")
    print(f"  Grid metadata folder:       {grid_metadata_folder}")


def main() -> None:
    check_gdal_tools_available()

    jobs = build_folder_jobs()

    existing_input_folders = [
        Path(job.folder) for job in jobs if Path(job.folder).exists()
    ]
    if not existing_input_folders:
        raise FileNotFoundError("None of the INPUT_FOLDERS exist.")

    common_root = Path(BASE_FOLDER)
    grid_metadata_folder = common_root / GRID_METADATA_FOLDER_NAME

    print_configuration_summary(jobs, grid_metadata_folder)

    jobs_by_zone = group_jobs_by_zone(jobs)

    grid_paths = write_zone_grid_files(
        jobs_by_zone=jobs_by_zone,
        grid_metadata_folder=grid_metadata_folder,
    )

    if SCAN_ONLY:
        print("\nSCAN_ONLY=True, stopping after grid scan.")
        return

    print("\nBuilding mosaics...")

    futures = []

    with ProcessPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
        for job in jobs:
            grid_path = grid_paths[job.zone]
            futures.append(
                executor.submit(
                    build_mosaic_for_job,
                    job,
                    str(grid_path),
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Mosaics"):
            try:
                future.result()
            except Exception as exc:
                print(f"\n[ERROR] {exc}", file=sys.stderr)
                raise

    if VALIDATE_ALIGNMENT:
        validate_alignment(jobs_by_zone)

    print("\nAll done.")


if __name__ == "__main__":
    main()
