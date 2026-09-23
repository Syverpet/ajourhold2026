#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Pattern, Tuple

from osgeo import gdal

# ============================================================
# CONFIG — edit these variables at the top
# ============================================================

# --- Folders ---
INPUT_DIR: Path = Path(r"D:\Data\Kalkkart NGU\AE2024\2024")
# A staging folder where we write "north-up" normalized tiles
STAGING_DIR: Path = Path(r"D:\Data\Kalkkart NGU\AE2024\2024_northup")
# Final merged output folder (e.g. ~80 tiles)
OUTPUT_DIR: Path = Path(r"D:\Data\Kalkkart NGU\AE2024\merged_80")

# --- Input discovery ---
INPUT_GLOB: str = "*.tif"  # e.g. "*.tif", "*.tiff", "**/*.tif" (recursive)

# --- Grouping / naming ---
TILE_REGEX: str = r"_tile_(\d{2})_(\d{2})"
OUTPUT_NAME_TEMPLATE: str = "{tile_id}.tif"

# --- Parallelism ---
# Keep this modest if you are I/O bound (network drive). Often 4–12 is plenty.
WORKERS: int = max(1, (os.cpu_count() or 2) - 1)

# --- Output format & performance ---
OUTPUT_DTYPE: str = "int8"  # for DL, float32 is often enough + faster I/O
NODATA: float | None = None

COMPRESS: str = "ZSTD"  # "LZW", "DEFLATE", "NONE"
BIGTIFF: str = "IF_SAFER"  # "YES", "NO", "IF_SAFER"
TILED: bool = True  # recommended for patch reads

# --- North-up normalization settings ---
# Your tile is 10 m pixels. Use positive values here; GDAL will write north-up with negative Y GT.
TARGET_SRS: str = "EPSG:32632"
XRES: float = 10.0
YRES: float = 10.0
TAP_ALIGN: bool = True  # -tap (align to pixel grid)
RESAMPLE_ALG: str = "near"  # "near" for embeddings/features; "bilinear" for imagery

# --- Pipeline behavior ---
NORMALIZE_ONLY_IF_NEEDED: bool = (
    True  # if True, only warp tiles with positive NS pixel size
)
OVERWRITE_STAGING: bool = False  # if False, skips already-normalized tiles in staging
CLEAN_STAGING_ON_START: bool = False  # if True, deletes STAGING_DIR at start

# Mosaic stage method:
# After normalization, VRT->Translate is fast and safe.
MOSAIC_USE_WARP: bool = False  # keep False after normalization

# Stability-first:
WARP_MULTITHREAD: bool = False  # keep False; parallelism comes from processes
GDAL_CACHEMAX_MB: int = 2048  # per process

# Logging
LOG_LEVEL: str = "INFO"

# ============================================================
# End CONFIG
# ============================================================

log = logging.getLogger("mosaic")


# ---------------------------
# Your grouping function
# ---------------------------
def group_by_tile(paths: List[Path], tile_re: Pattern[str]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    skipped: List[Path] = []
    for p in paths:
        m = tile_re.search(p.name)
        if not m:
            skipped.append(p)
            continue
        row, col = m.group(1), m.group(2)
        tile_id = f"tile_{row}_{col}"
        groups.setdefault(tile_id, []).append(p)

    if skipped:
        log.warning(
            "Skipped %d files that did not match TILE_REGEX (%s). Example: %s",
            len(skipped),
            TILE_REGEX,
            skipped[0].name,
        )

    for k in groups:
        groups[k] = sorted(groups[k])
    return groups


# ---------------------------
# Utilities
# ---------------------------
def _collect_inputs(input_dir: Path, pattern: str) -> List[Path]:
    return [p for p in sorted(input_dir.glob(pattern)) if p.is_file()]


def _is_north_up(p: Path) -> bool:
    """Return True if geotransform pixel height (GT[5]) is negative (north-up)."""
    ds = gdal.Open(str(p), gdal.GA_ReadOnly)
    if ds is None:
        return False
    gt = ds.GetGeoTransform(can_return_null=True)
    ds = None
    if gt is None:
        return False
    return gt[5] < 0


def _normalize_one_tile(
    src: Path,
    dst: Path,
    *,
    target_srs: str,
    xres: float,
    yres: float,
    tap_align: bool,
    resample_alg: str,
    output_dtype: str,
    nodata: float | None,
    compress: str,
    bigtiff: str,
    tiled: bool,
    warp_multithread: bool,
    gdal_cachemax_mb: int,
    normalize_only_if_needed: bool,
    overwrite: bool,
) -> Tuple[str, str, bool]:
    """
    Normalize one tile to north-up GeoTIFF at dst.
    Returns (src, dst, did_work)
    """
    gdal.UseExceptions()
    os.environ.setdefault("GDAL_CACHEMAX", str(gdal_cachemax_mb))

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        return str(src), str(dst), False

    if normalize_only_if_needed and _is_north_up(src):
        # If already north-up, we can just copy (or skip).
        # Copying keeps a consistent staging folder.
        if not dst.exists() or overwrite:
            shutil.copy2(src, dst)
        return str(src), str(dst), True

    creation_opts = [f"COMPRESS={compress}", f"BIGTIFF={bigtiff}"]
    if tiled:
        creation_opts.append("TILED=YES")

    out_type_map = {
        "float64": gdal.GDT_Float64,
        "float32": gdal.GDT_Float32,
        "int16": gdal.GDT_Int16,
        "uint16": gdal.GDT_UInt16,
        "byte": gdal.GDT_Byte,
    }
    if output_dtype not in out_type_map:
        raise ValueError(
            f"Unsupported output_dtype={output_dtype}. Choose from {list(out_type_map)}"
        )

    warp_opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS=target_srs,
        xRes=xres,
        yRes=yres,
        targetAlignedPixels=tap_align,
        resampleAlg=resample_alg,
        outputType=out_type_map[output_dtype],
        dstNodata=nodata,
        creationOptions=creation_opts,
        multithread=warp_multithread,
    )

    out_ds = gdal.Warp(str(dst), str(src), options=warp_opts)
    if out_ds is None:
        raise RuntimeError(f"Normalization warp failed: {src.name}")
    out_ds = None
    return str(src), str(dst), True


def _mosaic_one_group(
    tile_id: str,
    input_paths: List[Path],
    out_path: Path,
    *,
    dtype: str,
    nodata: float | None,
    compress: str,
    bigtiff: str,
    tiled: bool,
    use_warp: bool,
    warp_multithread: bool,
    resample_alg: str,
    xres: float,
    yres: float,
    tap_align: bool,
    target_srs: str,
    gdal_cachemax_mb: int,
) -> Tuple[str, str]:
    """
    Mosaic one output tile group from already-normalized (north-up) inputs.
    Default: VRT->Translate (fast). Optionally Warp.
    """
    gdal.UseExceptions()
    os.environ.setdefault("GDAL_CACHEMAX", str(gdal_cachemax_mb))

    if not input_paths:
        raise ValueError(f"{tile_id}: no input paths")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    creation_opts = [f"COMPRESS={compress}", f"BIGTIFF={bigtiff}"]
    if tiled:
        creation_opts.append("TILED=YES")

    out_type_map = {
        "float64": gdal.GDT_Float64,
        "float32": gdal.GDT_Float32,
        "int16": gdal.GDT_Int16,
        "uint16": gdal.GDT_UInt16,
        "byte": gdal.GDT_Byte,
    }
    if dtype not in out_type_map:
        raise ValueError(
            f"{tile_id}: unsupported dtype={dtype}. Choose from {list(out_type_map)}"
        )

    if use_warp:
        warp_opts = gdal.WarpOptions(
            format="GTiff",
            dstSRS=target_srs,
            xRes=xres,
            yRes=yres,
            targetAlignedPixels=tap_align,
            resampleAlg=resample_alg,
            outputType=out_type_map[dtype],
            dstNodata=nodata,
            creationOptions=creation_opts,
            multithread=warp_multithread,
        )
        out_ds = gdal.Warp(
            str(out_path), [str(p) for p in input_paths], options=warp_opts
        )
        if out_ds is None:
            raise RuntimeError(f"{tile_id}: Warp mosaic failed")
        out_ds = None
        return tile_id, str(out_path)

    # VRT -> Translate fast path
    with tempfile.TemporaryDirectory(prefix=f"vrt_{tile_id}_") as tmpdir:
        vrt_path = Path(tmpdir) / f"{tile_id}.vrt"
        vrt_ds = gdal.BuildVRT(str(vrt_path), [str(p) for p in input_paths])
        if vrt_ds is None:
            raise RuntimeError(
                f"{tile_id}: BuildVRT failed (unexpected after normalization)"
            )
        vrt_ds = None

        translate_opts = gdal.TranslateOptions(
            format="GTiff",
            outputType=out_type_map[dtype],
            noData=nodata,
            creationOptions=creation_opts,
        )
        out_ds = gdal.Translate(str(out_path), str(vrt_path), options=translate_opts)
        if out_ds is None:
            raise RuntimeError(f"{tile_id}: Translate failed")
        out_ds = None

    return tile_id, str(out_path)


def parse_args() -> argparse.Namespace:
    """
    Optional CLI overrides (you can run with no args).
    """
    p = argparse.ArgumentParser(
        description="Normalize south-up GeoTIFF tiles to north-up, then mosaic by filename groups into merged tiles."
    )
    p.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    p.add_argument("--staging-dir", type=Path, default=STAGING_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--pattern", default=INPUT_GLOB)

    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument(
        "--dtype",
        default=OUTPUT_DTYPE,
        choices=["float64", "float32", "int16", "uint16", "byte"],
    )
    p.add_argument("--nodata", type=float, default=NODATA)

    p.add_argument("--compress", default=COMPRESS)
    p.add_argument("--bigtiff", default=BIGTIFF)
    p.add_argument("--tiled", action="store_true", default=TILED)
    p.add_argument("--no-tiled", action="store_true")

    p.add_argument("--target-srs", default=TARGET_SRS)
    p.add_argument("--xres", type=float, default=XRES)
    p.add_argument("--yres", type=float, default=YRES)
    p.add_argument("--tap", action="store_true", default=TAP_ALIGN)
    p.add_argument("--no-tap", action="store_true")
    p.add_argument("--resample", default=RESAMPLE_ALG)

    p.add_argument(
        "--normalize-only-if-needed",
        action="store_true",
        default=NORMALIZE_ONLY_IF_NEEDED,
    )
    p.add_argument(
        "--overwrite-staging", action="store_true", default=OVERWRITE_STAGING
    )
    p.add_argument(
        "--clean-staging-on-start", action="store_true", default=CLEAN_STAGING_ON_START
    )

    p.add_argument("--mosaic-use-warp", action="store_true", default=MOSAIC_USE_WARP)

    p.add_argument("--warp-multithread", action="store_true", default=WARP_MULTITHREAD)
    p.add_argument("--gdal-cachemax-mb", type=int, default=GDAL_CACHEMAX_MB)

    p.add_argument("--log-level", default=LOG_LEVEL)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tiled = args.tiled and not args.no_tiled
    tap = args.tap and not args.no_tap

    log.info("INPUT_DIR   = %s", args.input_dir)
    log.info("STAGING_DIR = %s", args.staging_dir)
    log.info("OUTPUT_DIR  = %s", args.output_dir)
    log.info("WORKERS     = %d", args.workers)

    if not args.input_dir.exists():
        log.error("Input dir does not exist: %s", args.input_dir)
        return 2

    if args.clean_staging_on_start and args.staging_dir.exists():
        log.warning("Cleaning staging dir: %s", args.staging_dir)
        shutil.rmtree(args.staging_dir)

    src_tiles = _collect_inputs(args.input_dir, args.pattern)
    if not src_tiles:
        log.error(
            "No input files found in %s with pattern %s", args.input_dir, args.pattern
        )
        return 2

    # ---------------------------
    # Stage 1: Normalize to north-up into staging folder
    # ---------------------------
    log.info("Stage 1: normalizing %d tiles into %s", len(src_tiles), args.staging_dir)

    norm_failures = 0
    staged_paths: List[Path] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for src in src_tiles:
            dst = args.staging_dir / src.name
            f = ex.submit(
                _normalize_one_tile,
                src,
                dst,
                target_srs=args.target_srs,
                xres=args.xres,
                yres=args.yres,
                tap_align=tap,
                resample_alg=args.resample,
                output_dtype=args.dtype,
                nodata=args.nodata,
                compress=args.compress,
                bigtiff=args.bigtiff,
                tiled=tiled,
                warp_multithread=args.warp_multithread,
                gdal_cachemax_mb=args.gdal_cachemax_mb,
                normalize_only_if_needed=args.normalize_only_if_needed,
                overwrite=args.overwrite_staging,
            )
            futures[f] = (src, dst)

        for fut in as_completed(futures):
            src, dst = futures[fut]
            try:
                _, _, did_work = fut.result()
                staged_paths.append(dst)
                if did_work:
                    log.debug("Normalized: %s -> %s", src.name, dst.name)
            except Exception as e:
                norm_failures += 1
                log.exception("FAILED normalize: %s (%s)", src, e)

    if norm_failures:
        log.error("Normalization finished with %d failures", norm_failures)
        return 1

    # Safety: re-scan staging (ensures we use what actually exists)
    staged_tiles = _collect_inputs(args.staging_dir, args.pattern)
    if not staged_tiles:
        log.error("No staged tiles found in %s after normalization", args.staging_dir)
        return 2

    # ---------------------------
    # Stage 2: Group staged tiles and mosaic to outputs
    # ---------------------------
    tile_re = re.compile(TILE_REGEX)
    groups = group_by_tile(staged_tiles, tile_re)
    if not groups:
        log.error("No groups formed. Check TILE_REGEX or filenames.")
        return 2

    log.info("Stage 2: mosaicking %d groups into %s", len(groups), args.output_dir)
    log.info("Mosaic mode: %s", "Warp" if args.mosaic_use_warp else "VRT->Translate")

    mosaic_failures = 0
    jobs: List[Tuple[str, List[Path], Path]] = []
    for tile_id, paths in sorted(groups.items()):
        out_name = OUTPUT_NAME_TEMPLATE.format(tile_id=tile_id)
        out_path = args.output_dir / out_name
        jobs.append((tile_id, paths, out_path))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _mosaic_one_group,
                tile_id,
                in_paths,
                out_path,
                dtype=args.dtype,
                nodata=args.nodata,
                compress=args.compress,
                bigtiff=args.bigtiff,
                tiled=tiled,
                use_warp=args.mosaic_use_warp,
                warp_multithread=args.warp_multithread,
                resample_alg=args.resample,
                xres=args.xres,
                yres=args.yres,
                tap_align=tap,
                target_srs=args.target_srs,
                gdal_cachemax_mb=args.gdal_cachemax_mb,
            ): tile_id
            for tile_id, in_paths, out_path in jobs
        }

        for fut in as_completed(futures):
            tile_id = futures[fut]
            try:
                tid, outp = fut.result()
                log.info("Done: %s -> %s", tid, outp)
            except Exception as e:
                mosaic_failures += 1
                log.exception("FAILED mosaic: %s (%s)", tile_id, e)

    if mosaic_failures:
        log.error("Mosaicking finished with %d failures", mosaic_failures)
        return 1

    log.info(
        "All done. Normalized tiles in: %s | Merged tiles in: %s",
        args.staging_dir,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
