import json
import shutil
import subprocess
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set, Tuple

# =========================
# USER SETTINGS (edit these)
# =========================
INDEX_FILE = r"F:/Data/AE/only_Norway_satellite_embedding_v1_annual_aef_index_2017-2025.gpkg"  # <-- change to your file path
BILLING_PROJECT = "gee-norway-embeddings"


# Only years that exist in the index will be downloaded:
YEARS = [2025]  # e.g. [2025] will do nothing if 2025 isn't in the index


OUT_DIR = r"F:/Data/AE/data/data/test2025aug"  #   <-- where to save
MAX_WORKERS = 20
SKIP_EXISTING = True

PATH_FIELD = "path"
YEAR_FIELD = "year"

# If Python can't find gcloud in your environment, set this explicitly:
# GCLOUD_EXE_OVERRIDE = r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
GCLOUD_EXE_OVERRIDE = None
# =========================


def find_gcloud_exe() -> str:
    if GCLOUD_EXE_OVERRIDE:
        p = Path(GCLOUD_EXE_OVERRIDE)
        if not p.exists():
            raise SystemExit(f"ERROR: GCLOUD_EXE_OVERRIDE not found: {p}")
        return str(p)

    for name in ("gcloud.exe", "gcloud.cmd", "gcloud"):
        p = shutil.which(name)
        if p:
            return p

    raise SystemExit(
        "ERROR: Could not find gcloud on PATH.\n"
        "Fix options:\n"
        "  1) Ensure `gcloud --version` works in the same terminal you run Python from\n"
        "  2) If using an IDE/conda env, ensure PATH includes the Cloud SDK bin folder\n"
        "  3) Set GCLOUD_EXE_OVERRIDE to the full path to gcloud.cmd"
    )


def read_records_from_geojson(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    recs = []
    for feat in data.get("features", []):
        props = feat.get("properties", {}) or {}
        recs.append(props)
    return recs


def read_records_from_gpkg(path: Path) -> List[dict]:
    try:
        import geopandas as gpd
    except ImportError:
        raise SystemExit(
            "ERROR: Reading .gpkg requires geopandas.\n"
            "Install with: pip install geopandas\n"
            "Or export the GPKG to GeoJSON and rerun."
        )

    df = gpd.read_file(path, ignore_geometry=True)
    return df.to_dict(orient="records")


def coerce_year(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(float(str(v).strip()))
    except Exception:
        return None


def local_filename_for_gsuri(gsuri: str, out_dir: Path, year: int) -> Path:
    """
    Store as OUT_DIR/<year>/<utm_zone>/<filename>.tiff
    """
    parts = gsuri.split("/")
    try:
        utm_zone = parts[parts.index("annual") + 2]  # annual / YEAR / UTM_ZONE
        filename = parts[-1]
    except Exception:
        utm_zone = "unknown_zone"
        filename = parts[-1]
    return out_dir / str(year) / utm_zone / filename


def download_one(
    gcloud_exe: str,
    gsuri: str,
    dest_file: Path,
    billing_project: str,
    skip_existing: bool,
):
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and dest_file.exists() and dest_file.stat().st_size > 0:
        return ("skipped", gsuri, None)

    cmd = [
        gcloud_exe,
        "storage",
        "cp",
        f"--billing-project={billing_project}",
        gsuri,
        str(dest_file),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        return ("error", gsuri, msg)

    return ("ok", gsuri, None)


def main():
    gcloud_exe = find_gcloud_exe()
    print(f"Using gcloud: {gcloud_exe}")

    index_path = Path(INDEX_FILE)
    out_dir = Path(OUT_DIR)

    if not index_path.exists():
        raise SystemExit(f"ERROR: INDEX_FILE not found: {index_path}")

    suffix = index_path.suffix.lower()
    if suffix in (".geojson", ".json"):
        records = read_records_from_geojson(index_path)
    elif suffix == ".gpkg":
        records = read_records_from_gpkg(index_path)
    else:
        raise SystemExit("ERROR: Unsupported index format. Use .gpkg or .geojson/.json")

    if not records:
        raise SystemExit("ERROR: No features/rows found in index file.")

    wanted_years = set(int(y) for y in YEARS)

    # Build job set (year, path) strictly from index (no synthesis)
    jobs: Set[Tuple[int, str]] = set()
    years_seen: Set[int] = set()

    for r in records:
        p = r.get(PATH_FIELD)
        if not p:
            continue
        y = coerce_year(r.get(YEAR_FIELD))
        if y is None:
            continue

        years_seen.add(y)
        if y in wanted_years:
            jobs.add((y, str(p)))

    years_seen_sorted = sorted(years_seen)
    print(f"Years requested: {sorted(wanted_years)}")
    print(f"Years available in index: {years_seen_sorted}")

    if not jobs:
        raise SystemExit(
            f"No paths found for requested YEARS={sorted(wanted_years)}.\n"
            f"Years in index: {years_seen_sorted}"
        )

    jobs_list = sorted(jobs)
    print(f"Total unique downloads to attempt: {len(jobs_list)}")

    ok = skipped = err = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for year, gsuri in jobs_list:
            dest = local_filename_for_gsuri(gsuri, out_dir, year)
            futures.append(
                ex.submit(
                    download_one,
                    gcloud_exe,
                    gsuri,
                    dest,
                    BILLING_PROJECT,
                    SKIP_EXISTING,
                )
            )

        for f in as_completed(futures):
            status, gsuri, msg = f.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                err += 1
                print(f"\nERROR downloading: {gsuri}\n{msg}\n")

    print("\nDone.")
    print(f"Downloaded: {ok}")
    print(f"Skipped:    {skipped}")
    print(f"Errors:     {err}")
    print(f"Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
