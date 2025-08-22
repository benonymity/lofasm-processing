import os
import subprocess
import shutil
import sys
import time
import pickle
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import bbx
import matplotlib.pyplot as plt
from matplotlib import colors
import multiprocessing
import re
from contextlib import contextmanager

# ======================================================================================================================
# USER CONFIGURATION
# ======================================================================================================================

# List of rclone remotes/configs to cycle through if one breaks - placeholders
RCLONE_REMOTES = [
    "google_drive_lofasm_1",
    "google_drive_lofasm_2",
    # "google_drive_lofasm_3",
    # Add more as needed
]

# Remote base path containing all data (under /data/), excluding "processing" folder
REMOTE_BASE_PATH = "google_drive_lofasm_1:lofasm/data"
UPLOAD_BASE_PATH = REMOTE_BASE_PATH + "/processing/new_processed_data"

# Local directories for processing
LOCAL_BASE_DIR = Path("/home/bbassett/research/processing")
# LOCAL_BASE_DIR = Path(os.path.expanduser("~/Developer/LoFASM-processing/final"))
LOCAL_LOFASM_DIR = LOCAL_BASE_DIR / "lofasm"
LOCAL_BBX_DIR = LOCAL_BASE_DIR / "bbx"
LOCAL_NPY_DIR = LOCAL_BASE_DIR / "npy"
LOCAL_PNG_VMAX_DIR = LOCAL_BASE_DIR / "pngs_vmax"
LOCAL_PNG_LOG_DIR = LOCAL_BASE_DIR / "pngs_log"


# Path to lofasm3bbx executable
LOFASM3BBX_EXEC = LOCAL_BASE_DIR / "lofasm2bbx"

# rclone config file (pass to all rclone commands). Use path relative to this script.
RCLONE_CONFIG_FILE = LOCAL_BASE_DIR / "rclone.conf"

# Number of parallel jobs to use (max CPU cores)
MAX_WORKERS = multiprocessing.cpu_count() // 2

# Valid .bbx suffixes
VALID_BBX_SUFFIXES = ["_AA.bbx.gz", "_BB.bbx.gz", "_CC.bbx.gz", "_DD.bbx.gz"]

# Minimal logging control
VERBOSE = False

# rclone progress frequency
RC_PROGRESS_STATS_SECONDS = 2

# Caching configuration
USE_CACHE = True
DATE_DIR_CACHE_TTL_SECONDS = 24 * 60 * 60
PROCESSED_DATES_CACHE_TTL_SECONDS = 5 * 60
CACHE_DIR = LOCAL_BASE_DIR / ".cache"
CACHE_FILE = CACHE_DIR / "discovery_cache.pkl"


def _cache_get(key: str, ttl_seconds: int):
    if not USE_CACHE:
        return None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not CACHE_FILE.exists():
            return None
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        ts = cache.get(f"{key}_ts")
        if ts is None:
            return None
        if (time.time() - ts) > ttl_seconds:
            return None
        return cache.get(key)
    except Exception:
        return None


def _cache_set(key: str, value):
    if not USE_CACHE:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = {}
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
        cache[key] = value
        cache[f"{key}_ts"] = time.time()
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass


# ======================================================================================================================
# UTILS
# ======================================================================================================================


def rclone_run_with_failover(args, capture_output=False, text=True, check=True):
    """
    Run an rclone command with failover over multiple remotes.
    args: list of rclone command parts after remote name, e.g. ["lsf", "data/", ...]
    Returns: subprocess.CompletedProcess of first successful run.
    Raises: subprocess.CalledProcessError if all fail.
    """
    last_exc = None
    for remote in RCLONE_REMOTES:
        new_args = []
        for arg in args:
            if ":" in arg:
                prefix, rest = arg.split(":", 1)
                # Replace prefix with current remote name
                new_args.append(f"{remote}:{rest}")
            else:
                new_args.append(arg)
        cmd = ["rclone", "--config", str(RCLONE_CONFIG_FILE)] + new_args
        try:
            if VERBOSE:
                print(f"🛰️ Trying rclone remote '{remote}': {' '.join(cmd)}")
            completed = subprocess.run(
                cmd, capture_output=capture_output, text=text, check=check
            )
            return completed
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Rclone failed on remote '{remote}': {' '.join(cmd)}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
            print("…trying next remote…")
            last_exc = e
    print("❌ All rclone remotes failed for command.")
    raise last_exc


def check_executable_exists(exec_path):
    # Accepts either a Path or a string
    if isinstance(exec_path, str):
        exec_path_str = os.path.expanduser(exec_path)
        exec_path_obj = Path(exec_path_str)
    else:
        exec_path_obj = (
            exec_path.expanduser() if hasattr(exec_path, "expanduser") else exec_path
        )

    if not exec_path_obj.is_file():
        raise FileNotFoundError(f"Executable not found: {exec_path_obj}")
    if not os.access(str(exec_path_obj), os.X_OK):
        raise PermissionError(f"Executable is not executable: {exec_path_obj}")


def cleanup_local_files():
    print("🗑️ Cleaning up local processed files and folders... 🖕💥")
    for folder in [
        LOCAL_LOFASM_DIR,
        LOCAL_BBX_DIR,
        LOCAL_NPY_DIR,
        LOCAL_PNG_VMAX_DIR,
        LOCAL_PNG_LOG_DIR,
    ]:
        if folder.exists():
            try:
                shutil.rmtree(folder)
                print(f"Deleted folder {folder}")
            except Exception as e:
                print(f"Failed to delete folder {folder}: {e}")
        folder.mkdir(parents=True, exist_ok=True)


def get_all_lofasm_files(remote_base: str):
    # Recursive list of all .lofasm.gz files excluding /processing/ folder anywhere in path
    args = [
        "lsf",
        f"{remote_base}",
        "--include",
        "**/*.lofasm.gz",
        "--exclude",
        "**/processing/**",
        "--recursive",
    ]
    print(
        "🖕 Fetching lofasm file list from remote, skipping /processing/ folder... 🦄💨"
    )
    result = rclone_run_with_failover(args, capture_output=True)
    files = result.stdout.strip().splitlines()
    print(f"🖕 Found {len(files)} lofasm files outside 'processing' folder.")
    return files


def get_lofasm_date_directories(remote_base: str):
    """
    Return a list of (date, subpath) tuples for directories under top-level F-* folders
    that look like YYYYMMDD. The subpath is relative to remote_base and may be nested
    (e.g., "F-5-20-23/20210623" or "F-5-20-23/some/thing/20210623").
    Only traverses F-* folders; ignores others like 'processing'.
    """
    # 0) Try cache
    cached = _cache_get("date_dirs", DATE_DIR_CACHE_TTL_SECONDS)
    if cached is not None:
        if VERBOSE:
            print(f"🗃️ Using cached date directories ({len(cached)})")
        return cached

    # 1) List top-level folders and filter for F-*
    top_args = [
        "lsf",
        f"{remote_base}",
        "--dirs-only",
        "--max-depth",
        "1",
    ]
    print("🖕 Discovering top-level 'F-*' folders (dirs only) …")
    try:
        top_result = rclone_run_with_failover(top_args, capture_output=True)
        top_dirs = [
            d.strip().rstrip("/")
            for d in top_result.stdout.strip().splitlines()
            if d.strip()
        ]
    except subprocess.CalledProcessError:
        print("⚠️ Failed to list top-level directories under remote base.")
        return []

    f_roots = [d for d in top_dirs if d.startswith("F-")]
    if not f_roots:
        print("⚠️ No 'F-*' folders found under remote base.")
        return []

    # 2) For each F-* folder, list subdirectories up to depth 4 and collect YYYYMMDD dirs
    date_dir_pairs = []
    for f_root in f_roots:
        sub_args = [
            "lsf",
            f"{remote_base}/{f_root}",
            "--dirs-only",
            "--recursive",
            "--max-depth",
            "5",  # covers 0-3 nested directories before the date folder
        ]
        try:
            sub_result = rclone_run_with_failover(sub_args, capture_output=True)
            sub_dirs = [
                d.strip().rstrip("/")
                for d in sub_result.stdout.strip().splitlines()
                if d.strip()
            ]
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to list subdirectories under '{f_root}'. Skipping.")
            continue

        for rel in sub_dirs:
            # rel is relative to {remote_base}/{f_root}
            # It can be like '20210623' or 'some/dir/20210623'
            last_component = rel.split("/")[-1]
            if re.fullmatch(r"\d{8}", last_component):
                full_subpath = f"{f_root}/{rel}"
                date_dir_pairs.append((last_component, full_subpath))

    # De-duplicate
    seen = set()
    unique_pairs = []
    for date, subpath in date_dir_pairs:
        key = (date, subpath)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((date, subpath))

    print(f"🖕 Found {len(unique_pairs)} date directories under F-* folders.")

    # Save to cache
    _cache_set("date_dirs", unique_pairs)

    return unique_pairs


def list_lofasm_files_in_directory(remote_dir: str):
    """
    List .lofasm.gz files directly inside the given remote directory (non-recursive).
    Returns filenames (no path components).
    """
    args = [
        "lsf",
        f"{remote_dir}",
        "--include",
        "*.lofasm.gz",
        # Non-recursive on purpose: list files only in this directory
    ]
    try:
        result = rclone_run_with_failover(args, capture_output=True)
        files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
        return files
    except subprocess.CalledProcessError:
        print(f"⚠️ Failed to list files in {remote_dir}.")
        return []


def rclone_copy_directory(remote_dir: str, local_dir: Path):
    """Copy an entire remote directory to a local directory with progress output.
    If the local directory already exists and is non-empty, skip copying.
    """
    # Check if directory exists and has files
    if local_dir.exists():
        try:
            files = list(local_dir.iterdir())
            if files:
                print(
                    f"✅ Skipping copy: '{local_dir}' already exists with {len(files)} items"
                )
                return
        except Exception as e:
            print(f"⚠️ Error checking directory {local_dir}: {e}")
            # If we can't check, assume it's corrupted and remove it
            try:
                shutil.rmtree(local_dir)
                print(f"🗑️ Removed corrupted directory {local_dir}")
            except Exception as rm_e:
                print(f"❌ Failed to remove corrupted directory {local_dir}: {rm_e}")
                raise

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"📥 Copying '{remote_dir}' → '{local_dir}'")

    # For remote-to-local copying, we need to use the first remote from the list
    # and construct the command manually to avoid the failover logic interfering
    remote_name = remote_dir.split(":")[0]
    remote_path = remote_dir.split(":", 1)[1] if ":" in remote_dir else remote_dir

    cmd = [
        "rclone",
        "--config",
        str(RCLONE_CONFIG_FILE),
        "copy",
        f"{remote_name}:{remote_path}",
        str(local_dir),
        "--transfers",
        "16",
        "--checkers",
        "32",
        "--fast-list",
        "--tpslimit",
        "10",
        "--tpslimit-burst",
        "10",
        "--drive-chunk-size",
        "128M",
        "-P",
        "--stats",
        str(RC_PROGRESS_STATS_SECONDS) + "s",
    ]

    try:
        print(f"🔄 Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"✅ Copy completed: {remote_dir} → {local_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to copy directory {remote_dir}: {e}")
        raise


def rclone_download_file(remote_path, filename, local_dir):
    src = f"{remote_path}/{filename}"
    dest = local_dir / filename
    print(f"🎉 Downloading {filename} from {remote_path} … grab it! 🦄💨 🖕")

    # For remote-to-local copying, construct the command manually
    remote_name = remote_path.split(":")[0]
    remote_file_path = (
        remote_path.split(":", 1)[1] if ":" in remote_path else remote_path
    )

    cmd = [
        "rclone",
        "--config",
        str(RCLONE_CONFIG_FILE),
        "copyto",
        f"{remote_name}:{remote_file_path}/{filename}",
        str(dest),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {filename}: {e}")
        raise

    return dest


def rclone_upload_folder(local_folder, remote_folder):
    print(f"⬆️ Syncing {local_folder.name} → {remote_folder}")
    args = [
        "sync",
        str(local_folder),
        remote_folder,
        "--transfers",
        "16",
        "--checkers",
        "32",
        "--fast-list",
        "--tpslimit",
        "10",
        "--tpslimit-burst",
        "10",
        "--drive-chunk-size",
        "128M",
        "--delete-excluded",
        "-P",
        "--stats",
        str(RC_PROGRESS_STATS_SECONDS) + "s",
    ]
    rclone_run_with_failover(args)


def run_lofasm3bbx_batch(batch_files):
    total = len(batch_files)
    print(f"⚙️ Converting {total} lofasm.gz file(s) to .bbx.gz…")

    def run_single(f):
        cmd = [str(LOFASM3BBX_EXEC), str(f)]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() if e.stderr else "Unknown error"
            print(f"\n❌ Error converting {f.name}: {err_msg}")
        return f

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single, f): f for f in batch_files}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            f = futures[future]
            # Single-line progress
            sys.stdout.write(f"\r  → [{completed}/{total}] {f.name}")
            sys.stdout.flush()
    print("")


def convert_single_bbx_to_npy(file):
    try:
        lf = bbx.LofasmFile(str(file))
        lf.read_data()
        arr = lf.data.astype(np.float32)
        base_name = file.stem.split(".")[0]
        npy_save_path = LOCAL_NPY_DIR / f"{base_name}.npy"
        np.save(npy_save_path, arr)
    except Exception as e:
        print(f"❌ Error converting {file.name}: {e}")


def convert_bbx_to_npy():
    all_bbx_files = list(LOCAL_BBX_DIR.rglob("*.bbx.gz"))
    filtered_files = [
        f
        for f in all_bbx_files
        if any(f.name.endswith(suffix) for suffix in VALID_BBX_SUFFIXES)
    ]
    if not filtered_files:
        print("⚠️ No .bbx.gz files found for conversion.")
        return

    LOCAL_NPY_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"🖕 Converting {len(filtered_files)} .bbx.gz files to .npy with max {MAX_WORKERS} cores…"
    )
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 10)) as executor:
        futures = [
            executor.submit(convert_single_bbx_to_npy, f) for f in filtered_files
        ]
        completed = 0
        total = len(futures)
        for _ in as_completed(futures):
            completed += 1
            # Single-line progress
            sys.stdout.write(f"\r  → [{completed}/{total}] .npy")
            sys.stdout.flush()
    print("")


def plot_and_save(
    data, base_name, title_suffix, cmap_norm, save_dir: Path, filename_suffix: str
):
    plt.figure()
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (seconds)")
    x = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    x_labels = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
    y = [602.25, 1204.5, 1806.75, 2409, 3011.25]
    y_labels = ["50", "100", "150", "200", "250"]
    plt.xticks(x, x_labels)
    plt.yticks(y, y_labels)
    plt.title(f"LoFASM V {base_name} {title_suffix}")
    plt.imshow(data, aspect="auto", norm=cmap_norm)
    plt.colorbar()

    # Safe filenames for vmax and log
    suffix_safe = filename_suffix.replace(" ", "_")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{base_name}_{suffix_safe}.png"
    plt.savefig(save_path)
    plt.close()
    if VERBOSE:
        if filename_suffix == "vmax":
            print(f"✨ Saved vmax plot {save_path.name}")
        else:
            print(f"✨ Saved log plot {save_path.name}")


def render_one_npy(file_path):
    """
    Render a single .npy file to PNG plots (vmax and log versions).
    This function must be defined at module level for multiprocessing compatibility.
    Based on the working plotting code.
    """
    import os

    # Set backend before importing pyplot in each process
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import numpy as np

    try:
        data = np.load(file_path)
    except Exception as e:
        return (file_path.name, False, f"Error loading: {e}")

    if data.ndim == 1:
        n_cols = 1024
        if data.size % n_cols == 0:
            data = data.reshape((-1, n_cols))
        else:
            data = data.reshape((-1, 1))
    elif data.ndim > 2:
        return (file_path.name, False, "data has > 2 dimensions")

    rows, cols = data.shape
    slice_rows = min(rows, 4000)
    slice_cols = min(cols, 1000)
    data_to_plot = np.abs(data[:slice_rows, :slice_cols])

    base_name = file_path.stem

    # Prepare output paths
    vmax_path = LOCAL_PNG_VMAX_DIR / f"{base_name}_vmax.png"
    log_path = LOCAL_PNG_LOG_DIR / f"{base_name}_log.png"
    LOCAL_PNG_VMAX_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PNG_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Plot vmax - using the working approach
    try:
        plt.figure()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Time (seconds)")
        x = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        x_labels = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
        y = [602.25, 1204.5, 1806.75, 2409, 3011.25]
        y_labels = ["50", "100", "150", "200", "250"]
        plt.xticks(x, x_labels)
        plt.yticks(y, y_labels)
        plt.title(f"LoFASM V {base_name} (vmax=150000)")
        plt.imshow(data_to_plot, aspect="auto", norm=plt.Normalize(vmin=0, vmax=150000))
        plt.colorbar()
        plt.savefig(vmax_path)
        plt.close()
    except Exception as e:
        return (file_path.name, False, f"Error plotting vmax: {e}")

    # Plot log - using the working approach
    try:
        plt.figure()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Time (seconds)")
        plt.xticks(x, x_labels)
        plt.yticks(y, y_labels)
        plt.title(f"LoFASM V {base_name} (LogNorm)")
        plt.imshow(data_to_plot + 1e-9, aspect="auto", norm=colors.LogNorm())
        plt.colorbar()
        plt.savefig(log_path)
        plt.close()
    except Exception as e:
        return (file_path.name, False, f"Error plotting log: {e}")

    return (file_path.name, True, "")


def plot_npy_files():
    """
    Parallel plotting of .npy files using a process pool for matplotlib rendering.
    Each process sets MPLBACKEND to 'Agg' for safe, isolated plotting.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    npy_files = list(LOCAL_NPY_DIR.glob("*.npy"))
    if not npy_files:
        print("⚠️ No .npy files found for plotting.")
        return

    total = len(npy_files)
    processed_count = 0

    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, 12)) as pool:
        futures = {pool.submit(render_one_npy, f): f for f in npy_files}
        for i, future in enumerate(as_completed(futures), 1):
            fname, ok, msg = future.result()
            if ok:
                processed_count += 1
            else:
                print(f"❌ {fname}: {msg}")
            sys.stdout.write(f"\r  → [{i}/{total}] plotted")
            sys.stdout.flush()

    print(f"\n🎉 Plotting complete for {processed_count} files. 🦄💥🖕")


def get_processed_dates_from_pngs(processing_folder: Path):
    """Scan all png files in processing folder and return a set of YYYYMMDD strings indicating processed dates.

    Uses a short-lived cache since processed outputs can change during a session.
    """
    # Here, processing_folder is remote; this helper was only used for local paths earlier.
    # We keep it for completeness if used elsewhere.
    if not processing_folder.exists():
        return set()
    png_files = list(processing_folder.glob("*.png"))
    dates = set()
    for f in png_files:
        name = f.stem
        if len(name) >= 8 and name[:8].isdigit():
            dates.add(name[:8])
    print(
        f"🖕 Found {len(dates)} processed date batches already done (from PNGs). Skipping those! 🤘🦄"
    )
    return dates


# ======================================================================================================================
# MAIN PROCESS
# ======================================================================================================================


def main():
    check_executable_exists(LOFASM3BBX_EXEC)

    # Make sure local processing directories exist
    for folder in [
        LOCAL_LOFASM_DIR,
        LOCAL_BBX_DIR,
        LOCAL_NPY_DIR,
        LOCAL_PNG_VMAX_DIR,
        LOCAL_PNG_LOG_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    # Discover date directories under F-* folders without listing all files
    date_dir_pairs = get_lofasm_date_directories(REMOTE_BASE_PATH)
    if not date_dir_pairs:
        print("⚠️ No date directories found under F-* folders. Exiting... 🦄🖕")
        return

    # Get processed dates from PNGs in remote processing folder (cached)
    remote_processing_folder = f"{REMOTE_BASE_PATH}/processing"
    cached_proc = _cache_get("processed_dates", PROCESSED_DATES_CACHE_TTL_SECONDS)
    try:
        if cached_proc is None:
            args = [
                "lsf",
                f"{remote_processing_folder}",
                "--include",
                "*.png",
                "--recursive",
            ]
            result = rclone_run_with_failover(args, capture_output=True)
            remote_png_files = result.stdout.strip().splitlines()
            _cache_set("processed_dates_raw", remote_png_files)
        else:
            if VERBOSE:
                print("🗃️ Using cached processed dates raw list")
            remote_png_files = (
                _cache_get("processed_dates_raw", PROCESSED_DATES_CACHE_TTL_SECONDS)
                or []
            )
    except subprocess.CalledProcessError:
        print(
            "⚠️ Failed to list PNG files in processing folder. Assuming no files processed yet."
        )
        remote_png_files = []

    processed_dates = set()
    for f in remote_png_files:
        name = Path(f).stem
        if len(name) >= 8 and name[:8].isdigit():
            processed_dates.add(name[:8])

    _cache_set("processed_dates", processed_dates)
    print(
        f"🖕 Skipping {len(processed_dates)} date batches already processed (from remote PNGs). 🤘🦄"
    )

    # Group date directories by date so we process all folders for the same day together
    from collections import defaultdict

    date_to_subpaths = defaultdict(list)
    for date, subpath in date_dir_pairs:
        date_to_subpaths[date].append(subpath)

    # Process each date lazily: copy directories then process recursively (newest first)
    unprocessed_dates = sorted(
        [d for d in date_to_subpaths.keys() if d not in processed_dates],
        reverse=True,
    )
    for date in unprocessed_dates:
        date_start = time.time()

        remote_dirs_for_date = [
            f"{REMOTE_BASE_PATH}/{sp}" for sp in date_to_subpaths[date]
        ]

        # Copy entire remote directories for this date into a single consolidated local directory
        t_copy_start = time.time()
        local_date_root = LOCAL_LOFASM_DIR / date
        local_date_root.mkdir(parents=True, exist_ok=True)

        print(
            f"📁 Consolidating {len(date_to_subpaths[date])} subpaths for date {date} into single local directory"
        )

        for sp, remote_date_dir in zip(date_to_subpaths[date], remote_dirs_for_date):
            print(f"📁 Processing subpath: {sp}")
            print(f"   Remote: {remote_date_dir}")

            try:
                # Copy directly into the date root directory, not into subdirectories
                rclone_copy_directory(remote_date_dir, local_date_root)
            except Exception as e:
                print(f"❌ Failed to copy directory {remote_date_dir}: {e}")
                continue

        # Collect all .lofasm.gz files recursively for this date
        print(f"🔍 Scanning for .lofasm.gz files in {local_date_root}")
        lofasm_files = list(local_date_root.rglob("*.lofasm.gz"))
        print(f"📊 Found {len(lofasm_files)} .lofasm.gz files locally for date {date}")

        if not lofasm_files:
            print(f"⚠️ No .lofasm.gz files found locally for date {date}. Skipping.")
            continue

        t_copy = time.time() - t_copy_start
        print(
            f"\n🖕 Processing date {date} with total {len(lofasm_files)} files — Time to shred! 🦄💥 🤘"
        )

        # Convert downloaded lofasm files to bbx
        t_bbx_start = time.time()
        if lofasm_files:
            run_lofasm3bbx_batch(lofasm_files)
        else:
            print(f"⚠️ No lofasm files found locally to convert for date {date}!")
        t_bbx = time.time() - t_bbx_start

        # Convert bbx files to npy using max cores
        t_npy_start = time.time()
        convert_bbx_to_npy()
        t_npy = time.time() - t_npy_start

        # Plot npy files
        t_plot_start = time.time()
        plot_npy_files()
        t_plot = time.time() - t_plot_start

        # Upload npy and png folders to remote processing folder, grouped by date
        t_up_start = time.time()
        rclone_upload_folder(LOCAL_NPY_DIR, f"{UPLOAD_BASE_PATH}/npy/{date}_npy")
        rclone_upload_folder(
            LOCAL_PNG_VMAX_DIR, f"{UPLOAD_BASE_PATH}/vmax/{date}_pngs_vmax"
        )
        rclone_upload_folder(
            LOCAL_PNG_LOG_DIR, f"{UPLOAD_BASE_PATH}/log/{date}_pngs_log"
        )
        t_upload = time.time() - t_up_start

        # Cleanup local folders before next date batch
        cleanup_local_files()
        t_total = time.time() - date_start

        # Timing breakdown
        print(
            "\n⏱️ Date timing (s): "
            f"download+copy={t_copy:.1f} | bbx={t_bbx:.1f} | npy={t_npy:.1f} | plot={t_plot:.1f} | upload={t_upload:.1f} | total={t_total:.1f}"
        )

    print("\n🎉 All batches processed. You crushed it! 🦄🔥🖕")


if __name__ == "__main__":
    main()
