import os
import subprocess
import shutil
from pathlib import Path
import bbx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ===== USER CONFIG =====

RCLONE_REMOTE = "google_drive_lofasm"
REMOTE_BASE_PATH = "data/F-5-20-25/2025"

# Only process these dates (YYYYMMDD format)
TARGET_DATES = ["20250224", "20250218"]  # <-- Change this list as needed

LOCAL_BASE_DIR = Path("/Users/jesse/Downloads")
LOCAL_LOFASM_DIR = LOCAL_BASE_DIR / "lofasm"
LOCAL_BBX_DIR = LOCAL_BASE_DIR / "bbx"
LOCAL_NPY_DIR = LOCAL_BASE_DIR / "npy"
LOCAL_PNG_DIR = LOCAL_BASE_DIR / "pngs_from_npy_files"

LOFASM3BBX_EXEC = LOCAL_BASE_DIR / "lofasm3bbx"

VALID_BBX_SUFFIXES = ["_AA.bbx.gz", "_BB.bbx.gz", "_CC.bbx.gz", "_DD.bbx.gz"]

BATCH_SIZE = 1  # Unused now but left for potential future use

FAILED_LOG_FILE = LOCAL_BASE_DIR / "failed_files.txt"

# Cache for remote PNG listings per date
REMOTE_PNG_CACHE = {}

# ===== UTILS =====


def check_executable_exists(exec_path: Path):
    if not exec_path.is_file():
        raise FileNotFoundError(f"Executable not found: {exec_path}")
    if not os.access(exec_path, os.X_OK):
        raise PermissionError(f"Executable is not executable: {exec_path}")


def run_lofasm3bbx_batch(batch_files):
    print(f"⚙️ Converting {len(batch_files)} lofasm.gz file(s) to .bbx.gz...")
    cmd = [str(LOFASM3BBX_EXEC)] + [str(f) for f in batch_files]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: lofasm3bbx failed on batch with error: {e}")
        for f in batch_files:
            log_failed_file(f)
    except Exception as e:
        print(f"❌ Unexpected error running lofasm3bbx: {e}")
        for f in batch_files:
            log_failed_file(f)


def log_failed_file(file_path):
    try:
        with open(FAILED_LOG_FILE, "a") as log_f:
            log_f.write(str(file_path) + "\n")
    except Exception as e:
        print(f"❌ Failed to log failed file {file_path}: {e}")


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

    for file in filtered_files:
        try:
            lf = bbx.LofasmFile(str(file))
            lf.read_data()
            arr = lf.data
            base_name = file.stem.split(".")[0]
            npy_save_path = LOCAL_NPY_DIR / f"{base_name}.npy"
            np.save(npy_save_path, arr)
            print(f"✅ Converted {file.name} → {npy_save_path.name}")
        except Exception as e:
            print(f"❌ Error converting {file.name}: {e}")


def plot_npy_files():
    npy_files = list(LOCAL_NPY_DIR.glob("*.npy"))
    if not npy_files:
        print("⚠️ No .npy files found for plotting.")
        return

    for npy_file in npy_files:
        try:
            data = np.load(npy_file)
        except Exception as e:
            print(f"❌ Error loading {npy_file.name}: {e}")
            continue

        if data.ndim == 1:
            n_cols = 1024
            if data.size % n_cols == 0:
                data = data.reshape((-1, n_cols))
            else:
                data = data.reshape((-1, 1))
        elif data.ndim > 2:
            print(f"⚠️ Skipping {npy_file.name}: data has > 2 dimensions")
            continue

        rows, cols = data.shape
        slice_rows = min(rows, 4000)
        slice_cols = min(cols, 1000)
        data_to_plot = np.abs(data[:slice_rows, :slice_cols])

        base_name = npy_file.stem

        # vmax norm plot
        plt.figure()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Time (seconds)")
        plt.xticks(
            [100, 200, 300, 400, 500, 600, 700, 800, 900],
            ["10", "20", "30", "40", "50", "60", "70", "80", "90"],
        )
        plt.yticks(
            [602.25, 1204.5, 1806.75, 2409, 3011.25], ["50", "100", "150", "200", "250"]
        )
        plt.title(f"LoFASM V {base_name} (vmax=150000)")
        plt.imshow(data_to_plot, aspect="auto", norm=plt.Normalize(vmin=0, vmax=150000))
        plt.colorbar()
        vmax_save_path = LOCAL_PNG_DIR / f"{base_name}_vmax.png"
        plt.savefig(vmax_save_path)
        plt.close()
        print(f"✅ Saved: {vmax_save_path}")

        # log norm plot
        plt.figure()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Time (seconds)")
        plt.xticks(
            [100, 200, 300, 400, 500, 600, 700, 800, 900],
            ["10", "20", "30", "40", "50", "60", "70", "80", "90"],
        )
        plt.yticks(
            [602.25, 1204.5, 1806.75, 2409, 3011.25], ["50", "100", "150", "200", "250"]
        )
        plt.title(f"LoFASM V {base_name} (LogNorm)")
        plt.imshow(data_to_plot + 1e-9, aspect="auto", norm=colors.LogNorm())
        plt.colorbar()
        log_save_path = LOCAL_PNG_DIR / f"{base_name}_log.png"
        plt.savefig(log_save_path)
        plt.close()
        print(f"✅ Saved: {log_save_path}")


def cleanup_local_files():
    print("🗑️ Cleaning up local processed files and folders...")
    for folder in [LOCAL_LOFASM_DIR, LOCAL_BBX_DIR, LOCAL_NPY_DIR, LOCAL_PNG_DIR]:
        if folder.exists():
            try:
                shutil.rmtree(folder)
                print(f"Deleted folder {folder}")
            except Exception as e:
                print(f"Failed to delete folder {folder}: {e}")
        folder.mkdir(parents=True, exist_ok=True)


def rclone_list_files(remote_path):
    remote_path = remote_path.rstrip("/")
    cmd = [
        "rclone",
        "lsf",
        "--files-only",
        "--include",
        "*.lofasm.gz",
        "--max-depth",
        "1",
        f"{RCLONE_REMOTE}:{remote_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    files = result.stdout.strip().splitlines()
    return files


def rclone_list_png_files(remote_path):
    remote_path = remote_path.rstrip("/")
    cmd = [
        "rclone",
        "lsf",
        "--files-only",
        "--include",
        "*.png",
        "--max-depth",
        "1",
        f"{RCLONE_REMOTE}:{remote_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    files = result.stdout.strip().splitlines()
    return files


def rclone_download_file(remote_path, filename, local_dir):
    remote_path = remote_path.rstrip("/")
    src = f"{RCLONE_REMOTE}:{remote_path}/{filename}"
    dest = local_dir / filename
    cmd = ["rclone", "copyto", src, str(dest)]
    print(f"⬇️ Downloading {filename} from {remote_path} ...")
    subprocess.run(cmd, check=True)
    return dest


def rclone_upload_folder(local_folder, remote_folder):
    cmd = [
        "rclone",
        "sync",
        str(local_folder),
        f"{RCLONE_REMOTE}:{remote_folder}",
        "--delete-excluded",
    ]
    print(f"⬆️ Syncing {local_folder.name} to remote {remote_folder} ...")
    subprocess.run(cmd, check=True)


def get_remote_subfolders(remote_base):
    remote_base = remote_base.rstrip("/")
    cmd = [
        "rclone",
        "lsf",
        "--dirs-only",
        "--max-depth",
        "1",
        f"{RCLONE_REMOTE}:{remote_base}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    folders = result.stdout.strip().splitlines()
    folders = [
        f.rstrip("/").strip() for f in folders if f.rstrip("/").lower() != "processing"
    ]
    return folders


def organize_files_by_date(
    files, local_folder, ext, suffix="", create_folder_func=None
):
    date_folders = {}
    for file in files:
        name = file.name
        if len(name) < 8:
            print(f"⚠️ Filename too short to extract date prefix: {name}")
            continue
        date_prefix = name[:8]
        folder_name = f"{date_prefix}_{ext[1:]}"
        if ext == ".png":
            if suffix == "_vmax.png":
                folder_name = f"{date_prefix}_vmax_png"
            elif suffix == "_log.png":
                folder_name = f"{date_prefix}_log_png"
            else:
                folder_name = f"{date_prefix}_png"

        folder_path = local_folder / folder_name
        if create_folder_func:
            create_folder_func(folder_path)
        else:
            folder_path.mkdir(parents=True, exist_ok=True)

        dest_path = folder_path / name
        shutil.move(str(file), dest_path)
        date_folders.setdefault(folder_name, []).append(dest_path)
    return date_folders


def check_pngs_exist_locally(base_name):
    vmax_png = LOCAL_PNG_DIR / f"{base_name}_vmax.png"
    log_png = LOCAL_PNG_DIR / f"{base_name}_log.png"
    return vmax_png.exists() and log_png.exists()


def load_remote_png_lists(date_prefix):
    vmax_folder = f"data/processing/new_processed_data/vmax/{date_prefix}_vmax_png"
    log_folder = f"data/processing/new_processed_data/log/{date_prefix}_log_png"
    print(f"🔍 Loading remote PNG listings for date {date_prefix} ...")
    vmax_files = rclone_list_png_files(vmax_folder)
    log_files = rclone_list_png_files(log_folder)
    REMOTE_PNG_CACHE[date_prefix] = {
        "vmax": set(vmax_files),
        "log": set(log_files),
    }


def check_pngs_exist_remotely(date_prefix, base_name):
    if date_prefix not in REMOTE_PNG_CACHE:
        load_remote_png_lists(date_prefix)
    vmax_exists = f"{base_name}_vmax.png" in REMOTE_PNG_CACHE[date_prefix]["vmax"]
    log_exists = f"{base_name}_log.png" in REMOTE_PNG_CACHE[date_prefix]["log"]
    return vmax_exists and log_exists


# ===== MAIN =====


def main():
    check_executable_exists(LOFASM3BBX_EXEC)
    cleanup_local_files()

    subfolders = get_remote_subfolders(REMOTE_BASE_PATH)
    print(f"Found {len(subfolders)} subfolders in {REMOTE_BASE_PATH}: {subfolders}")

    # Gather total files count to process for progress tracking
    total_files_to_process = 0
    # Map date → list of files to process for that date
    files_to_process_by_date = {}

    for subfolder in subfolders:
        subfolder = subfolder.rstrip("/")

        # Skip subfolder if exactly a date and not in target dates
        if (
            len(subfolder) == 8
            and subfolder.isdigit()
            and subfolder not in TARGET_DATES
        ):
            continue

        remote_subfolder = f"{REMOTE_BASE_PATH.rstrip('/')}/{subfolder}"
        lofasm_files = rclone_list_files(remote_subfolder)
        if not lofasm_files:
            continue

        # Filter files by TARGET_DATES
        filtered_files = [
            f for f in lofasm_files if len(f) >= 8 and f[:8] in TARGET_DATES
        ]

        for f in filtered_files:
            base_name = f[:-9]  # strip ".lofasm.gz"
            date_prefix = f[:8]

            # Skip files already processed locally
            if check_pngs_exist_locally(base_name):
                continue
            # Skip files already processed remotely (cached per date)
            if check_pngs_exist_remotely(date_prefix, base_name):
                continue

            files_to_process_by_date.setdefault(date_prefix, []).append(
                (remote_subfolder, f)
            )
            total_files_to_process += 1

    if total_files_to_process == 0:
        print("✅ No new files to process for the target dates.")
        return

    print(f"📊 Total files to process: {total_files_to_process}\n")

    processed_count = 0

    # Process files grouped by date (full day at a time)
    for date_prefix in TARGET_DATES:
        if date_prefix not in files_to_process_by_date:
            print(f"⏭️ No files to process for date {date_prefix}")
            continue

        files_for_date = files_to_process_by_date[date_prefix]
        print(
            f"\n📅 Processing full day for {date_prefix}: {len(files_for_date)} files"
        )

        downloaded_files = []
        for remote_subfolder, filename in files_for_date:
            local_file = rclone_download_file(
                remote_subfolder, filename, LOCAL_LOFASM_DIR
            )
            downloaded_files.append(local_file)

        run_lofasm3bbx_batch(downloaded_files)

        convert_bbx_to_npy()
        plot_npy_files()

        npy_files = list(LOCAL_NPY_DIR.glob("*.npy"))
        png_files = list(LOCAL_PNG_DIR.glob("*.png"))

        organize_files_by_date(npy_files, LOCAL_NPY_DIR, ".npy")
        vmax_pngs = [f for f in png_files if f.name.endswith("_vmax.png")]
        log_pngs = [f for f in png_files if f.name.endswith("_log.png")]

        organize_files_by_date(vmax_pngs, LOCAL_PNG_DIR, ".png", suffix="_vmax.png")
        organize_files_by_date(log_pngs, LOCAL_PNG_DIR, ".png", suffix="_log.png")

        base_remote_upload = "data/processing/new_processed_data"

        for folder in LOCAL_NPY_DIR.iterdir():
            if folder.is_dir():
                rclone_upload_folder(folder, f"{base_remote_upload}/npy/{folder.name}")

        for folder in LOCAL_PNG_DIR.iterdir():
            if folder.is_dir() and folder.name.endswith("_vmax_png"):
                rclone_upload_folder(folder, f"{base_remote_upload}/vmax/{folder.name}")

        for folder in LOCAL_PNG_DIR.iterdir():
            if folder.is_dir() and folder.name.endswith("_log_png"):
                rclone_upload_folder(folder, f"{base_remote_upload}/log/{folder.name}")

        cleanup_local_files()

        processed_count += len(files_for_date)
        remaining = total_files_to_process - processed_count
        print(
            f"✅ Finished processing date {date_prefix}. Files done: {processed_count}, Remaining: {remaining}"
        )

    print("\n🎉 All target dates processed.")


if __name__ == "__main__":
    main()
