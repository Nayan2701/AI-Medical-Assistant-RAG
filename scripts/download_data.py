from pathlib import Path
import os

import gdown


def main():
    file_id = os.getenv("GDRIVE_FILE_ID")
    if not file_id:
        raise SystemExit("Set GDRIVE_FILE_ID env var (the Google Drive file id).")

    out = Path(os.getenv("DATASET_PATH", "dataset/patients_data.json"))
    out.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading to {out} ...")
    gdown.download(url, str(out), quiet=False)
    print("Done.")


if __name__ == "__main__":
    main()