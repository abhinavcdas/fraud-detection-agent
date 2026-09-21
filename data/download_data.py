"""Download the Kaggle Credit Card Fraud Detection dataset from verified Hugging Face mirror."""

import os
import sys
import urllib.request
from pathlib import Path

DATASET_URL = "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv"
RAW_DIR = Path(__file__).resolve().parent / "raw"
TARGET_FILE = RAW_DIR / "creditcard.csv"

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {mb_down:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)")
        sys.stdout.flush()

def download_dataset(force: bool = False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    if TARGET_FILE.exists() and not force:
        size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        print(f"Dataset already exists at {TARGET_FILE} ({size_mb:.1f} MB). Skipping download.")
        return TARGET_FILE
    
    print(f"Downloading credit card fraud dataset from:\n  {DATASET_URL}")
    print(f"Target location: {TARGET_FILE}")
    
    try:
        urllib.request.urlretrieve(DATASET_URL, TARGET_FILE, reporthook=download_progress)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nFailed to download: {e}")
        if TARGET_FILE.exists():
            TARGET_FILE.unlink()
        raise

    # Validation
    size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved file size: {size_mb:.2f} MB")
    
    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        row_count = sum(1 for _ in f)
    
    print(f"Rows count: {row_count:,}")
    print(f"Columns ({len(header)}): {header[:5]} ... {header[-3:]}")
    return TARGET_FILE

if __name__ == "__main__":
    download_dataset()
