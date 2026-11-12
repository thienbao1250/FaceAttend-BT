"""
scripts/download_models.py
Tải model SCRFD và ArcFace từ insightface (buffalo_l pack).
Chạy: python scripts/download_models.py

Nội dung pack buffalo_l:
  - det_10g.onnx    → SCRFD-10GF (face detection)
  - w600k_r50.onnx  → ArcFace R50 (face recognition)
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

# buffalo_l từ insightface releases
PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# Tên file sau khi giải nén
EXPECTED_FILES = {
    "det_10g.onnx": "SCRFD-10GF (face detector)",
    "w600k_r50.onnx": "ArcFace R50 (face recognizer)",
}


def download_with_progress(url: str) -> bytes:
    print(f"Đang tải: {url}")
    response = urllib.request.urlopen(url, timeout=120)
    total = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    chunks = []
    chunk_size = 1024 * 64  # 64 KB

    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
        downloaded += len(chunk)
        if total > 0:
            pct = downloaded / total * 100
            mb = downloaded / 1_048_576
            total_mb = total / 1_048_576
            print(f"\r  {pct:.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    print()
    return b"".join(chunks)


def main():
    # Kiểm tra file đã có chưa
    all_exist = all((MODELS_DIR / f).exists() for f in EXPECTED_FILES)
    if all_exist:
        print("Tất cả model đã có sẵn trong models/:")
        for fname, desc in EXPECTED_FILES.items():
            print(f"  ✓ {fname}  ({desc})")
        return

    # Tải zip
    try:
        data = download_with_progress(PACK_URL)
    except Exception as e:
        print(f"\nLỗi khi tải: {e}")
        print("Thử tải thủ công từ:")
        print(f"  {PACK_URL}")
        print(f"Rồi giải nén vào thư mục: {MODELS_DIR}")
        sys.exit(1)

    # Giải nén
    print("Đang giải nén...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            fname = Path(member).name
            if fname in EXPECTED_FILES:
                dest = MODELS_DIR / fname
                dest.write_bytes(zf.read(member))
                size_mb = dest.stat().st_size / 1_048_576
                print(f"  ✓ {fname}  ({size_mb:.1f} MB)")

    # Kiểm tra
    missing = [f for f in EXPECTED_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        print(f"\nCảnh báo: Không tìm thấy trong zip: {missing}")
        sys.exit(1)

    print("\nHoàn thành! Model đã lưu vào:", MODELS_DIR)


if __name__ == "__main__":
    main()

