
from pathlib import Path

import numpy as np
import pandas as pd
import spectral


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_DIR = ROOT / "dataset" / "raw" / "SWIR" / "13_G"

HDR_PATH = SAMPLE_DIR / "13_G.hdr"

OUTPUT_DIR = ROOT / "artifacts" / "dataset_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD CUBE
# ------------------------------------------------------------

print("=" * 70)
print("SWIR SATURATION PATTERN ANALYSIS")
print("=" * 70)

img = spectral.open_image(str(HDR_PATH))
cube = np.asarray(img.load())

saturation = cube == 65535

affected_bands = np.where(
    np.any(saturation, axis=(0, 1))
)[0]

print("\nAffected bands:", affected_bands.tolist())


# ------------------------------------------------------------
# ANALYZE EACH AFFECTED BAND
# ------------------------------------------------------------

records = []

for band_index in affected_bands:

    band_mask = saturation[:, :, band_index]

    rows, cols = np.where(band_mask)

    print("\n" + "-" * 60)

    print("Band:", band_index)

    print(
        "Wavelength:",
        float(img.bands.centers[band_index])
    )

    print("Saturated pixels:", len(rows))

    print("Affected rows:", len(np.unique(rows)))

    print("Affected columns:", len(np.unique(cols)))

    print(
        "Row range:",
        (int(rows.min()), int(rows.max()))
    )

    print(
        "Column range:",
        (int(cols.min()), int(cols.max()))
    )

    print(
        "First 20 coordinates:",
        list(zip(rows[:20].tolist(), cols[:20].tolist()))
    )

    records.append({
        "band_index": int(band_index),
        "wavelength_nm": float(
            img.bands.centers[band_index]
        ),
        "saturated_pixels": len(rows),
        "affected_rows": len(np.unique(rows)),
        "affected_columns": len(np.unique(cols)),
        "min_row": int(rows.min()),
        "max_row": int(rows.max()),
        "min_column": int(cols.min()),
        "max_column": int(cols.max())
    })


# ------------------------------------------------------------
# SAVE REPORT
# ------------------------------------------------------------

report = pd.DataFrame(records)

report_path = OUTPUT_DIR / "swir_saturation_pattern_report.csv"

report.to_csv(report_path, index=False)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SATURATION PATTERN ANALYSIS COMPLETED")
print("=" * 70)

print("Report:", report_path)
