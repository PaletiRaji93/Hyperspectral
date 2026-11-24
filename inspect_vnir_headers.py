from pathlib import Path
import re
import csv

ROOT = Path(r"E:\Infotact_projects\Hyperspectral\dataset\raw\VNIR")

# ENVI data type code -> bytes per value
DATA_TYPES = {
    1: 1, 2: 2, 3: 4, 4: 4, 5: 8,
    6: 8, 9: 16, 12: 2, 13: 4,
    14: 8, 15: 8
}

def get_value(header, key):
    pattern = rf"(?im)^\s*{re.escape(key)}\s*=\s*(.+?)(?=\n\s*\w[\w -]*\s*=|\Z)"
    match = re.search(pattern, header, re.S)
    return match.group(1).strip() if match else None

def inspect_header(path):
    content = path.read_text(errors="ignore")

    fields = {}
    for key in ["samples", "lines", "bands", "data type",
                "interleave", "byte order", "header offset"]:
        fields[key] = get_value(content, key)

    wavelength_text = get_value(content, "wavelength")
    wavelengths = []
    if wavelength_text:
        wavelengths = re.findall(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
            wavelength_text.strip().strip("{}")
        )

    folder = path.parent
    data_files = list(folder.glob("*.hyspex"))
    csv_files = list(folder.glob("*.csv"))
    jpg_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))

    try:
        lines = int(fields["lines"])
        samples = int(fields["samples"])
        bands = int(fields["bands"])
        dtype = int(fields["data type"])
        offset = int(fields["header offset"] or 0)

        expected_bytes = (
            lines * samples * bands * DATA_TYPES[dtype] + offset
        )
    except (TypeError, ValueError, KeyError):
        expected_bytes = None

    actual_bytes = data_files[0].stat().st_size if data_files else None

    return {
        "folder": folder.name,
        "lines": fields["lines"],
        "samples": fields["samples"],
        "bands": fields["bands"],
        "data_type": fields["data type"],
        "interleave": fields["interleave"],
        "byte_order": fields["byte order"],
        "header_offset": fields["header offset"],
        "wavelength_count": len(wavelengths),
        "first_wavelength": wavelengths[0] if wavelengths else None,
        "last_wavelength": wavelengths[-1] if wavelengths else None,
        "hyspex_bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "size_matches": (
            actual_bytes == expected_bytes
            if actual_bytes is not None and expected_bytes is not None
            else None
        ),
        "hyspex_count": len(data_files),
        "jpg_count": len(jpg_files),
        "csv_count": len(csv_files),
        "csv_files": "; ".join(x.name for x in csv_files)
    }


results = []

for folder in sorted(ROOT.iterdir()):
    if folder.is_dir():
        hdr_files = list(folder.glob("*.hdr"))
        if len(hdr_files) == 1:
            results.append(inspect_header(hdr_files[0]))
        else:
            results.append({
                "folder": folder.name,
                "error": f"Expected 1 HDR, found {len(hdr_files)}"
            })

output = ROOT.parent.parent / "vnir_inventory.csv"

if results:
    columns = sorted({key for row in results for key in row})
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

print("Total inspected folders:", len(results))
print("Inventory saved to:", output)

for row in results:
    print(
        row.get("folder"),
        "| dimensions:", row.get("lines"), "x",
        row.get("samples"),
        "| bands:", row.get("bands"),
        "| wavelengths:", row.get("wavelength_count"),
        "| size matches:", row.get("size_matches"),
        "| CSV files:", row.get("csv_count")
    )