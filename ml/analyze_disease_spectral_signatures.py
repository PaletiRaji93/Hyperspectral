
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "outputs" / "ml_datasets"
    / "multisensor_leaf_ml_dataset.csv"
)

OUTPUT_DIR = ROOT / "artifacts" / "spectral_signature_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSORS = ["VNIR", "SWIR"]
GENOTYPES = ["EPO_67", "EPO_68"]


# ============================================================
# FEATURE DISCOVERY
# ============================================================

def discover_features(df, sensor):

    prefix = f"{sensor}_mean_"

    columns = [
        col for col in df.columns
        if col.startswith(prefix)
    ]

    features = []

    for col in columns:

        match = re.search(r"mean_([0-9.]+)nm$", col)

        if not match:
            continue

        wavelength = float(match.group(1))

        features.append((wavelength, col))

    features.sort(key=lambda x: x[0])

    return features


# ============================================================
# EFFECT SIZE
# ============================================================

def calculate_cohens_d(infected, noninfected):

    n1 = len(infected)
    n0 = len(noninfected)

    mean1 = np.mean(infected)
    mean0 = np.mean(noninfected)

    std1 = np.std(infected, ddof=1)
    std0 = np.std(noninfected, ddof=1)

    denominator = n1 + n0 - 2

    if denominator <= 0:
        return np.nan

    pooled_variance = (
        ((n1 - 1) * std1 ** 2)
        + ((n0 - 1) * std0 ** 2)
    ) / denominator

    pooled_std = np.sqrt(pooled_variance)

    if pooled_std == 0:
        return np.nan

    return (mean1 - mean0) / pooled_std


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================

def analyze_sensor_genotype(df, sensor, genotype):

    features = discover_features(df, sensor)

    if not features:
        raise ValueError(
            f"No {sensor} mean spectral features found."
        )

    subset = df[df["Genotype"] == genotype]

    infected = subset[subset["Disease_Label"] == 1]
    noninfected = subset[subset["Disease_Label"] == 0]

    if len(infected) < 2 or len(noninfected) < 2:
        raise ValueError(
            f"Insufficient records for {sensor}, {genotype}"
        )

    print(f"\n{sensor} - {genotype}")
    print("-" * 55)
    print(f"Infected leaves: {len(infected)}")
    print(f"Non-infected leaves: {len(noninfected)}")

    records = []

    for wavelength, column in features:

        infected_values = infected[column].astype(float).values
        noninfected_values = noninfected[column].astype(float).values

        infected_mean = np.mean(infected_values)
        noninfected_mean = np.mean(noninfected_values)

        difference = infected_mean - noninfected_mean

        effect_size = calculate_cohens_d(
            infected_values,
            noninfected_values
        )

        records.append({
            "Sensor": sensor,
            "Genotype": genotype,
            "Wavelength_nm": wavelength,
            "Infected_Mean": infected_mean,
            "Noninfected_Mean": noninfected_mean,
            "Difference": difference,
            "Absolute_Difference": abs(difference),
            "Cohens_d": effect_size,
            "Absolute_Cohens_d": abs(effect_size)
            if np.isfinite(effect_size) else np.nan,
        })

    result = pd.DataFrame(records)

    return result, features, infected, noninfected


# ============================================================
# PLOTTING
# ============================================================

def plot_spectral_curves(
    sensor,
    genotype,
    features,
    infected,
    noninfected
):

    wavelengths = [item[0] for item in features]
    columns = [item[1] for item in features]

    infected_mean = infected[columns].mean(axis=0).values
    noninfected_mean = noninfected[columns].mean(axis=0).values

    plt.figure(figsize=(12, 6))

    plt.plot(
        wavelengths,
        noninfected_mean,
        label="Non-infected",
        linewidth=2
    )

    plt.plot(
        wavelengths,
        infected_mean,
        label="Infected",
        linewidth=2
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean sensor intensity")
    plt.title(
        f"{sensor} Spectral Signatures - {genotype}"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{sensor}_{genotype}_spectral_signatures.png"
    )

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_effect_sizes(result, sensor, genotype):

    result = result.sort_values("Wavelength_nm")

    plt.figure(figsize=(12, 5))

    plt.plot(
        result["Wavelength_nm"],
        result["Cohens_d"],
        linewidth=1.5
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Cohen's d")
    plt.title(
        f"Disease Effect Size by Wavelength - "
        f"{sensor}, {genotype}"
    )

    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{sensor}_{genotype}_effect_sizes.png"
    )

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved plot: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("TERRASPECTRA - DISEASE SPECTRAL SIGNATURE ANALYSIS")
    print("=" * 75)

    df = pd.read_csv(DATA_PATH)

    required = [
        "Genotype",
        "Treatment",
        "Disease_Label",
        "Pot",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    if df[required].isna().any().any():
        raise ValueError("Missing metadata values detected.")

    all_results = []

    for sensor in SENSORS:

        for genotype in GENOTYPES:

            result, features, infected, noninfected = (
                analyze_sensor_genotype(
                    df,
                    sensor,
                    genotype
                )
            )

            all_results.append(result)

            plot_spectral_curves(
                sensor,
                genotype,
                features,
                infected,
                noninfected
            )

            plot_effect_sizes(
                result,
                sensor,
                genotype
            )

            top_features = (
                result.sort_values(
                    "Absolute_Cohens_d",
                    ascending=False
                )
                .head(15)
            )

            print("\nTop 15 wavelength associations:")
            print(
                top_features[
                    [
                        "Wavelength_nm",
                        "Difference",
                        "Cohens_d",
                    ]
                ].round(4).to_string(index=False)
            )

    # --------------------------------------------------------
    # SAVE ALL WAVELENGTH RESULTS
    # --------------------------------------------------------

    combined_results = pd.concat(
        all_results,
        ignore_index=True
    )

    all_path = OUTPUT_DIR / "disease_wavelength_analysis.csv"
    combined_results.to_csv(all_path, index=False)

    # --------------------------------------------------------
    # SAVE TOP 15 PER SENSOR AND GENOTYPE
    # --------------------------------------------------------

    top_results = (
        combined_results
        .sort_values(
            "Absolute_Cohens_d",
            ascending=False
        )
        .groupby(["Sensor", "Genotype"])
        .head(15)
        .reset_index(drop=True)
    )

    top_path = OUTPUT_DIR / "top_disease_wavelengths.csv"
    top_results.to_csv(top_path, index=False)

    print("\n" + "=" * 75)
    print("SPECTRAL SIGNATURE ANALYSIS COMPLETED")
    print("=" * 75)

    print(f"\nAll wavelength results: {all_path}")
    print(f"Top wavelength results: {top_path}")
    print(f"Plots saved in: {OUTPUT_DIR}")

    print("\nInterpretation caution:")
    print(
        "These are exploratory leaf-level associations. "
        "Leaves from the same pot are not independent."
    )
    print(
        "Sensor intensity is not confirmed reflectance. "
        "Do not interpret these plots as calibrated reflectance."
    )


if __name__ == "__main__":
    main()
