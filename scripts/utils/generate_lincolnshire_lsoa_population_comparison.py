"""Generate a comparison CSV of ONS Resident Population vs GP Registered Patients for all Lincolnshire LSOAs.

Columns included:
- LSOA_Code: 2021 LSOA Code (e.g. E01026007)
- LSOA_Name: 2021 LSOA Name (e.g. Boston 001A)
- Local_Authority_District: Local Authority District Name (e.g. Boston, East Lindsey, Lincoln, etc.)
- Resident_Population_ONS_2024: ONS Small Area Population Estimate (Mid-2024 SAPE)
- GP_Registered_Patients: NHS Digital Registered GP Patients (July 2026 release)
- Number_Difference: GP Registered Patients minus ONS Resident Population
- Percentage_Difference_Pct: ((GP Patients - ONS Pop) / ONS Pop) * 100%
- GP_Registration_Coverage_Pct: (GP Patients / ONS Pop) * 100%
"""

from pathlib import Path
import pandas as pd


def generate_comparison():
    root = Path(__file__).resolve().parent.parent.parent
    pop_csv = root / "datasets" / "population_estimates" / "lincolnshire_lsoa_population_estimates_2024.csv"

    if not pop_csv.exists():
        raise FileNotFoundError(f"Missing base population dataset at: {pop_csv}")

    df = pd.read_csv(pop_csv)

    comparison_df = pd.DataFrame(
        {
            "LSOA_Code": df["LSOA_CODE"],
            "LSOA_Name": df["LSOA_NAME"],
            "Local_Authority_District": df["LAD_NAME"],
            "Resident_Population_ONS_2024": df["ONS_Pop_Total_2024"].astype(int),
            "GP_Registered_Patients": df["GP_Registered_Patients"].astype(int),
            "Number_Difference": (df["GP_Registered_Patients"] - df["ONS_Pop_Total_2024"]).astype(int),
            "Percentage_Difference_Pct": (
                ((df["GP_Registered_Patients"] - df["ONS_Pop_Total_2024"]) / df["ONS_Pop_Total_2024"].replace(0, 1)) * 100.0
            ).round(2),
            "GP_Registration_Coverage_Pct": (
                (df["GP_Registered_Patients"] / df["ONS_Pop_Total_2024"].replace(0, 1)) * 100.0
            ).round(2),
        }
    )

    # Sort descending by Number Difference to easily see areas with highest list inflation / registration gaps
    comparison_df = comparison_df.sort_values(by="Number_Difference", ascending=False).reset_index(drop=True)

    # Export to datasets and scripts/utils
    out_datasets = root / "datasets" / "population_estimates" / "lincolnshire_lsoa_population_vs_gp_registered_comparison.csv"
    comparison_df.to_csv(out_datasets, index=False)
    print(f"Saved: {out_datasets}")

    out_utils = root / "scripts" / "utils" / "lincolnshire_lsoa_population_vs_gp_registered_comparison.csv"
    comparison_df.to_csv(out_utils, index=False)
    print(f"Saved: {out_utils}")

    print(f"\nTotal Lincolnshire 2021 LSOAs: {len(comparison_df)}")
    print(f"Total ONS Resident Population (2024): {comparison_df['Resident_Population_ONS_2024'].sum():,}")
    print(f"Total GP Registered Patients: {comparison_df['GP_Registered_Patients'].sum():,}")
    print(f"Net Difference: {comparison_df['Number_Difference'].sum():,}")
    print("\nTop 5 Areas with Highest GP List Inflation (GP Patients > ONS Resident Pop):")
    print(comparison_df.head(5)[["LSOA_Code", "LSOA_Name", "Resident_Population_ONS_2024", "GP_Registered_Patients", "Number_Difference", "Percentage_Difference_Pct"]].to_string(index=False))

    print("\nTop 5 Areas with Highest Registration Gap (ONS Resident Pop > GP Patients):")
    print(comparison_df.tail(5)[["LSOA_Code", "LSOA_Name", "Resident_Population_ONS_2024", "GP_Registered_Patients", "Number_Difference", "Percentage_Difference_Pct"]].to_string(index=False))


if __name__ == "__main__":
    generate_comparison()

