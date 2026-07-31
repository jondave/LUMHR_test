# LUMHR_test

Short Streamlit app for exploring Lincolnshire mental health need at LSOA level.

## What it does

The app loads GP registration, QOF, prescribing, GP location, and LSOA geography data, then builds an interactive map with:

- a Need Index choropleth
- Depression Prevalence
- SMI Prevalence
- Antidepressant Items Per Patient
- optional GP location markers

## Data folders

The supporting datasets live in [datasets](datasets/):

- [quality_outcomes_framework](datasets/quality_outcomes_framework/) - QOF depression and mental health extracts
- [gp_prescribing_data](datasets/gp_prescribing_data/) - antidepressant prescribing data
- [patients_registered_gp_practice](datasets/patients_registered_gp_practice/) - GP-to-LSOA registration matrices
- [gp_locations](datasets/gp_locations/) - GP practice coordinate data
- [lincolnshire_lsoa](datasets/lincolnshire_lsoa/) - LSOA boundary file
- [IMD](datasets/IMD/) - deprivation data used by the broader analysis set

## Run

From [scripts](scripts/), activate the project environment and start Streamlit:

```bash
streamlit run app.py
```

If the app cannot find the datasets directory, make sure you are running it from this repository layout.