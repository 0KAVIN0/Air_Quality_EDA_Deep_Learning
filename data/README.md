# Dataset Documentation: UCI Air Quality Dataset

This folder houses the official **Air Quality Dataset** from the UC Irvine Machine Learning Repository.

## Official Source Information
- **Repository Page:** [https://archive.ics.uci.edu/dataset/360/air%20quality](https://archive.ics.uci.edu/dataset/360/air%20quality)
- **Direct Archive URL:** [https://archive.ics.uci.edu/static/public/360/air+quality.zip](https://archive.ics.uci.edu/static/public/360/air+quality.zip)
- **Digital Object Identifier (DOI):** [10.24432/C59K5F](https://doi.org/10.24432/C59K5F)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## Automated Fetching
The project includes automated download logic. If `AirQualityUCI.csv` is not present in this folder, running `python -m src.data_loader` or running the training pipeline will automatically fetch and extract the official archive into this directory.

## Manual Download Instructions
If working in an offline environment without outbound internet access:
1. Navigate to the official UCI page: `https://archive.ics.uci.edu/dataset/360/air%20quality`
2. Download `air+quality.zip` (approximately 1.5 MB).
3. Unzip the downloaded file and place `AirQualityUCI.csv` and/or `AirQualityUCI.xlsx` directly into this `data/` directory.

## File Specifications
- **File Name:** `AirQualityUCI.csv`
- **File Size:** ~1.4 MB
- **Delimiter:** Semicolon (`;`)
- **Decimal Format:** European Comma (`,`)
- **Total Valid Observations:** 9,357 hourly records (plus 114 blank trailing lines resulting in 9,471 raw lines).
- **Missing Value Sentinel:** `-200.0`
- **Target Variable:** `CO(GT)` (Ground-Truth Carbon Monoxide concentration, mg/m³).
