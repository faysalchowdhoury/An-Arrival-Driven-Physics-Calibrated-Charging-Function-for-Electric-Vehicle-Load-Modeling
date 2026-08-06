# An Arrival-Driven, Physics-Calibrated Charging Function for Electric Vehicle Load Modeling

**Validation Using 143 Days of Real Charging-Network Data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-Applied%20Energy-green.svg)](#citation)

Companion code and data pipeline for the paper:

> Faysal A. Chowdhoury & Lu Sun. *An Arrival-Driven, Physics-Calibrated Charging Function for Electric Vehicle Load Modeling: Validation Using 143 Days of Real Charging-Network Data*.

---

## Overview

This repository provides a fully reproducible pipeline that:

1. Builds a **class-segmented road-load energy model** (passenger EV, transit bus EV, freight EV) on a real suburban road network (Henrietta, NY) with measured grades and traffic volumes.
2. Calibrates a **charging-physics function** against **143 days (3,756 sessions)** of real observed charging from the [Caltech Adaptive Charging Network (ACN)](https://ev.caltech.edu/).
3. Distinguishes a **conditional (explanatory) form** suitable as a PINN prior from a **causal, forecast-feasible form**.
4. Embeds the calibrated function as a prior in a **physics-informed neural network (PINN)** and compares it against ridge regression, gradient boosting, a data-only neural network, and an LSTM at both system and link granularities.

### Key Empirical Findings

| Result | Value |
|--------|-------|
| Conditional physics function (out-of-sample R²) | **0.821** |
| Forecast-feasible variant (out-of-sample R²) | **0.656** |
| Coefficient mass carried by arrivals + hour-of-day interactions | **79.3%** |
| Coefficient mass carried by direct driving-energy term | **5.2%** |
| Coefficient disparity (arrivals vs. driving energy) | **10.3×** |
| System-level PINN vs. data-only NN | Directional advantage (not statistically significant) |
| Link-level physics regularization | Monotonically degrades accuracy |

**Main conclusion:** Hourly facility charging load is arrival-process driven, not traction-energy driven. Physics priors help at the granularity at which they were calibrated.

---

## Repository Structure

```
.
├── PINN_EV_Pipeline_CLEAN_v5_MULTISEED_v4.ipynb   # Main analysis notebook
├── data/                                           # Generated modeling tables
│   ├── link_hour.csv
│   └── link_hour_with_acn_physics_function.csv
├── data_acn/                                       # Cached ACN hourly series
│   └── acn_hourly_calib.csv
├── figures/                                        # Publication figures (PNG)
├── results/                                        # Model comparison & sensitivity CSVs
├── LICENSE
└── README.md
```

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Core packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `tqdm`
- Geospatial: `osmnx`, `geopandas`, `shapely`, `py3dep`
- Optional (Colab): Google Drive mount for backup

Install core dependencies:

```bash
pip install osmnx py3dep pandas numpy geopandas shapely requests \
            scikit-learn xgboost torch matplotlib tqdm
```

---

## API Keys (Required)

The notebook fetches live data. **Do not hard-code keys.** Set them as environment variables:

```bash
# Linux / macOS
export ACN_API_TOKEN="your_acn_token"
export NREL_API_KEY="your_nrel_key"
export OPENCHARGEMAP_API_KEY="your_ocm_key"

# Windows PowerShell
$env:ACN_API_TOKEN="your_acn_token"
$env:NREL_API_KEY="your_nrel_key"
$env:OPENCHARGEMAP_API_KEY="your_ocm_key"
```

| Key | Source | Notes |
|-----|--------|-------|
| `ACN_API_TOKEN` | [ev.caltech.edu](https://ev.caltech.edu/) | Required for real charging sessions |
| `NREL_API_KEY` | [developer.nrel.gov](https://developer.nrel.gov/) | EVI-Pro Lite profiles |
| `OPENCHARGEMAP_API_KEY` | [openchargemap.org](https://openchargemap.org/) | Charger locations |

> **Security note:** Earlier versions of this notebook contained plaintext keys. Treat any previously committed keys as compromised and rotate them.

---

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/faysalchowdhoury/An-Arrival-Driven-Physics-Calibrated-Charging-Function-for-Electric-Vehicle-Load-Modeling.git
   cd An-Arrival-Driven-Physics-Calibrated-Charging-Function-for-Electric-Vehicle-Load-Modeling
   ```

2. **Set API keys** (see above).

3. **Open the notebook**
   ```bash
   jupyter notebook PINN_EV_Pipeline_CLEAN_v5_MULTISEED_v4.ipynb
   ```
   or upload it to Google Colab.

4. **Run order** (important):
   ```
   Setup → Config → ACN real-load fetch → Network & grades →
   Road-load physics → Gravity allocation → Physics-function calibration (9B) →
   Link-hour dataset → Model training (multi-seed) → Figures / tables →
   Significance test → Reproducibility checksum
   ```

The notebook is configured with `REAL_DATA_ONLY = True`. It will refuse to invent a synthetic charging-load label.

---

## Two-Tier Evaluation Design

| Tier | Purpose | Window | Cost |
|------|---------|--------|------|
| **Calibration** | Fit charging-physics function | 2019-10-10 → 2020-02-29 (~143 days) | Low |
| **Model comparison** | PINN vs. baselines at system & link level | 2020-01-17 → 2020-02-29 (44 days) | Higher |

- Temporal train/test split only (no shuffling).
- Feature normalization uses **training-period maxima exclusively**.
- Conditional vs. forecast-feasible variants are labeled honestly.

---

## Models Compared

| Model | Description |
|-------|-------------|
| Seasonal/context ridge | Linear baseline with hour/day features |
| Gradient boosting (XGBoost) | Tree ensemble |
| Data-only neural network | Feed-forward NN, λ_phys = 0 |
| Physics-informed NN (PINN) | Same architecture + physics loss |
| LSTM | 6-hour sequence baseline (link level) |

Physics-loss weight sweep: `λ_phys ∈ {0.0, 0.05, 0.10, 0.22, 0.50}`.

---

## Reproducibility

A checksum cell at the end of the notebook prints:

- Python / PyTorch versions
- Random seed (default `42`)
- Link sampling fraction (must be `1.0` for official numbers)
- Training epochs
- ACN window
- Gravity / charger-match radii
- cuDNN deterministic flags

Matching the printed checksum guarantees that results correspond to the manuscript tables.

Multi-seed link-level results are reported as mean ± SD over 5 seeds.

---

## Data Honesty Statement

- **Aggregate hourly target** = real ACN observed charging energy.
- **Link-level EV trip counts** = deterministic estimates from NYSDOT AADT, NY DMV EV registration shares, and the real ACN hourly profile (not random draws).
- **Link-level load** = gravity allocation of the real system total, parameterized by observed charger access, AADT, employment density, EV density, POI density, and the physics energy prior.
- Public observed link-level hourly EV charging loads do **not** exist for the study network. The allocation is therefore modeled; sensitivity across four weighting schemes is reported.

---

## Citation

If you use this code or the calibrated coefficients, please cite:

```bibtex
@article{chowdhoury2026arrival,
  title   = {An Arrival-Driven, Physics-Calibrated Charging Function for Electric Vehicle Load Modeling: Validation Using 143 Days of Real Charging-Network Data},
  author  = {Chowdhoury, Faysal A. and Sun, Lu},
  journal = {Applied Energy},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work was supported in part by the U.S. Department of Labor (DOL) grant 23A60HG000052.

ACN-Data is provided by the Caltech Adaptive Charging Network project. All other inputs (OSM, USGS 3DEP, NYSDOT AADT, NREL EVI-Pro Lite, Open Charge Map, U.S. Census LODES, NY DMV EV registrations) are public open data.

---

## Contact

- **Faysal A. Chowdhoury** — faysal.chowdhoury@gmail.com 
- **Lu Sun** — corresponding author (`lxsite@rit.edu`)  
  Department of Civil Engineering Technology, Environmental Management and Safety  
  Department of Computing and Information Sciences  
  Rochester Institute of Technology, Rochester, NY 14623, USA
