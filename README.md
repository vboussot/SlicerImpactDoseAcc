# 🧠 Slicer IMPACT-DoseAcc

<img src="ImpactDoseAcc.png" alt="IMPACT reg Logo" width="150" align="right">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vboussot/IMPACTDoseacc/blob/main/LICENSE)


## 🎥 Demonstration Video

In coming

## Overview

IMPACT-DoseAcc is a 3D Slicer extension for dose accumulation and uncertainty quantification in radiotherapy. It provides utilities for:

- Accumulating dose over multiple fractions
- Estimating voxel-wise uncertainty (standard deviation / min / max)
- Computing clinical metrics (MAE, gamma, DVH) with optional uncertainty-aware outputs
- Visualizing DVHs and charts and exporting results for further analysis

## Figures

Some example visualizations are available in the `docs/` folder:

<p align="center">
  <img src="docs/dose_accumulation.png" alt="Dose accumulation" width="45%" />
  <img src="docs/delivered_dose.png" alt="Delivered dose" width="45%" />
</p>

<p align="center">
  <img src="docs/DVH.png" alt="DVH example" width="45%" />
  <img src="docs/QA.png" alt="QA example" width="45%" />
</p>

> Note: If you want the `ImpactDoseacc.png` logo to appear next to the title, add the file to the repository root (or update the `src` to point to an image inside `docs/`).

## Installation

1. Clone the repository:

```bash
git clone https://github.com/vboussot/SlicerImpactDoseAcc.git
```

2. Install the extension in Slicer:
   - Open the Extensions Manager → "Install from file" → select the extension, or follow the build instructions where applicable.
   - Alternatively, use the Slicer extension manager or local extension directory.

3. Optional dependencies (required for specific features):
   - `pymedphys` (gamma computation), `numpy`, `matplotlib`. Note: `vtk` and many imaging dependencies are provided by Slicer.

## Quick Start

- Phase 1 — Prescription: prepare reference volumes and parameters.
- Phase 2 — Accumulation: choose accumulation strategy and uncertainty options.
- Phase 3 — Metrics: compute MAE, mean dose, mean uncertainty, gamma pass rates.
- Phase 4 — DVH: generate per-segment DVHs with optional uncertainty traces if an `uncertainty_*` volume is available in the same Subject Hierarchy folder.

See the `docs/` folder for example workflows and screenshots.

## Contributing

Contributions are welcome:

- Open an issue to discuss changes or feature requests
- Submit a focused pull request with tests and a clear description

## License

This project is distributed under the **Apache License 2.0**. See the `LICENSE` file.





