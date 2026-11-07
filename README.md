# Code and data: fragmentation expansions and optimal patch counts in a non-local Klausmeier model

This repository contains the code, saved data and verification scripts for the manuscript

> Q. Zhang, *Second-order fragmentation expansions and certified optimal patch counts in a non-local Klausmeier model*, submitted for publication (2026).

The manuscript studies how splitting a habitat of fixed total length into `m` equal patches with fixed outer endpoints changes the principal eigenvalue of a non-local dispersal operator, and which patch count maximises a stationary exclusion margin that couples seed dispersal with water drainage. Several results are computer-assisted: continuous spectral enclosures, the global optima at 80 and 70 patches over all integer counts, and four switching ratios. Every strict comparison can be re-checked from the saved exact data with the Python standard library.

## Contents

| Folder / file | Content |
|---|---|
| `scripts/` | Python sources. `validated_*.py` and `verify_*.py` use only the standard library; the `*_study.py` drivers also use NumPy |
| `data/` | Saved results (CSV and JSON). Certified endpoints and trial vectors are stored as exact decimal strings |
| `generated/` | LaTeX tables written from `data/` |
| `figures/` | Vector figures of the Supplementary material |
| `REPRODUCIBILITY.txt` | Full instructions, including a table of which verifier re-checks which result of the manuscript |
| `requirements-spectral.txt` | Pinned Python dependencies |
| `build_manuscript.ps1` | Optional PowerShell driver (regenerate, verify, compile); compiling needs the LaTeX sources of the manuscript, which are not part of this repository |

## Quick start

Re-check all saved certificates (standard library only, about two to three minutes):

```
python scripts/verify_certificates.py
python scripts/verify_boundary.py
python scripts/verify_sharp.py
python scripts/verify_second_order.py
python scripts/verify_signed_second.py
python scripts/verify_cubic_anchor.py
python scripts/verify_hilbert_remainder.py
python scripts/verify_predictor.py
python scripts/verify_extensions.py
python scripts/verify_coupled.py
python scripts/verify_coupled_optimum.py
python scripts/verify_transitions.py
```

Each script exits with a non-zero code if any enclosure or strict inequality fails.

Regenerate the data (Python 3.12, `pip install -r requirements-spectral.txt`):

```
python scripts/shape_spectral_examples.py
```

## Licence

The code is released under the MIT licence (`LICENSE`). The data in `data/`, `generated/` and `figures/` are released under CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

Please cite the archived version of this repository (see `CITATION.cff` and the DOI badge of the Zenodo record) together with the manuscript.
