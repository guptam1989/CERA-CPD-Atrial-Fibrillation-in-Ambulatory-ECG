# CERA-CPD: Conformal Certification of Sequential Detectors for Paroxysmal Atrial Fibrillation in Ambulatory ECG

Code and manuscript for

> M. Gupta, P. Kumar, H. Verma. *Conformal Certification of Sequential Detectors for
> Paroxysmal Atrial Fibrillation Delineation and Burden Estimation in Ambulatory ECG.*
> Manuscript under review, 2026.

CERA-CPD is an online framework that delineates paroxysmal atrial fibrillation (AF)
episodes in two-lead ambulatory ECG and estimates AF burden with uncertainty intervals.
It combines

* beat-synchronous, quality-gated blocks of beats with explicit release times (causal);
* a dual-stream encoder of RR irregularity and QRST-residual atrial activity with
  quality- and rate-aware fusion and ectopy-aware supervised contrastive training;
* conformal e-values ranked against the wearer's own history and accumulated by a
  Shiryaev and Roberts e-detector, with retrospective beat-level localisation;
* a conformal certificate that gives alarms of any sequential detector (e.g. CUSUM) a
  run-length bound, and an optional wearer memory across episodes and recordings;
* patient-level split-conformal burden intervals that account for quality-rejected time.

## Repository layout

| Path | Contents |
|---|---|
| `manuscript/` | LaTeX source (`main.tex`, Elsevier `elsarticle`), tables, figures, highlights and the compiled PDF |
| `experiments/cera_exp/` | Complete pipeline that produces every number, table and figure in the manuscript (see its `README.md`) |
| `experiments/cera_exp/hpc/` | PBS job scripts used on the cluster |
| `experiments/cera_exp/tests/` | Regression and integrity tests, including a check against the official CPSC2021 scorer |
| `experiments/cera_exp/docs/` | Pre-specified protocol of the wearer-memory extension |
| `experiments/cluster_pipeline/` | Download and checksum verification of the PhysioNet databases |

## Data

All recordings are public PhysioNet databases and are **not** redistributed here.
Download them from PhysioNet (the acquisition scripts verify the official SHA-256
checksums):

* CPSC2021, 4th China Physiological Signal Challenge (training, selection, calibration)
* MIT-BIH Atrial Fibrillation Database (AFDB) and Long-Term AF Database (LTAFDB)
* MIT-BIH Normal Sinus Rhythm Database (NSRDB) and MIT-BIH Arrhythmia Database
* MIT-BIH Noise Stress Test Database (NSTDB)

Please cite the original database publications listed in the manuscript when using them.

## Reproducing the results

Requirements: Python 3.11 and the packages in `requirements.txt` (PyTorch 2.6 with a
CUDA GPU for training). Run all commands from the repository root, so that the package
`experiments.cera_exp` can be imported.

```bash
pip install -r requirements.txt
python -m unittest discover -s experiments/cera_exp/tests -v      # regression checks
python -m experiments.cera_exp.simulate --out sim.json --reps 1000  # false-alarm bound check
```

The full pipeline (data preparation, nested encoder selection, training of 25 fold and
seed models, detection, evaluation, tables and figures) is described step by step in
[`experiments/cera_exp/README.md`](experiments/cera_exp/README.md). The PBS scripts in
`hpc/` expect the environment variables `CERA_ROOT` (project directory) and `CERA_PY`
(Python interpreter).

To compile the manuscript, run `manuscript/build.cmd` (Windows, MiKTeX or TeX Live with
`pdflatex` and `bibtex`).

## Licence

The source code is released under the MIT licence (see `LICENSE`), so it can be reused
and extended freely with attribution. The manuscript text, figures and tables in
`manuscript/` are © the authors, all rights reserved, pending journal publication.

## Citation

If you use this code, please cite the article above (bibliographic details will be
updated after publication).

## Contact

Muktesh Gupta, Department of Computer Science and Engineering, Dr. B. R. Ambedkar
National Institute of Technology Jalandhar, India.
