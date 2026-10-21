# CERA-CPD experiment pipeline

This package produces every number, table and figure in the manuscript
(`manuscript/main.tex`). It was run on the NIT Jalandhar PBS cluster (one H100 MIG slice for
training, 8-core CPU jobs for preparation and evaluation). The PBS job files are in
`hpc/`; set `CERA_ROOT` (project directory with `code/`, `prep/`, `runs/`, `logs/`) and
`CERA_PY` (Python interpreter) before submitting them. Python 3.11, PyTorch 2.6, NumPy, SciPy, wfdb 4.3 and matplotlib are required.

## Pipeline

| Step | Module | Output |
|---|---|---|
| Download, checksum-verify PhysioNet releases | `experiments/cluster_pipeline/acquire_files.py` | `data/<db>/` |
| Beat-synchronous conditioning and block features | `prep.py` (`records.py`, `features.py`, `noise.py`) | `prep/<db>/*.npz`, `index.json` |
| Patient-level outer folds and pre-selected recordings | `folds.py` | `prep/folds.json` |
| Nested encoder selection (4 candidates x 5 folds, seed 0) | `train.py`, `select_cfg.py`, `hpc/train_select.pbs` | `runs/cand_*` |
| Dual-stream encoder, 5 seeds; ablations, 3 seeds; score export | `train.py`, `model.py`, `hpc/train_main.pbs` | `runs/main/fold*/<variant>_s*/` |
| Deep sequence comparators (RR-LSTM, 1-D U-Net) | `seq_baselines.py`, `hpc/train_seq.pbs` | `runs/main/fold*/{rr_lstm,unet}_s*/` |
| Detection, validation tuning, test evaluation, burden calibration, external cohorts | `run_eval.py` (`detect.py`, `rr_baselines.py`, `evaluate.py`) | `results/main/*.json.gz` |
| Operating curves and alarm rates vs threshold | `fa_sweep.py` | `results/sweep/` |
| Monte Carlo checks of the false-alarm bounds | `simulate.py` | `results/sim/sim.json` |
| Aggregation, paired statistics, tables, figures | `aggregate.py`, `tables.py`, `figures.py`, `trace_export.py` | `paper_outputs/` |
| Computational cost | `cost.py` | `cost.json` |
| Secondary: conformally gated CUSUM (`detect.gated`, variant `gate` in `run_eval.py`) | `run_eval.py`, `hpc/eval.pbs` | `results/main/fold*_gate_s*.json.gz` |
| Secondary: mechanism (ideal-encoder simulation, delay vs history) | `mechanism.py` | `mech_sim.json`, `mech_emp.json` |
| Additional manuscript tables | `supp_tables.py` | `tables/tab_s_*.tex` |

## Design and validation rules

* Patients: CPSC2021 recording names `data_<patient>_<record>` give 105 patients
  (0-53 in set I, 54-104 in set II). Five outer folds stratified by patient type;
  within each fold 51–52 training, 15–16 validation and 17 calibration patients
  remain after the 20–22 outer-test patients. The pre-selected
  recording of each patient (burden calibration and coverage) is its longest.
* Reference-peak track: database beat annotations are the R peaks (AFDB: `qrsc`
  where present, otherwise the automatic `qrs`).
* Beat quality: spurious-QRS-energy index (`features.beat_quality`); kurtosis was
  rejected because it removed most premature ventricular beats.
* Encoder configuration: per fold, the candidate with the lowest validation fused BCE.
* Detector thresholds and all comparator parameters: per fold, maximum mean CPSC2021
  challenge score on validation recordings. Nominal budgets use the explicitly
  conditional run-length and repeated-alarm bounds described with the detector.
* Decision-clipped nonconformity: `max(+-logit, 0)`; exact null validity is unchanged.
* The official CPSC2021 score is reproduced exactly (`evaluate.official_point`,
  cross-checked against the organisers' `score_2021.py` in `tests/test_official_score.py`).

## Local checks

```powershell
python -m experiments.cera_exp.tests.test_official_score DATA_DIR path\to\score_2021.py
python -m experiments.cera_exp.cost --out cost.json
python -m experiments.cera_exp.simulate --out sim.json --reps 1000
```

## Integrity corrections and locked replay

Evaluation schema 2 keeps ECG observation support distinct from reference-label
support. Unknown reference time contributes neither negative episodes nor labelled
exposure. Boundaries adjacent to unknown labels are censored. Empty ECG support
remains empty and produces an unassessable burden interval; it is not replaced by
the entire recording.

Burden predictions and intervals use ECG support and do not depend on which
reference labels happen to be available. Point-error evaluation uses labelled
time. When the full true burden is not identifiable, exact coverage is undefined;
the proportions of prediction intervals containing or intersecting the possible
true-burden range give lower and upper empirical coverage bounds.

Block availability waits for both the next R peak plus its confirmation latency
and the final beat's 0.5-second signal-quality window. Recomputing these times
for existing prepared files preserves features, quality decisions and trained
models. The correction affects alarm timing without requiring retraining.

Operating sweeps retain episode counts and finite alarm delays. F1 is computed
from pooled counts and delay from pooled observations, separately for each seed;
legacy averages of fold F1 or medians are rejected. MIT-BIH paired comparisons
use the same non-AF population as the descriptive table. The main delay metric
is overlap-based: an alarm may precede a later reference onset when a predicted
episode spans several reference episodes. Finite negative delays are clipped to
zero for the median; the by-30-second proportion includes early alarms. This is
not a fresh-onset response rate. The separate empirical mechanism analysis uses
the first subsequent alarm, through 60 seconds after the reference offset,
without requiring a one-to-one match.

The ideal-encoder mechanism simulation stores its actual replicate count, seed,
source hash and software versions. The repaired manuscript uses 300 replicates
per setting and seed 3. Gating and mechanism analyses remain explicitly secondary;
they were specified after the primary results were known.

For a correction that preserves original validation choices, use
`run_eval --locked-evaluations ORIGINAL_EVALUATIONS` with a fresh output
directory. It replays the exact saved detector settings and Markov model, then
re-evaluates calibration, test and external predictions. Source/input manifests
are recorded. Historical outputs are not overwritten.

## Revision analyses (20 October 2026)

All three analyses reuse the 25 locked encoders and the locked detector settings;
nothing is retrained or retuned on test, external or noise data.

| Analysis | Modules | Output |
|---|---|---|
| End-to-end track: causal two-lead Pan--Tompkins-type QRS detector replaces the reference beats; detector confirmation times drive block release | `qrs.py`, `prep_qrs.py`, `score_export.py`, `run_eval.py --locked-evaluations` | `prep_qrs/`, `runs_qrs/`, `results/qrs/` |
| Standard AF-detector summaries (EC57-style AF-time and episode Se/PPV/Sp, episodes >= 30 s, Bland--Altman, CCC) on the reference-peak track | `evaluate.py` (extra row fields), `locked_subset.py`, `run_eval.py --locked-evaluations` | `results/ext_ref/` |
| Secondary: conformal certification of smoothing, percentile, BOCPD and RR-LSTM proposals (only A_c tuned on validation patients) | `detect.gated_generic` and proposers, `revision_eval.py` | `results/gatex/` |
| Compact summary and LaTeX tables | `revision_aggregate.py`, `revision_tables.py` | `revision_summary.json`, `tab_ec57.tex`, `tab_qrs.tex`, `tab_gatex.tex` |

The QRS detector's band-pass starts from the steady state of the first sample,
its peak levels are re-learned after 3 s without a detection, and detections
below 0.1 mV peak-to-peak on both leads are discarded (lead-off safeguard).
Detector accuracy is computed within each record's annotated span. With the
certificate disabled (A_c -> 0, W = 0) the generic gate reproduces ungated
smoothing and BOCPD exactly. Cluster stages must run one job at a time.

## Wearer-memory extension (20 October 2026, secondary)

Developed on validation patients only after all results above were known; the
protocol was fixed before any test evaluation
(`docs/memory_extension_PROTOCOL.md`).

| Component | Code |
|---|---|
| State-conditional wearer memory (bag per mode kept across episodes and a wearer's recordings; blocks after the estimated change move between bags at an alarm), margin clipping `max(+-l - gamma, 0)`, resolution-matched (dyadic) calibrator | `detect.WearerMemory`, `detect.cera_memory`, `detect.gated_memory`, `detect.log_dyadic_calibrator` |
| Nested tuning on validation patients, test, external, noise and end-to-end evaluation; published configurations replayed for paired tests | `memory_eval.py` |
| Summary, paired tests, LaTeX table and figure | `memory_aggregate.py`, `memory_tables.py` |

With `reset=True` (bags emptied at alarms and gaps) `cera_memory` and
`gated_memory` reproduce `cera` and `gated` exactly (unit tests). CPSC2021
recordings of a patient are processed in ascending record number with one memory;
each external or noise recording is its own wearer.

```bash
python -m experiments.cera_exp.memory_eval --prep PREP/cpsc2021 --folds PREP/folds.json --runs RUNS/main \
    --locked LOCKED_REF --out RESULTS/memory --tasks "0:0,...,4:4" --external PREP/afdb,...,PREP/noise_bw_12 \
    --qrs-prep PREP_QRS/cpsc2021 --qrs-runs RUNS_QRS --qrs-external PREP_QRS/afdb,... --workers 7
python -m experiments.cera_exp.memory_aggregate --memory RESULTS/memory --folds PREP/folds.json --out memory_summary.json
python -m experiments.cera_exp.memory_tables --summary memory_summary.json --tables TABLES --figures FIGURES
```

Cluster run: job 36507 (code, logs and `results/memory`); summary `results/memory_summary_20261020.json`. The cluster
outputs reproduced the local CPSC2021 runs record by record (375 method-model pairs).

Meaningful regression checks:

```powershell
.venv/Scripts/python.exe -m unittest discover -s experiments/cera_exp/tests -v
```
