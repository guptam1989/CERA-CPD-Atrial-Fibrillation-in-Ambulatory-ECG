# Pre-specified protocol for the wearer-memory extension (fixed 2026-10-20, before any test evaluation)

Development used validation patients only (screen*.py, diag*.py). No test, calibration,
external or noise data were inspected for these variants.

Final variants (nested: tuned per fold-seed model on that fold's validation patients by
mean challenge score U; ties -> first grid entry), then applied unchanged to test:

1. CERA-CPD-M (standalone). Nonconformity max(+-logit - c, 0); state-conditional
   wearer memory (bag of mode m = all blocks attributed to mode m for the wearer,
   within and across the wearer's recordings in file order; observation gaps reset
   evidence but keep memory; at an alarm, blocks after the estimated change move from
   the old to the new mode's bag). W = 4 (warm-up only while the mode's bag holds <= W
   blocks), W0 = 4. Grid: c in {0,1,2,3,4} x A in A_GRID (9 values) = 45 configs.
2. CERA-CPD-GM (gated). Published gate (CUSUM proposes, conformal SR certifies,
   W = 4, symmetric) with wearer memory (c = 0). Grid: published gate grid,
   drift {1,2,4} x h {2,5,10,20,50} x A_c {3,10,30,100,1121} = 75 configs.

Ablations (same tuning protocol): margin only (published reset policy, c x A);
wearer memory only (c = 0, A); memory within recordings only (c x A); gate with
memory within recordings only (gate grid).

Addendum (fixed after the first CPSC2021 test evaluation of variants 1-2 and the
ablations, before any further test, external or noise evaluation): resolution-matched
(dyadic) calibrator g_n(p) = (K+1)^-1 sum_{k=0..K} 2^k 1{p <= 2^-k}, K = ceil(log2 n),
n = current bag size; developed on validation patients only (screen4/5).
3. CERA-CPD-MD: variant 1 with the dyadic calibrator (same 45-config grid).
4. CERA-CPD-GMD: variant 2 with the dyadic calibrator (same 75-config gate grid).
5. Certification curves (descriptive, not tuned): published gate, gate + memory and
   gate + memory + dyadic at A_c in {3,10,30,100,300,1121,11246}, proposer fixed at the
   validation-tuned CUSUM of the same model.
External cohorts (each recording its own wearer), NSTDB noise copies and the end-to-end
QRS track use the configurations selected on CPSC2021 validation patients, unchanged.

Comparators: published locked configurations (cera_tuned, cusum, smooth, bocpd,
gate_tuned), replayed with identical code. Primary comparisons: CERA-CPD-M vs
CERA-CPD (published) and vs CUSUM; CERA-CPD-GM vs CUSUM and vs the published gate;
patient-level paired sign-flip tests, Holm within each family.
