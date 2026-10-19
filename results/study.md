# REHAB record-aware evaluation

Analyzed 4,257 paired records and 76,626 windows across 15 usable movement classes. Published files checksum-checked: 32; quarantined files: 1.

This is an independent fixed ridge baseline, not a reproduction of the authors' GRU/LSTM scores. Record rows are not patient IDs; no comparison establishes performance on unseen patients.

| Split | Signals | Mean macro-F1 | Seed range | Mean balanced accuracy | Test windows overlapping train | Paired full-record content seen in train | Selected-modality full-record content seen in train |
|---|---|---:|---|---:|---:|---:|---:|
| window_random | paired | 0.6574 | 0.6539–0.6602 | 0.6602 | 99.98% | 100.00% | 100.00% |
| record_disjoint | paired | 0.6546 | 0.6399–0.6620 | 0.6585 | 0.00% | 22.83% | 22.83% |
| content_group_disjoint | paired | 0.6440 | 0.6404–0.6506 | 0.6551 | 0.00% | 0.00% | 0.00% |
| content_group_disjoint | limb | 0.5188 | 0.5114–0.5244 | 0.5397 | 0.00% | 0.00% | 0.12% |
| content_group_disjoint | glove | 0.3234 | 0.3117–0.3397 | 0.3728 | 0.00% | 0.00% | 0.00% |

![Split and signal comparisons](scores.svg)

Seeds vary the split, not the data acquisition. Seed ranges are descriptive, not confidence intervals. Windows overlap by 80% at adjacent starts; repeated windows are not independent participants. The window-random comparator is class-stratified and seeded, rather than a bit-for-bit replay of the upstream unseeded split.

The content-group holdout globally shuffles unique full-record signal hashes, without class stratification. All copies of a content hash, including copies assigned different labels, stay on one side. A seed that removes training or test support for any class is rejected without redraw. The protocols have different held-out examples and class supports; score differences do not isolate a causal effect of leakage. Exact-content separation also does not rule out near duplicates, shared participants or shared sessions.

Content groups are defined by the concatenated limb-and-glove record. Two distinct paired records can still have identical limb-only or glove-only contents. The single-modality comparisons retain the same paired-content split for comparability; they are not guaranteed to exclude duplicates of the selected input block. The last two table columns separately report paired-content and selected-modality sharing as fractions of test windows. Per-run audits also give record-level counts and fractions. Source-row hashes for both blocks are saved only in the locally generated record audit.

## Input audit

- Movement class 014: excluded from all paired and single-modality comparisons — at least one paired source file is quarantined.

1108 records have an all-channel constant suffix of at least 200 samples. After preprocessing, a constant suffix is only a candidate-padding indicator; it may also be stationary data. No inferred suffix is silently removed or relabeled as known padding.

Exact repeated full-record contents beyond their first occurrence: 636. Repeated-content groups: 489; groups spanning more than one movement label: 1. No repeated or flat record is removed. Per-run content-sharing statistics are retained separately from row-ID sharing; locally generated record audit and split tables map the content hashes back to every source row. These row-level hashes, assignments and predictions are not included in the public results.

## Reproduce and inspect

`rehab-audit fetch` downloads the versioned public files. `rehab-audit run` regenerates these outputs. `rehab-audit verify` checks source and result hashes; `rehab-audit check` additionally revalidates cached inputs, reruns the study, and compares every public result file (complete JSON structures with 1e-10 float tolerance; other files exactly). When a local record-level directory is present, check also requires all three private tables and compares them exactly after gzip decoding.

[Input inventory](inventory.json) · [Per-class metrics](per_class.csv) · [Full summary and confusion matrices](summary.json) · [Provenance](manifest.json)

Record audit, split assignments and held-out predictions are generated under the local `record-level/` subdirectory. They are excluded from the public manifest and publication; the public summary retains aggregate duplicate-group counts only.

The limb/glove/paired comparisons use exactly the same content-group split, usable classes and windows. Training-set-only standardization precedes fixed-alpha ridge fitting; no hyperparameter is selected on the test scores. Features are per-channel mean, standard deviation, range, RMS and mean absolute first difference; angle wrapping is not corrected. The limb/glove ablation tests information available in each signal block, not robustness to unexpected sensor loss.
