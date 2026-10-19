# REHAB protocol notes

This project studies evaluation choices on the public REHAB exercise arrays. It is an independent implementation, not a reproduction of the authors' complete clinical study or trained models.

## Split unit

The inspected upstream revision is [`1073d6cb147e502cbfccd86c3011e7c487cbdb70`](https://github.com/zzzzz-123/REHAB/tree/1073d6cb147e502cbfccd86c3011e7c487cbdb70). Its [windowing script](https://github.com/zzzzz-123/REHAB/blob/1073d6cb147e502cbfccd86c3011e7c487cbdb70/Rehab_training/sliding_window.py) uses 200-point windows with stride 40 and centers each window. The [dataset assembly](https://github.com/zzzzz-123/REHAB/blob/1073d6cb147e502cbfccd86c3011e7c487cbdb70/Rehab_training/data/dataset_gen.py) retains features and class labels, without record identities. [GRU](https://github.com/zzzzz-123/REHAB/blob/1073d6cb147e502cbfccd86c3011e7c487cbdb70/Rehab_training/GRU.py) and [LSTM](https://github.com/zzzzz-123/REHAB/blob/1073d6cb147e502cbfccd86c3011e7c487cbdb70/Rehab_training/LSTM.py) then randomly split windows 80:20.

Adjacent windows share 160 of 200 source positions. Random assignment can therefore place overlapping portions of one recording on both sides of an evaluation split. A record-disjoint protocol assigns each original array row before windowing and keeps all its windows together. Comparing these protocols tests sensitivity to the evaluation unit; it does not establish that one classifier is superior to the published models.

The implementation extracts each record's windows independently, then assigns all windows from the same row together. With no learned preprocessing before the split, this is equivalent to assigning the row first. The window-random comparator is seeded and class-stratified; the inspected upstream split is neither. This baseline uses statistics of the processed angles without the upstream's additional per-window centering, then fits feature standardization on training data only. It is a protocol comparison with a fixed ridge model, not a numerical replay of the GRU/LSTM pipeline.

The content-group protocol groups byte-identical numeric contents of complete paired records (880 samples × 12 channels, serialized as little-endian float64). It shuffles unique hashes globally rather than stratifying by class, because identical contents can occur under different labels. Every copy stays on one side. It rejects a seed lacking training or test support for a class, without redrawing. This removes exact paired-content sharing but not necessarily sharing of an individual six-channel block, near duplicates, or shared patients. Per-modality sharing is audited separately; all three signal-block comparisons retain the same paired-content assignments.

## Identity and channel semantics

The [paper's Data Records section](https://www.nature.com/articles/s41597-026-07802-2) describes exercise arrays by movement ID, sensor ID and row. It does not document a patient-to-row mapping. The inspected training scripts provide none. Assessment folders `001`–`222` identify assessment samples, not unique patients. Row order must not be used to invent patient or visit groups. **Record-disjoint is not patient-disjoint:** an unseen record may still come from a training-set patient.

Exercise sensor group 1 contains the pitch, yaw and roll of two limb IMUs; group 2 contains five finger-flexion angles and a wrist angle. These are angle channels, not twelve raw acceleration or angular-velocity channels. See the paper's Tables 8–10.

The two sensor files are paired by movement ID and row index, following the upstream row-order convention. No timestamps or record-level pairing manifest independently verify acquisition synchronization; matching shapes are a necessary check, not proof of synchronization.

## Tail sensitivity

The paper describes truncating or zero-padding exercise signals to 880 points, followed by filtering and centering. A constant tail in a processed array is therefore only a **candidate padding proxy**, not a recovered acquisition boundary. Centering can make padded values nonzero; stationary movement or sensor behavior can also produce flat signals. Report the proxy's rule and sensitivity separately, preserve an unmodified baseline, and do not label proxy-based removal as verified artifact correction.

## Reuse boundaries

- The [dataset record](https://doi.org/10.57760/sciencedb.37018) is registered under CC BY 4.0; retain attribution and identify derived outputs. Its license is distinct from the paper's.
- The [paper](https://www.nature.com/articles/s41597-026-07802-2) is CC BY-NC-ND 4.0. This repository generates its own figures from data rather than adapting paper figures.
- The inspected [upstream repository](https://github.com/zzzzz-123/REHAB/tree/1073d6cb147e502cbfccd86c3011e7c487cbdb70) has no license file. Its code and model weights are not vendored here; links document the comparison protocol.

These analyses concern offline signal evaluation. They do not validate clinical assessment, treatment recommendations, or patient-independent deployment.
