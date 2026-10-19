# Data attribution and reuse

This project performs secondary analysis of **REHAB: A wearable sensor–based kinematic dataset collected under standardized rehabilitation tasks from 120 stroke patients**, deposited by **Liquan Guo** in Science Data Bank, version **V1**, DOI [10.57760/sciencedb.37018](https://doi.org/10.57760/sciencedb.37018).

The public dataset record declares **CC BY 4.0** and public access. The [versioned file inventory](config/source_files.json) records the official file-list query, file identifiers, sizes and MD5 values retrieved on 2026-10-19. Downloads are checked against these values; the analysis additionally records SHA-256. The software neither reconstructs patient identities nor links these records to other personal data.

Source arrays are downloaded into ignored `data/cache/` and are not republished here. Published aggregate results and figures in `results/` are attributed to the dataset above and distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Record-level audit tables, predictions and split assignments are generated locally under ignored `results/record-level/`; they are not redistributed in this repository or its CI artifacts. The transformations are described in the study; these outputs are not author-supplied labels, clinical results or original participant data collection by this repository's maintainer.

Related paper: Lv M, Gao Y, Cheng G, Wang L, Cai M, Zhou W, Guo L, He K. *A wearable sensor–based kinematic dataset collected under standardized rehabilitation tasks from 120 post-stroke patients*. Scientific Data 13, 1136 (2026). [10.1038/s41597-026-07802-2](https://doi.org/10.1038/s41597-026-07802-2).

The paper's CC BY-NC-ND 4.0 license is distinct from the dataset license. No paper figures are copied or adapted. The paper-linked [REHAB code repository](https://github.com/zzzzz-123/REHAB/tree/1073d6cb147e502cbfccd86c3011e7c487cbdb70) had no license file at the inspected revision; no upstream source code or pretrained weights are vendored. This repository's code is independently implemented and MIT licensed.
