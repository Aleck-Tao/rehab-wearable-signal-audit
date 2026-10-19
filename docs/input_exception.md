# Published-file exception: movement 014, sensor group 1

The 2026-10-19 acquisition checked all 32 published exercise arrays against their ScienceDB byte counts and MD5 values. `014_1.npy` passes those download checks but fails numeric-array ingestion:

| Check | Observation |
|---|---|
| ScienceDB file ID | `d08cd076ec256bcfb82e44e11669566c` |
| Published/downloaded bytes | 26,975,550 |
| SHA-256 | `def65fc09291bfd08d5dd5ac2b87fee7684b91288c29b71d49b9ef665f23f2c0` |
| First six bytes | `0a ef bf bd 4e 55` |
| Required NPY signature | `93 4e 55 4d 50 59` |

The byte prefix includes a newline and UTF-8 replacement-character encoding. This is consistent with an encoding-damaged artifact, but the cause of the damage is not established. Matching the publisher checksum distinguishes the observed published artifact from a merely incomplete local download. Replacing the header would not recover information lost elsewhere in the payload, so the analysis does not attempt a repair.

The original bytes remain in the local cache. The [inventory](../results/inventory.json) records the exception. Movement **014** is excluded from every signal-block comparison, including the glove-only model, so the models share the same 15-class evaluation universe. The valid `014_2.npy` is still inspected and hashed. This exception is not a patient exclusion rule and does not identify a participant.

If a corrected dataset version is published, update the versioned source manifest explicitly and rerun all comparisons. Do not silently substitute a new file behind the old checksum or compare a future 16-class score directly with this 15-class result.
