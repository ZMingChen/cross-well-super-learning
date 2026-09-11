# Recovered experiment provenance

Recovered on 2026-09-11 from the local project that produced the manuscript results.

| Migrated module | Original file | Original SHA-256 |
|---|---|---|
| `step21_pipeline.py` | `scripts/run_step21_ensemble_reaudit.py` | `ED47288FA999D9371E0FB01AAD1E4BDBFF490FD9004068C58BBB0E0E32F2014B` |
| `final_application.py` | `scripts/fit_step21_final_and_annotate.py` | `29D19EBA40A6447E2F1E4C09F6CDCF09D6F2B6A1A2774B176113B706AE61378A` |
| `failure_analysis.py` | `scripts/analyze_step21_failure_modes.py` | `CABDF0CB1F98D9B95763237F57D0F41E792D91499D8B0DE9A8E82835081AABBE` |
| `las_preprocessing.py` | `outputs/scripts/build_las_log_dataset_step1.py` | `7A77103728BF179AEDB09CD2D9710B36D1D23F1DB100A4F41B7243CA6F165094` |
| `feature_engineering.py` | `outputs/scripts/extract_electrofacies_features_step2.py` | `99D21284BB1F6E79589F8C3621C2001803884DD30C25F153EBCE52335128298B` |
| `label_intervals.py` | `outputs/scripts/calibrate_target_electrofacies_step7.py` | `0E7F205E0DC40EE0604EB5A5A42C0C80DAEF9626A87AD49AFB351F71430DD095` |

The migrated modules replace local project paths with CLI arguments and remove the field-specific naming/reporting layer. The model definitions, fitting rules, OOF optimization, calibration and output metrics remain traceable to the original Step 21 code.

The original result files record 42,033 human-labelled points from 93 wells and 165 features, with two repetitions of five grouped outer folds and three grouped inner folds. All ten outer folds completed. The selected CalibratedSuperLearner had mean Macro-F1 0.6827422444891493.

The raw-data location recorded by the old scripts was found mounted during recovery. Its location is intentionally omitted here because the directory and its contents are confidential and must not enter a public repository.
