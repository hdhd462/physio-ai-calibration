# Calibration drift and low-cost recalibration of physiological AI across populations

Code for the manuscript *"Calibration drift and low-cost recalibration of physiological AI across populations: evidence from mortality prediction in NHANES and ECG-based atrial fibrillation detection"*.

## Structure

```
code/                     # all analysis scripts (flat), paths.py = central config
```

## What the code does

- **Data preparation**: `prep_ptbxl.py`, `prep_chapman.py`, `extract_features.py`
- **Model training**: `fusion_e2e.py` (FusionNet), `fusion_e2e_multi.py` (five-outcome multi-task), `fusion_e2e_unweighted.py` (loss-weight sensitivity), `fusion_stacking.py`, `fusion_feature_level.py`, `fusion_retrain_code15.py`, `routeA_train.py`, `routeB_logistic.py`, `dl_georgia.py`
- **Inference / evaluation**: `code15_all_models.py`, `eval_multi.py`, `infer_multi_external.py`, `fs_compare.py`, `calibrate_probs.py`
- **Calibration & recalibration analyses**: `crossfit_recalibration.py`, `drift_decomposition.py`, `recal_sample_size_curve.py`, `zero_data_repair.py`, `shap_routeA.py`
- **Figures & tables**: `make_fig12_new.py`, `regen_all_figures.py`, `make_fig_cal_dca_fusion.py`, `make_fig_recalibration.py`, `make_figS_recal_drift.py`, `make_figS_subgroup.py`, `make_figS_cal_dca.py`, `make_fig7_ab.py`, `make_fig7_routeB.py`, `make_fig9_10.py`, `make_table6*.py`, `make_table7_or.py`, `make_table8*.py`, `make_tableS67.py`, `make_tableS8.py`, `make_tableS11.py`, `make_tableS12.py`, `make_tableS13.py`
- **NHANES arm**: `nhanes_calibration.py`, `nhanes_cal_check.py`

## Data sources (all public)

| Dataset | Source |
|---|---|
| PTB-XL | PhysioNet — https://physionet.org/content/ptb-xl/1.0.3/ |
| Chapman-Shaoxing ECG | PhysioNet — https://physionet.org/content/ecg-arrhythmia/1.0.0/ |
| CODE-15% | PhysioNet — https://physionet.org/content/code-15/ |
| NHANES 2011–2014 + NCHS Linked Mortality | https://www.cdc.gov/nchs/nhanes/ |

## Environment & running

- Python 3.12, dependencies in `requirements.txt`.
- All filesystem roots are defined in **`code/paths.py`**; edit that one file to point at your local data folders. Raw datasets, model checkpoints (`*.pt`) and intermediate files (`*.npz`, `*.csv`) are **not** included.
- Random processes use fixed seeds in the scripts.

## Note

Curated, analysis-only subset of the working directory (manuscript-editing and paraphrase scripts excluded). License not assigned.
