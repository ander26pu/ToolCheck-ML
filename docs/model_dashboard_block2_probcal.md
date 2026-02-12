# Dashboard Probabilidad + Calibracion (Bloque 2)

Script principal:
- `scripts/build_model_dashboard_probcal.py`

## Objetivo
Generar analisis probabilistico por modelo y comparativo entre modelos:
- calibracion de SVM (sigmoid vs isotonic) con seleccion en validation,
- reliability diagram + ECE/MCE,
- ROC y PR multiclass one-vs-rest,
- comparativos globales y por clase.

## Ejecucion
```powershell
python scripts/build_model_dashboard_probcal.py `
  --features-split artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz `
  --svm-summary artifacts/svm_hog_pca_v1/logs/summary.json `
  --artifacts-root artifacts `
  --output-root artifacts/model_dashboard_v2_probcal
```

## Salidas
### Por modelo
- `artifacts/model_dashboard_v2_probcal/models/<modelo>/logs/probability_metrics_test.csv`
- `artifacts/model_dashboard_v2_probcal/models/<modelo>/logs/reliability_bins.csv`
- `artifacts/model_dashboard_v2_probcal/models/<modelo>/logs/roc_auc_by_class.csv`
- `artifacts/model_dashboard_v2_probcal/models/<modelo>/logs/pr_auc_by_class.csv`
- `artifacts/model_dashboard_v2_probcal/models/<modelo>/plots/reliability_diagram.(png|html)`
- `artifacts/model_dashboard_v2_probcal/models/<modelo>/plots/confidence_histogram.(png|html)`
- `artifacts/model_dashboard_v2_probcal/models/<modelo>/plots/roc_ovr_multiclass.(png|html)`
- `artifacts/model_dashboard_v2_probcal/models/<modelo>/plots/pr_ovr_multiclass.(png|html)`

### Comparacion
- `artifacts/model_dashboard_v2_probcal/comparison/logs/probability_metrics_summary.csv`
- `artifacts/model_dashboard_v2_probcal/comparison/logs/delta_vs_svm_calibrated.csv`
- `artifacts/model_dashboard_v2_probcal/comparison/logs/roc_auc_by_class_by_model.csv`
- `artifacts/model_dashboard_v2_probcal/comparison/logs/pr_auc_by_class_by_model.csv`
- `artifacts/model_dashboard_v2_probcal/comparison/logs/svm_calibration_val_metrics.csv`
- `artifacts/model_dashboard_v2_probcal/comparison/logs/probcal_summary.json`

## Notas
- Se crea modelo adicional: `svm_calibrated` en
  `artifacts/model_dashboard_v2_probcal/models/svm_calibrated/model/svm_calibrated.joblib`.
- Baseline comparativo de este bloque: `svm_calibrated`.
