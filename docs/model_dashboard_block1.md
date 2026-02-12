# Dashboard Comparativo de Modelos (Bloque 1)

Script principal:
- `scripts/build_model_dashboard.py`

## Objetivo
Generar un panel comparativo con salida en `PNG + HTML + CSV` para:
- elegir mejor modelo global,
- entender fallos por clase,
- analizar separacion de features (PCA/t-SNE/UMAP).

## Ejecucion
```powershell
python scripts/build_model_dashboard.py `
  --features-split artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz `
  --features-all artifacts/preprocess_hog_pca_v3/features/hog_pca_features.npz `
  --artifacts-root artifacts `
  --output-root artifacts/model_dashboard_v1
```

## Salidas
### Por modelo
- `artifacts/model_dashboard_v1/models/<modelo>/logs/classification_report_test.csv`
- `artifacts/model_dashboard_v1/models/<modelo>/logs/metrics_summary.json`
- `artifacts/model_dashboard_v1/models/<modelo>/logs/top_confusions.csv`
- `artifacts/model_dashboard_v1/models/<modelo>/logs/feature_importance_top.csv` (si aplica)
- `artifacts/model_dashboard_v1/models/<modelo>/plots/confusion_matrix_counts.(png|html)`
- `artifacts/model_dashboard_v1/models/<modelo>/plots/confusion_matrix_normalized.(png|html)`
- `artifacts/model_dashboard_v1/models/<modelo>/plots/per_class_metrics.(png|html)`
- `artifacts/model_dashboard_v1/models/<modelo>/plots/feature_importance_top.(png|html)` (si aplica)

### Comparacion entre modelos
- `artifacts/model_dashboard_v1/comparison/logs/model_metrics_test.csv`
- `artifacts/model_dashboard_v1/comparison/logs/class_f1_by_model.csv`
- `artifacts/model_dashboard_v1/comparison/logs/delta_vs_baseline_class_f1.csv`
- `artifacts/model_dashboard_v1/comparison/logs/delta_vs_baseline_macro.csv`
- `artifacts/model_dashboard_v1/comparison/logs/bootstrap_ci_metrics.csv`
- `artifacts/model_dashboard_v1/comparison/logs/mcnemar_vs_baseline.csv`
- `artifacts/model_dashboard_v1/comparison/logs/pairwise_prediction_agreement.csv`
- `artifacts/model_dashboard_v1/comparison/logs/predictions_test_by_model.csv`

### Embeddings (separacion de clases)
- `artifacts/model_dashboard_v1/comparison/plots/embedding_pca_2d.(png|html)`
- `artifacts/model_dashboard_v1/comparison/plots/embedding_pca_3d.(png|html)`
- `artifacts/model_dashboard_v1/comparison/plots/embedding_tsne_2d.(png|html)`
- `artifacts/model_dashboard_v1/comparison/plots/embedding_tsne_3d.(png|html)`
- `artifacts/model_dashboard_v1/comparison/plots/embedding_umap_2d.(png|html)`
- `artifacts/model_dashboard_v1/comparison/plots/embedding_umap_3d.(png|html)`

## Notas
- Baseline para deltas: `svm` (si existe), si no, el primer modelo encontrado.
- Para acelerar embeddings puedes bajar `--max-embed-samples`.
