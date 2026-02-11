# Train SVM sobre HOG+PCA

Script principal:
- `scripts/train_svm.py`

## Que hace
1. Carga `hog_pca_features_split.npz` (train/val/test ya separados).
2. Entrena una grilla de modelos SVM (`LinearSVC` y `SVC-RBF`).
3. Selecciona el mejor por metrica de validacion (`f1_macro` por default).
4. Reentrena la mejor configuracion con `train+val`.
5. Evalua en `test`.
6. Guarda modelo, reportes y plots.

## Ejecucion
```powershell
python scripts/train_svm.py `
  --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz `
  --output-root artifacts/svm_hog_pca_v1
```

Opcional, cambiar metrica de seleccion:
```powershell
python scripts/train_svm.py --selection-metric precision_macro
```

## Salidas
- `artifacts/svm_hog_pca_v1/models/svm_best_model.joblib`
- `artifacts/svm_hog_pca_v1/logs/model_selection_val.csv`
- `artifacts/svm_hog_pca_v1/logs/classification_report_val.csv`
- `artifacts/svm_hog_pca_v1/logs/classification_report_test.csv`
- `artifacts/svm_hog_pca_v1/logs/summary.json`
- `artifacts/svm_hog_pca_v1/plots/class_split_distribution.png`
- `artifacts/svm_hog_pca_v1/plots/model_selection_validation.png`
- `artifacts/svm_hog_pca_v1/plots/confusion_matrix_val.png`
- `artifacts/svm_hog_pca_v1/plots/confusion_matrix_test.png`
- `artifacts/svm_hog_pca_v1/plots/precision_by_class_val_vs_test.png`
