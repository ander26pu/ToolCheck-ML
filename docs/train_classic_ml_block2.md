# Train ML Clasico (Bloque 2)

Script principal:
- `scripts/train_classic_ml.py`

Modelos incluidos en bloque 2:
- `gradientboosting`
- `adaboost`
- `naivebayes`

## Que hace
1. Carga `hog_pca_features_split.npz` (train/val/test).
2. Prueba grilla de hiperparametros por modelo.
3. Selecciona el mejor por validacion (`f1_macro` por defecto).
4. Reentrena con `train+val`.
5. Evalua en `test`.
6. Guarda modelo, logs y plots comparables con SVM/RF.

## Ejecucion

GradientBoosting:
```powershell
python scripts/train_classic_ml.py `
  --model gradientboosting `
  --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz `
  --output-root artifacts/gradientboosting_hog_pca_v1
```

AdaBoost:
```powershell
python scripts/train_classic_ml.py `
  --model adaboost `
  --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz `
  --output-root artifacts/adaboost_hog_pca_v1
```

NaiveBayes:
```powershell
python scripts/train_classic_ml.py `
  --model naivebayes `
  --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz `
  --output-root artifacts/naivebayes_hog_pca_v1
```

## Salidas por modelo
- `artifacts/<modelo>_hog_pca_v1/models/<modelo>_best_model.joblib`
- `artifacts/<modelo>_hog_pca_v1/logs/model_selection_val.csv`
- `artifacts/<modelo>_hog_pca_v1/logs/classification_report_val.csv`
- `artifacts/<modelo>_hog_pca_v1/logs/classification_report_test.csv`
- `artifacts/<modelo>_hog_pca_v1/logs/summary.json`
- `artifacts/<modelo>_hog_pca_v1/plots/class_split_distribution.png`
- `artifacts/<modelo>_hog_pca_v1/plots/model_selection_validation.png`
- `artifacts/<modelo>_hog_pca_v1/plots/confusion_matrix_val.png`
- `artifacts/<modelo>_hog_pca_v1/plots/confusion_matrix_test.png`
- `artifacts/<modelo>_hog_pca_v1/plots/precision_by_class_val_vs_test.png`
