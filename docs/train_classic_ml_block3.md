# Train ML Clasico (Bloque 3)

Script principal:
- `scripts/train_classic_ml.py`

Modelos incluidos en bloque 3:
- `xgboost`
- `lightgbm`
- `catboost`

## Que hace
1. Carga `hog_pca_features_split.npz` (train/val/test).
2. Prueba grilla de hiperparametros por modelo.
3. Selecciona el mejor por validacion (`f1_macro` por defecto).
4. Reentrena con `train+val`.
5. Evalua en `test`.
6. Guarda modelo, logs y plots comparables con SVM/RF.

## Requisitos
Instalar dependencias una sola vez:
```powershell
python -m pip install xgboost lightgbm catboost
```

## Ejecucion

XGBoost:
```powershell
python scripts/train_classic_ml.py `
  --model xgboost `
  --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz `
  --output-root artifacts/xgboost_hog_pca_v1
```

LightGBM:
```powershell
python scripts/train_classic_ml.py `
  --model lightgbm `
  --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz `
  --output-root artifacts/lightgbm_hog_pca_v1
```

CatBoost:
```powershell
python scripts/train_classic_ml.py `
  --model catboost `
  --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz `
  --output-root artifacts/catboost_hog_pca_v1
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
