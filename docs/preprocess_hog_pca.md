# Preprocess + HOG + PCA

Script principal:
- `scripts/preprocess_hog_pca.py`

## Que hace
1. Lee `data/raw/<clase>/*.jpg`.
2. Calcula metricas QC (blur y brillo) y marca flags automaticos por percentiles.
3. Intenta calibrar camara con ArUco (si no es confiable, se desactiva solo).
4. Preprocesa cada imagen:
- deteccion robusta de ArUco (multi-paso)
- rectificacion por homografia usando IDs esperados
- remocion de fondo verde + mascara de marcadores
- ROI automatico
- denoise + CLAHE
- resize letterbox a `target-size`
5. Extrae HOG (OpenCV).
6. Ejecuta PCA (varianza objetivo).
7. Guarda artefactos comprimidos + plots + reportes.

## Ejecucion completa (dataset real)
```powershell
python scripts/preprocess_hog_pca.py --data-root data/raw --output-root artifacts/preprocess_hog_pca_v1
```

## Ejecucion rapida de prueba
```powershell
python scripts/preprocess_hog_pca.py --max-images 120 --calibration-max-images 120 --detect-max-dim 720 --output-root artifacts/preprocess_hog_pca_smoke
```

## Salidas principales
- `artifacts/preprocess_hog_pca_v1/preprocessed/final_gray/<clase>/*.png`
- `artifacts/preprocess_hog_pca_v1/debug_samples/`
- `artifacts/preprocess_hog_pca_v1/debug_failures/`
- `artifacts/preprocess_hog_pca_v1/features/hog_features.npz`
- `artifacts/preprocess_hog_pca_v1/features/hog_pca_features.npz`
- `artifacts/preprocess_hog_pca_v1/features/hog_scaler.joblib`
- `artifacts/preprocess_hog_pca_v1/features/hog_pca_model.joblib`
- `artifacts/preprocess_hog_pca_v1/plots/pca_scatter_pc1_pc2.png`
- `artifacts/preprocess_hog_pca_v1/plots/pca_cumulative_variance.png`
- `artifacts/preprocess_hog_pca_v1/logs/preprocess_metadata.csv`
- `artifacts/preprocess_hog_pca_v1/logs/summary.json`

## Notas
- El script guarda checkpoints completos solo para muestras por clase y para fallidas.
- La calibracion se calcula pero solo se aplica si pasa chequeos de plausibilidad.
- En esta fase no hay split train/val/test: PCA se ajusta sobre el conjunto procesado para exploracion.
