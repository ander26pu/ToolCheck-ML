# ToolCheck-ML

ToolCheck-ML es un proyecto de vision por computadora clasica y machine learning clasico para clasificar herramientas.
No usa deep learning, YOLO ni inferencia en nube.

Objetivo principal:
estandarizar el preprocesamiento de imagenes, extraer features HOG + PCA, entrenar varios modelos clasicos y compararlos con dashboards reproducibles.

## Resumen Ejecutivo

Estado actual del dataset y preprocesamiento:

| Item | Valor |
|---|---|
| Imagenes encontradas | 1681 |
| Preprocesadas con exito | 1680 |
| Fallidas | 1 |
| Clases | 8 |
| Split (train/val/test) | 1175 / 253 / 252 |
| Dimensionalidad HOG | 8100 |
| Dimensionalidad PCA | 542 |
| Varianza retenida por PCA | 0.9501 |

Fuentes:
- `artifacts/preprocess_hog_pca_v3/logs/summary.json`
- `artifacts/preprocess_hog_pca_v3/logs/dataset_split_by_class.csv`

Top de modelos en test por macro F1:

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| logreg | 0.8452 | 0.8427 |
| lightgbm | 0.8413 | 0.8366 |
| svm | 0.8333 | 0.8302 |
| rf | 0.8214 | 0.8192 |
| xgboost | 0.8135 | 0.8090 |

Fuente:
- `artifacts/model_dashboard_v1/comparison/logs/model_metrics_test.csv`

## Lectura Rapida Para Personas No Tecnicas

Este flujo se puede entender en 4 pasos:
1. Se corrigen y limpian las fotos para que todas sean comparables.
2. Cada imagen se convierte en numeros (features HOG + PCA).
3. Se entrenan varios modelos clasicos sobre esos mismos numeros.
4. Se comparan precision y calidad de confianza para elegir el mejor modelo.

Si es la primera vez que ves estos graficos:
1. Heatmap: color mas intenso significa valor mas alto.
2. Matriz de confusion: filas = clase real, columnas = clase predicha.
3. Grafico 3D de clases: cada punto es una imagen; grupos separados suelen indicar clasificacion mas facil.

## Clases

Las 8 clases actuales son:
- `alicate_corte`
- `calibrador`
- `cutter`
- `destornillador_chico_plano`
- `llave_chica_mixta`
- `llave_regulable`
- `pelacables`
- `wincha`

## Pipeline Tecnico de Preprocesamiento (AruCo + Segmentacion + HOG + PCA)

Pipeline implementado:
1. Flags de control de calidad (blur y brillo por percentiles).
2. Deteccion ArUco robusta con fallbacks.
3. Rectificacion geometrica a un plano comun.
4. Remocion de fondo verde y mascara de marcadores.
5. Extraccion automatica de ROI.
6. Denoise + CLAHE.
7. Imagen final estandarizada en grises.
8. Extraccion HOG.
9. Split estratificado (train/val/test).
10. `StandardScaler + PCA` ajustados solo en train.

Nota sobre calibracion de camara:
- El script intenta calibracion con ArUco y solo la aplica si pasa chequeos de plausibilidad.
- En `preprocess_hog_pca_v3`, la calibracion no paso validacion (`"success": false`), por eso el flujo final uso rectificacion robusta sin `undistort`.
- Ese comportamiento es intencional para no degradar imagenes con una calibracion inestable.

## Ejemplo Visual del Preprocesamiento

Imagen cruda:

![raw](artifacts/preprocess_hog_pca_v3/debug_samples/alicate_corte/001/00_raw.jpg)

Explicacion simple:
la captura original puede traer perspectiva y fondo no util para clasificar.

Rectificacion:

![rectified](artifacts/preprocess_hog_pca_v3/debug_samples/alicate_corte/001/03_rectified.jpg)

Explicacion simple:
la mesa se alinea para que todas las herramientas queden bajo una geometria comparable.

Segmentacion del objeto:

![object_rgba](artifacts/preprocess_hog_pca_v3/debug_samples/alicate_corte/001/05_object_rgba.png)

Explicacion simple:
se elimina gran parte del fondo para concentrar las features en la herramienta.

Imagen final para HOG:

![final_gray](artifacts/preprocess_hog_pca_v3/debug_samples/alicate_corte/001/09_final_gray.png)

Explicacion simple:
esta es la entrada estandar usada para extraer patrones de bordes y forma.

## Graficos de HOG y PCA

Distribucion de norma HOG:

![hog_norm_distribution](artifacts/preprocess_hog_pca_v3/plots/hog_norm_distribution.png)

Explicacion simple:
muestra si la magnitud de features es estable entre muestras.

Varianza acumulada PCA:

![pca_cumulative_variance](artifacts/preprocess_hog_pca_v3/plots/pca_cumulative_variance.png)

Explicacion simple:
indica cuantas componentes hacen falta para conservar la mayor parte de informacion.

Scatter PCA (PC1 vs PC2):

![pca_scatter_pc1_pc2](artifacts/preprocess_hog_pca_v3/plots/pca_scatter_pc1_pc2.png)

Explicacion simple:
proyecta las muestras a 2D; separacion parcial de clases sugiere features utiles.

Distribucion de clases por split:

![split_distribution_by_class](artifacts/model_dashboard_v1/comparison/plots/split_distribution_by_class.png)

Explicacion simple:
verifica balance de clases en train, validation y test.

## Alcance de Entrenamiento (Solo ML Clasico)

Modelos entrenados:
- svm
- rf
- extratrees
- logreg
- knn
- gradientboosting
- adaboost
- naivebayes
- xgboost
- lightgbm
- catboost

Guias:
- `docs/train_svm.md`
- `docs/train_random_forest.md`
- `docs/train_classic_ml_block1.md`
- `docs/train_classic_ml_block2.md`
- `docs/train_classic_ml_block3.md`

## Dashboard Comparativo de Modelos (Bloque 1)

Guia:
- `docs/model_dashboard_block1.md`

Leaderboard de metricas globales:

![leaderboard_global_metrics](artifacts/model_dashboard_v1/comparison/plots/leaderboard_global_metrics.png)

Explicacion simple:
ranking rapido del rendimiento general por modelo.

Heatmap de F1 por clase y modelo:

![class_f1_by_model_heatmap](artifacts/model_dashboard_v1/comparison/plots/class_f1_by_model_heatmap.png)

Explicacion simple:
permite ver en que clases cada modelo es fuerte o debil.

Separacion de clases en 3D (PCA):

![embedding_pca_3d](artifacts/model_dashboard_v1/comparison/plots/embedding_pca_3d.png)

Separacion de clases en 3D (t-SNE):

![embedding_tsne_3d](artifacts/model_dashboard_v1/comparison/plots/embedding_tsne_3d.png)

Separacion de clases en 3D (UMAP):

![embedding_umap_3d](artifacts/model_dashboard_v1/comparison/plots/embedding_umap_3d.png)

Explicacion simple:
si los grupos de clases se separan mejor, en general el clasificador tiene menos confusion.

## Dashboard de Probabilidades y Calibracion (Bloque 2)

Guia:
- `docs/model_dashboard_block2_probcal.md`

Leaderboard de ECE:

![leaderboard_ece](artifacts/model_dashboard_v2_probcal/comparison/plots/leaderboard_ece.png)

Explicacion simple:
ECE mide si la confianza reportada por el modelo es realista; menor valor es mejor.

Reliability diagram (ejemplo con `logreg`):

![reliability_logreg](artifacts/model_dashboard_v2_probcal/models/logreg/plots/reliability_diagram.png)

Explicacion simple:
si la curva esta cerca de la diagonal, la probabilidad del modelo es mas confiable.

Top de calibracion (ECE mas bajo):

| Modelo | ECE |
|---|---:|
| xgboost | 0.0600 |
| logreg | 0.0618 |
| lightgbm | 0.0844 |

Fuente:
- `artifacts/model_dashboard_v2_probcal/comparison/logs/probability_metrics_summary.csv`

Calibracion de SVM:
- se comparo `sigmoid` vs `isotonic` en validation
- metodo elegido: `isotonic`
- salida:
  `artifacts/model_dashboard_v2_probcal/models/svm_calibrated/model/svm_calibrated.joblib`

## Dashboard Ejecutivo (Bloque 3)

Guia:
- `docs/model_dashboard_block3_executive.md`

Pareto F1 vs ECE:

![pareto_f1_vs_ece](artifacts/model_dashboard_v3_executive/plots/pareto_f1_vs_ece.png)

Explicacion simple:
muestra el equilibrio entre clasificar bien y no sobreconfiar en la probabilidad.

Scores por perfil:

![profile_scores](artifacts/model_dashboard_v3_executive/plots/profile_scores.png)

Explicacion simple:
cada perfil cambia la prioridad entre performance, calibracion o balance.

Recomendaciones actuales:
- campeon performance: `logreg`
- campeon calibracion pura: `xgboost`
- campeon calibracion practica: `logreg`
- campeon balanceado: `logreg`

Fuente:
- `artifacts/model_dashboard_v3_executive/logs/recommendations.json`

## Conclusiones Tecnicas

1. El preprocesamiento (rectificacion ArUco + segmentacion + gris estandar) genero descriptores HOG estables y un espacio PCA util.
2. `logreg` es la mejor opcion global en este dataset por F1 macro y comportamiento balanceado.
3. `lightgbm` queda muy cerca en rendimiento de clasificacion.
4. `xgboost` destaca en calibracion probabilistica (menor ECE).
5. La eleccion final depende de prioridad operativa.
6. Si priorizas precision global, usa `logreg`.
7. Si priorizas calidad de confianza, evalua `xgboost`.
8. Si buscas compromiso general, `logreg` sigue siendo la opcion recomendada.

## Conclusiones en Lenguaje Sencillo

1. El sistema ya reconoce bien herramientas en fotos controladas.
2. El modelo mas equilibrado hoy es `logreg`.
3. Si importa mucho la calidad del porcentaje de confianza, `xgboost` es fuerte.
4. Los dashboards muestran claramente en que clases falla cada modelo para mejorar despues.

## Comandos de Reproducibilidad

Preprocesamiento:

```powershell
python scripts/preprocess_hog_pca.py --data-root data/raw --output-root artifacts/preprocess_hog_pca_v3
```

Entrenamiento (ejemplos):

```powershell
python scripts/train_svm.py --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz --output-root artifacts/svm_hog_pca_v1
python scripts/train_random_forest.py --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz --output-root artifacts/rf_hog_pca_v1
python scripts/train_classic_ml.py --model logreg --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz --output-root artifacts/logreg_hog_pca_v1
python scripts/train_classic_ml.py --model xgboost --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz --output-root artifacts/xgboost_hog_pca_v1
python scripts/train_classic_ml.py --model lightgbm --features-npz artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz --output-root artifacts/lightgbm_hog_pca_v1
```

Dashboards:

```powershell
python scripts/build_model_dashboard.py --output-root artifacts/model_dashboard_v1
python scripts/build_model_dashboard_probcal.py --output-root artifacts/model_dashboard_v2_probcal
python scripts/build_model_dashboard_executive.py --output-root artifacts/model_dashboard_v3_executive
```

## Indice de Documentacion

- `docs/preprocess_hog_pca.md`
- `docs/train_svm.md`
- `docs/train_random_forest.md`
- `docs/train_classic_ml_block1.md`
- `docs/train_classic_ml_block2.md`
- `docs/train_classic_ml_block3.md`
- `docs/model_dashboard_block1.md`
- `docs/model_dashboard_block2_probcal.md`
- `docs/model_dashboard_block3_executive.md`
