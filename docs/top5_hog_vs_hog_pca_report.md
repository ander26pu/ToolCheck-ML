# Reporte Comparativo Top-5: HOG vs HOG+PCA

Fecha de elaboracion: 15/02/2026

## 1) Objetivo

Comparar el rendimiento de los 5 modelos con mejor desempeno historico (`logreg`, `lightgbm`, `svm`, `rf`, `xgboost`) en dos configuraciones de features:

- `HOG+PCA` (pipeline base del proyecto).
- `HOG` puro (sin reduccion por PCA).

El objetivo fue medir:

- Calidad de clasificacion: `accuracy` y `f1_macro` en test.
- Costo computacional: tiempo de entrenamiento (`fit_sec_trainval`) bajo el mismo hardware.

## 2) Metodologia

### 2.1 Configuracion experimental

- Dataset preprocesado: `artifacts/preprocess_hog_pca_v3`
- Split: `train=1175`, `val=253`, `test=252`
- Clases: 8 herramientas
- Dimensiones:
  - HOG puro: 8100 features
  - HOG+PCA: 542 componentes (95.01% de varianza retenida)

Fuente:

- `artifacts/preprocess_hog_pca_v3/logs/summary.json`

### 2.2 Criterio de comparacion justa

Para comparar de forma controlada:

- Se usaron los mismos hiperparametros ganadores por modelo.
- Se reentreno cada modelo en `train+val` y se evaluo en `test`.
- Se midio tiempo de `fit` con el mismo protocolo en ambos casos (`HOG` y `HOG+PCA`).

Archivos de resultados:

- `artifacts/top5_hog_pca_benchmark_v1/logs/top5_hog_pca_metrics_test.csv`
- `artifacts/top5_hog_only_benchmark_v1/logs/top5_hog_only_metrics_test.csv`
- `artifacts/top5_hog_only_benchmark_v1/logs/comparison_hog_vs_hog_pca_top5.csv`

## 3) Resultados Numericos

| Modelo | F1 HOG+PCA | F1 HOG | Delta F1 (HOG - HOG+PCA) | Acc HOG+PCA | Acc HOG | Delta Acc | Fit HOG+PCA (s) | Fit HOG (s) | Ratio Tiempo HOG/HOG+PCA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | 0.8427 | 0.8581 | +0.0154 | 0.8452 | 0.8611 | +0.0159 | 0.08 | 6.05 | 72.94x |
| lightgbm | 0.8366 | 0.8582 | +0.0215 | 0.8413 | 0.8611 | +0.0198 | 44.43 | 367.57 | 8.27x |
| svm | 0.8302 | 0.8332 | +0.0030 | 0.8333 | 0.8373 | +0.0040 | 3.73 | 5133.43 | 1375.62x |
| rf | 0.8192 | 0.8540 | +0.0347 | 0.8214 | 0.8571 | +0.0357 | 3.52 | 5.31 | 1.51x |
| xgboost | 0.8090 | 0.8556 | +0.0466 | 0.8135 | 0.8571 | +0.0437 | 44.55 | 571.93 | 12.84x |

## 4) Graficos Comparativos

### 4.1 Calidad de clasificacion (F1 y Accuracy)

![Comparacion F1 y Accuracy](assets/top5_hog_vs_hog_pca/comparison_f1_accuracy.png)

Lectura tecnica:

- HOG supera a HOG+PCA en `f1_macro` y `accuracy` para los 5 modelos.
- Las mayores ganancias en F1 se observan en `xgboost` y `rf`.

Lectura simple:

- En este dataset, quitar PCA mejoro la precision final para todos los modelos probados.

### 4.2 Costo de entrenamiento (escala logaritmica)

![Comparacion tiempos](assets/top5_hog_vs_hog_pca/comparison_fit_time_log.png)

Lectura tecnica:

- El costo temporal crece fuerte en HOG puro por el salto de dimensionalidad (542 -> 8100).
- `svm` lineal es el caso extremo: pasa de segundos a ~1.43 horas.

Lectura simple:

- Sin PCA, casi todos entrenan mas lento, y en SVM el tiempo se dispara mucho.

### 4.3 Delta de mejora en metricas

![Delta de metricas](assets/top5_hog_vs_hog_pca/comparison_delta_metrics.png)

Lectura tecnica:

- Todos los deltas son positivos (HOG > HOG+PCA) en las dos metricas.
- La mejora relativa mas fuerte se concentra en arboles/boosting (`rf`, `xgboost`, `lightgbm`).

Lectura simple:

- No hubo ningun caso donde HOG+PCA ganara en precision final.

### 4.4 Ranking por F1

![Ranking F1](assets/top5_hog_vs_hog_pca/comparison_rankings_f1.png)

Lectura tecnica:

- Ranking HOG+PCA (F1): `logreg > lightgbm > svm > rf > xgboost`.
- Ranking HOG (F1): `lightgbm > logreg > xgboost > rf > svm`.

Lectura simple:

- Con HOG puro cambia el liderazgo: `lightgbm` pasa a primer lugar por una diferencia minima frente a `logreg`.

## 5) Graficos Independientes por Modelo

Estas vistas muestran, para cada modelo, `accuracy`, `f1_macro` y tiempo de entrenamiento entre ambas configuraciones.

### 5.1 Logistic Regression

![logreg](assets/top5_hog_vs_hog_pca/individual_logreg.png)

### 5.2 LightGBM

![lightgbm](assets/top5_hog_vs_hog_pca/individual_lightgbm.png)

### 5.3 SVM

![svm](assets/top5_hog_vs_hog_pca/individual_svm.png)

### 5.4 Random Forest

![rf](assets/top5_hog_vs_hog_pca/individual_rf.png)

### 5.5 XGBoost

![xgboost](assets/top5_hog_vs_hog_pca/individual_xgboost.png)

## 6) Interpretacion Tecnica de por que ocurre esto

### 6.1 Efecto de PCA en este problema

PCA comprime informacion para mejorar eficiencia y reducir ruido, pero tambien puede eliminar variaciones discriminativas finas de textura/borde que HOG captura bien. En este dataset, esa perdida de detalle parece costar mas en calidad de lo que ahorra en generalizacion.

### 6.2 Modelos de tipo arbol/boosting

`rf`, `xgboost` y `lightgbm` tienden a aprovechar relaciones no lineales y combinaciones de features originales. Al proyectar con PCA, parte de la estructura original se mezcla en componentes lineales; eso puede reducir capacidad de separacion para ciertos patrones de clase.

### 6.3 Modelos lineales

`logreg` y `svm` lineal suelen beneficiarse de reduccion de dimensionalidad para velocidad y estabilidad. Aqui, `logreg` aun mejora en calidad con HOG puro, pero con un costo temporal muy superior. `svm` mejora poco en calidad y penaliza demasiado en tiempo.

### 6.4 Relacion calidad-tiempo

- Si priorizas calidad maxima: HOG puro (`lightgbm`/`logreg`) es la mejor opcion en este corte.
- Si priorizas latencia de entrenamiento y mantenibilidad: HOG+PCA sigue siendo muy competitivo, especialmente con `logreg`.

## 7) Conclusiones Ejecutivas

1. En los 5 modelos evaluados, `HOG` puro mejora `F1 macro` y `accuracy` respecto a `HOG+PCA`.
2. La mejora de calidad viene con mayor costo de entrenamiento en todos los casos.
3. El peor trade-off tiempo/calidad fue `svm` con HOG puro (ganancia pequena y costo extremo).
4. El mejor compromiso practico observado fue:
   - `logreg + HOG+PCA` si se requiere entrenamiento muy rapido.
   - `lightgbm + HOG` si se busca la mejor precision final.

## 8) Recomendacion Operativa

Para siguiente iteracion del proyecto:

1. Mantener dos perfiles de ejecucion:
   - Perfil rapido: `logreg + HOG+PCA`.
   - Perfil maximo rendimiento: `lightgbm + HOG`.
2. Si se busca un solo modelo productivo con balance, priorizar `logreg` y ajustar threshold/calibracion segun caso de negocio.
3. Ejecutar tuning especifico para HOG puro (grilla propia), ya que este reporte uso comparacion controlada con hiperparametros fijos por modelo.

## 9) Reproducibilidad

Script para regenerar las figuras del reporte:

```bash
python scripts/build_top5_hog_vs_hog_pca_report.py
```

Salida de imagenes:

- `docs/assets/top5_hog_vs_hog_pca/`
