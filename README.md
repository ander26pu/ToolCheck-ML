# ToolCheck-ML
**ToolCheck-ML** es un sistema **offline** de identificación visual de herramientas para optimizar el préstamo y devolución en laboratorios universitarios. Usa **visión por computadora clásica (HOG)** y **Machine Learning (SVM)** para reconocer herramientas desde una **webcam**, mostrar una **predicción con confianza**, pedir **confirmación humana** y registrar la operación en un **CSV** con **foto-evidencia**.

> Fase 1: clasificación *single-tool* (una herramienta por captura), sin detección (sin bounding boxes) y sin deep learning.

---

## 🎯 Objetivo
Digitalizar y acelerar el flujo de préstamo/devolución de herramientas mediante identificación visual automática, reduciendo errores manuales y mejorando la trazabilidad.

---

## ✅ Alcance (Fase 1)
Incluye:
- Captura con webcam (modo controlado con fondo blanco)
- Preprocesamiento (resize, normalización básica)
- Extracción de características **HOG**
- Clasificación con **SVM** (LinearSVC y SVM-RBF)
- Umbral de confianza (default **0.70**) + confirmación humana
- Demo de inferencia en tiempo real
- Registro local en **CSV** + foto-evidencia
- Notebook de entrenamiento y evaluación
- Modelo serializado (`.pkl` / `.joblib`)
- Documentación y scripts reproducibles

Fuera de alcance:
- Deep learning (CNN/YOLO), GPU/nube
- Detección múltiple de herramientas en una imagen
- Integración con ERP/DB institucional
- Autenticación avanzada institucional
- Operación 24/7 en producción

---

## 📦 Entregables de la Fase 1
- Dataset organizado (20 clases × 200 imágenes/clase)
- Scripts de captura y preprocesamiento
- Notebook de entrenamiento + evaluación (k-fold=5)
- Métricas: accuracy, precision/recall, matriz de confusión, tiempo de inferencia
- Modelo entrenado serializado
- Demo webcam + UI mínima
- Registro local en CSV con evidencia fotográfica
- README + manual de uso

---

## 📊 Criterios de éxito (objetivos)
- Accuracy (test): **≥ 90%**
- Precisión promedio: **≥ 90%**
- Recall por clase: **≥ 88%**
- Inferencia: **< 500 ms**
- Confirmaciones manuales: **≤ 30%** (dependiendo del dataset/protocolo)

---

## 🧠 Enfoque técnico (HOG + SVM)
**HOG (Histogram of Oriented Gradients)** extrae descriptores robustos para formas/contornos y tolera variaciones moderadas de iluminación.  
**SVM (Support Vector Machine)** clasifica las herramientas usando esos descriptores.

Modelos evaluados:
- **LinearSVC**: rápido y simple
- **SVM-RBF**: mayor capacidad en clases parecidas

Umbral de confianza:
- `confidence >= 0.70` → sugerencia “alta”
- `confidence < 0.70` → requiere verificación/corrección humana

---

## 🧱 Arquitectura del sistema
Pipeline:
1. Captura (webcam)
2. Preprocesamiento (resize a 128×128, escala de grises, normalización)
3. Extracción HOG
4. Clasificación SVM
5. Evaluación de confianza
6. Confirmación/corrección manual
7. Registro en CSV + evidencia

---

## 🗂️ Estructura del repositorio (propuest

---

## Preprocessing HOG + PCA
Guia de ejecucion del pipeline de preprocesamiento y extraccion de features:
- `docs/preprocess_hog_pca.md`

## Training SVM (train/val/test)
Guia de entrenamiento y evaluacion con los features HOG+PCA:
- `docs/train_svm.md`

## Training Random Forest (train/val/test)
Guia de entrenamiento y evaluacion con los features HOG+PCA:
- `docs/train_random_forest.md`

## Training ML clasico (bloque 1)
Guia para entrenar ExtraTrees, Logistic Regression y KNN:
- `docs/train_classic_ml_block1.md`

## Training ML clasico (bloque 2)
Guia para entrenar GradientBoosting, AdaBoost y NaiveBayes:
- `docs/train_classic_ml_block2.md`

## Training ML clasico (bloque 3)
Guia para entrenar XGBoost, LightGBM y CatBoost:
- `docs/train_classic_ml_block3.md`

## Dashboard comparativo (bloque 1)
Guia para generar graficos por modelo y comparativos globales (PNG + HTML + CSV):
- `docs/model_dashboard_block1.md`

## Dashboard probabilidad + calibracion (bloque 2)
Guia para calibrar SVM y comparar ROC/PR/reliability por modelo:
- `docs/model_dashboard_block2_probcal.md`
