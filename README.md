# Proyecto Final - Minería de Datos II

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow.svg)

Este repositorio contiene el desarrollo del **Proyecto Final** para la asignatura de **Minería de Datos II**. En este trabajo se aplican técnicas avanzadas de extracción de conocimiento, preprocesamiento de datos, modelado predictivo y/o descriptivo, y evaluación de resultados sobre un conjunto de datos específico.

---

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Tecnologías y Librerías](#-tecnologías-y-librerías)
- [Requisitos Previos e Instalación](#-requisitos-previos-e-instalación)
- [Metodología de Trabajo](#-metodología-de-trabajo)
- [Resultados y Conclusiones](#-resultados-y-conclusiones)
- [Autores](#-autores)

---

## 🔍 Descripción General

El objetivo de este proyecto es abordar un problema complejo mediante el uso de pipeline completo de minería de datos:
1. **Entendimiento del Negocio/Problema:** Definición de objetivos y métricas clave de éxito.
2. **Entendimiento y Exploración de Datos (EDA):** Análisis exploratorio para identificar patrones, distribuciones y anomalías.
3. **Preprocesamiento:** Limpieza, imputación de valores faltantes, ingeniería de características (feature engineering) y normalización/escalado.
4. **Modelado:** Entrenamientos de diversos algoritmos de Aprendizaje Automático (Supervisado / No Supervisado).
5. **Evaluación:** Comparación del rendimiento de modelos con métricas pertinentes (Accuracy, Precision, Recall, F1-Score, ROC-AUC, RMSE, etc.).

---

## 📁 Estructura del Repositorio

```text
Proyecto-final-mineria2/
├── data/
│   ├── raw/              # Datos originales (sin procesar)
│   └── processed/        # Datos limpios y listos para modelar
├── notebooks/            # Notebooks de Jupyter / Google Colab
│   ├── 01_eda.ipynb      # Análisis Exploratorio de Datos
│   ├── 02_preprocessing.ipynb # Limpieza y Feature Engineering
│   └── 03_modeling.ipynb # Entrenamiento y evaluación de modelos
├── src/                  # Código fuente modularizado (.py)
│   ├── utils.py          # Funciones auxiliares
│   └── data_loader.py    # Carga y procesamiento
├── models/               # Modelos entrenados exportados (.pkl, .joblib)
├── reports/              # Gráficas, reportes y presentación final
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Documentación principal
```

---

## 🛠️ Tecnologías y Librerías

El desarrollo del proyecto está basado en **Python 3.x** y utiliza principalmente las siguientes herramientas:

- **Análisis y Manipulación de Datos:** `pandas`, `numpy`
- **Visualización:** `matplotlib`, `seaborn`, `plotly`
- **Machine Learning & Minería:** `scikit-learn`, `xgboost`, `lightgbm`
- **Entorno de Trabajo:** `Jupyter Notebook` / `VS Code` / `Google Colab`

---

## 🚀 Requisitos Previos e Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/DDev-max/Proyecto-final-mineria2.git
cd Proyecto-final-mineria2
```

### 2. Crear un entorno virtual (Opcional pero recomendado)
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 📈 Metodología de Trabajo

1. **Análisis Exploratorio (EDA):** Detección de outliers, correlaciones entre variables y distribuciones principales.
2. **Transformación:** Codificación de variables categóricas (One-Hot, Label Encoding) y estandarización.
3. **Selección de Modelos:** Experimentación con modelos base (Baseline) y modelos avanzados.
4. **Ajuste de Hiperparámetros:** Optimización mediante *GridSearchCV* / *RandomizedSearchCV*.

---

## 👥 Autores

- Roberto Coto Geuvara
- Roberto Montoya Leiva
- Claudio Poveda Sánchez
- Victor Rojas Navarro
- Kendall Solano Solís
---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.
