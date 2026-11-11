# 🫀 CardioPredict AI 2.0: Sistema Inteligente de Triaje Clínico y Detección de Cardiopatía Isquémica

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57.svg)](https://www.sqlite.org/)
[![Status: Production](https://img.shields.io/badge/Status-Production%20Live-success.svg)](https://iacun.lat/aprendizajeautomatico/aca2/)

---

## 🏛️ Información Institucional y Equipo de Investigación

* **Institución:** Corporación Unificada Nacional de Educación Superior (CUN)
* **Programa:** Especialización en Inteligencia Artificial
* **Asignatura:** Aprendizaje Automático — ACA 2 (Segundo Corte)
* **Docente:** Dr. Hamilton Rivera Flor
* **Grupo de Investigación:** **Grupo 12**
* **Integrantes del Equipo:**
  * **Marcela Avila Beltrán**
  * **Jaime Andres Cabrera Avendaño**
  * **Cristhian Javier Garzon Jimenez**
  * **Johan Smith Valderrama Sanchez**

---

## 🌐 Ecosistema en Producción y Enlaces Públicos

| Recurso | Enlace de Acceso | Descripción |
|---|---|---|
| **Portal Web Principal ACA 2** | [https://iacun.lat/aprendizajeautomatico/aca2/](https://iacun.lat/aprendizajeautomatico/aca2/) | Portal central con documentación, póster, visor de código y notebook interactivo. |
| **Aplicación Web en Vivo (Streamlit)** | [https://iacun.lat/aprendizajeautomatico/aca2/app/](https://iacun.lat/aprendizajeautomatico/aca2/app/) | Plataforma interactiva de inferencia neuronal, semáforo de triaje y consola SQL. |
| **Portal Base ACA 1 (Continuidad)** | [https://iacun.lat/aprendizajeautomatico/aca1/](https://iacun.lat/aprendizajeautomatico/aca1/) | Portal base del primer corte (Machine Learning clásico). |
| **Repositorio GitHub** | [https://github.com/johansmith/aca2-deep-learning-cun](https://github.com/johansmith/aca2-deep-learning-cun) | Código fuente, modelos en PyTorch y base de datos SQLite. |

---

## 📋 Descripción del Proyecto

**CardioPredict AI 2.0** es una solución clínica integral de diagnóstico asistido y triaje cardiovascular en salas de urgencias, que aborda las limitaciones del Machine Learning tradicional mediante la integración sinérgica de:

1. **Gestión Relacional de Datos en SQL (Paso 1 de la Rúbrica):** Estructuración, normalización, indexación y consulta de 918 pacientes de 4 centros hospitalarios en la base de datos relacional SQLite `database/cardiopredict.db`.
2. **Deep Learning en PyTorch (Paso 2 de la Rúbrica):** Modelado predictivo no lineal a través de redes neuronales artificiales profundas (**CardioPredictDeepMLP** y **1D-CNN**) con optimización de función de pérdida ponderada (`BCEWithLogitsLoss`, factor $1.2$) para castigar severamente los falsos negativos.
3. **Triaje Semafórico de 3 Niveles en Producción:**
   * 🚨 **Nivel Rojo (Prioridad I | $\ge 45\%$):** Alto Riesgo Coronario. Derivación urgente a cateterismo y UCI.
   * 🟡 **Nivel Amarillo (Prioridad II | $20\% - 44\%$):** Riesgo Moderado / Zona Gris de Observación. Casos atípicos para ergometría, Holter y enzimas seriadas.
   * 🟢 **Nivel Verde (Prioridad III | $< 20\%$):** Bajo Riesgo. Control ambulatorio preventivo.

---

## 📊 Dataset Multicéntrico Consolidado

* **Fuente:** Repositorio internacional UCI Heart Disease (918 pacientes de 4 hospitales):
  * *Cleveland Clinic Foundation* (EE.UU.)
  * *Hungarian Institute of Cardiology* (Budapest)
  * *University Hospital of Zurich* (Suiza)
  * *VA Medical Center de Long Beach* (California)
* **Población:** 508 pacientes con cardiopatía confirmada (55.3%) y 410 pacientes sanos/control (44.7%).
* **Variables:** 10 predictores clínicos (edad, sexo, tipo de dolor torácico, presión en reposo, colesterol, glucemia en ayunas, ECG en reposo, frecuencia cardíaca máxima, angina inducida y depresión del segmento ST con su pendiente) transformados en 15 dimensiones tensorales normalizadas.

---

## 🧠 Arquitectura de Red Neuronal (Deep MLP Campeón)

```
Vector de Entrada (15 características normalizadas)
       │
       ▼
┌───────────────────────────────────────────────┐
│ Linear(15 -> 64) + ReLU + Dropout(p=0.25)     │
└───────────────────────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────────────┐
│ Linear(64 -> 32) + ReLU + Dropout(p=0.20)     │
└───────────────────────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────────────┐
│ Linear(32 -> 16) + ReLU                       │
└───────────────────────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────────────┐
│ Linear(16 -> 1) -> Sigmoide (Probabilidad)     │
└───────────────────────────────────────────────┘
```

---

## 📈 Rendimiento Comparativo: ACA 1 (ML Clásico) vs ACA 2 (Deep Learning)

| Métrica Clínica / Técnica | ACA 1: Random Forest | ACA 2: Deep MLP (PyTorch) [CAMPEÓN] | Impacto Clínico |
|---|---|---|---|
| **Sensibilidad / Recall (Prueba)** | 85.29% | **93.14%** (+7.85%) | **Detecta 95 de 102 pacientes enfermos reales.** |
| **Área bajo la Curva ROC (AUC)** | 89.50% | **90.32%** (+0.82%) | Discriminación probabilística superior y calibrada. |
| **F1-Score en Prueba** | 86.14% | **87.96%** (+1.82%) | Balance óptimo entre precisión y cobertura. |
| **Falsos Negativos (de 102)** | 15 enfermos no detectados | **Solo 7 no detectados** | **Reducción de más del 53% en errores críticos.** |
| **Gestión de Datos** | Archivos CSV planos | **Base de Datos SQL (SQLite)** | Integridad, indexación y auditoría en vivo. |

---

## 📂 Estructura del Repositorio

```bash
aca2-deep-learning/
├── app/
│   ├── app.py                     # Aplicación Streamlit interactiva 2.0
│   ├── favicon.png                # Logo institucional CUN
│   ├── requirements.txt           # Dependencias Python
│   └── visor_codigo.html          # Visor embebido de código para el portal
├── assets/                        # Gráficas de curvas de pérdida, ROC y matrices
├── data/
│   ├── processed/                 # Dataset curado y normalizado
│   └── raw/                       # Datasets crudos de los 4 hospitales UCI
├── database/
│   ├── cardiopredict.db           # Base de datos relacional SQLite3
│   └── database_manager.py        # Script de creación de esquemas, vistas e índices
├── models/
│   ├── cardiopredict_deep_mlp.pt  # Pesos del modelo Deep MLP en PyTorch
│   ├── cardiopredict_1d_cnn.pt    # Pesos del modelo 1D-CNN en PyTorch
│   ├── preprocessor.joblib        # Pipeline ColumnTransformer ajustado
│   └── metrics_summary.json       # Métricas de entrenamiento y prueba
├── notebooks/
│   ├── CardioPredict_AI_2_Deep_Learning_CUN.ipynb  # Notebook interactivo documentado
│   └── CardioPredict_AI_2_Deep_Learning_CUN.html   # Exportación HTML para visor web
├── poster/
│   ├── documento_fuente_poster_infografia_aca2.md  # Fuente del póster infográfico
│   └── prompts_notebooklm_aca2.md                  # Prompts maestros para NotebookLM
├── scripts/
│   └── generate_notebook.py       # Script de generación del notebook
├── src/
│   └── model_training.py          # Script modular de entrenamiento PyTorch
├── video/
│   └── guion_video_sustentacion_aca2.md  # Guion del video de sustentación
├── index.html                     # Portal web principal de ACA 2
└── README.md                      # Ficha técnica y documentación del repositorio
```

---

## 🚀 Instalación y Ejecución Local

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/johansmith/aca2-deep-learning-cun.git
   cd aca2-deep-learning-cun
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r app/requirements.txt
   ```

4. **Ejecutar la aplicación web interactiva en Streamlit:**
   ```bash
   streamlit run app/app.py
   ```

5. Abrir en el navegador: `http://localhost:8501`.

---

## 📄 Licencia y Derechos Académicos

Este proyecto ha sido desarrollado con fines exclusivamente académicos para la **Especialización en Inteligencia Artificial** de la **Corporación Unificada Nacional de Educación Superior (CUN)** por el **Grupo 12**.
