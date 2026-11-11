# DOCUMENTO FUENTE: PÓSTER ACADÉMICO TIPO INFOGRAFÍA (ACA 2)
## CARDIOPREDICT AI 2.0: MODELADO CON REDES NEURONALES ARTIFICIALES (DEEP MLP & 1D-CNN), GESTIÓN SQL Y DESPLIEGUE EN STREAMLIT
**Especialización en Inteligencia Artificial — Corporación Unificada Nacional de Educación Superior (CUN)**  
**Asignatura:** Aprendizaje Automático | **Docente Evaluador:** Dr. Hamilton Rivera Flor  
**Entrega Oficial:** Actividad de Construcción Aplicada 2 (ACA 2)

---

### 1. Ficha Técnica y Equipo de Investigación
* **Institución:** Corporación Unificada Nacional de Educación Superior (CUN)
* **Escuela:** Escuela de Ingeniería y Ciencias Básicas — Posgrados en Inteligencia Artificial
* **Pilar Temático:** Proyecto 1: Salud (Detección y Triaje Clínico Asistido de Cardiopatías)
* **Dataset Oficial:** *UCI Heart Disease Dataset* (Janosi, Steinbrunn, Pfisterer, & Detrano, 1988) — 918 pacientes
* **Centros Hospitalarios:** 4 centros internacionales:
  1. Fundación Clínica Cleveland (EE. UU.)
  2. Instituto Húngaro de Cardiología (Budapest)
  3. Hospital Universitario de Zúrich (Suiza)
  4. Centro Médico VA de Long Beach (California)

#### Integrantes del Equipo:
1. **Marcela Avila Beltrán** (Ingeniería de Datos y Curaduría SQL)
2. **Jaime Andres Cabrera Avendaño** (Pipeline de Preprocesamiento y Data Augmentation)
3. **Cristhian Javier Garzon Jimenez** (Arquitectura de Redes Neuronales en PyTorch)
4. **Johan Smith Valderrama Sanchez** (Evaluación Multimétrica, Despliegue Streamlit y Sustentación)

---

### 2. Situación Problema y Pregunta Orientadora

#### Situación Problema
Las enfermedades cardiovasculares representan la principal causa de muerte en el mundo (OMS, 2024), cobrando más de 17.9 millones de vidas anuales. En los servicios de urgencias y consulta externa, la estratificación rápida y confiable del riesgo coronario resulta crítica. Sin embargo, los diagnósticos tradicionales enfrentan limitaciones cuando se analizan interacciones no lineales complejas entre biomarcadores (presión arterial, perfil lipídico, alteraciones del segmento ST y angina inducida por ejercicio). En el ACA 1 se demostró la viabilidad de modelos clásicos (Random Forest con 85.29% de sensibilidad); no obstante, persiste el desafío de minimizar al máximo los **falsos negativos** (pacientes enfermos dados de alta por error) y procesar bioseñales complejas.

#### Pregunta Orientadora (Sección 1 del ACA 2)
*¿Cómo aprovechar las arquitecturas de redes neuronales artificiales profundas (Deep MLP y 1D-CNN) junto con la gestión estructurada en bases de datos relacionales SQL para extraer patrones de alta complejidad en datos clínicos de salud, optimizando el rendimiento predictivo y permitiendo una interacción práctica en tiempo real mediante una interfaz médica en Streamlit?*

#### Objetivo General
Desarrollar, entrenar y evaluar una solución avanzada de Inteligencia Artificial basada en Redes Neuronales Artificiales (Multilayer Perceptron profundo y Red Convolucional 1D) implementada en PyTorch, sustentada en la gestión de datos relacional con SQLite (`cardiopredict.db`), aumento de datos (Data Augmentation) y desplegada en producción en una aplicación web interactiva con Streamlit en la nube.

---

### 3. Ruta Metodológica en 5 Pasos (Flujo de Aprendizaje Oficial)

```
[Paso 1: Exploración SQL] -> [Paso 2: Pipeline & Augmentation] -> [Paso 3: Deep Learning] -> [Paso 4: Evaluación Cuantitativa] -> [Paso 5: Despliegue Streamlit]
  SQLite (cardiopredict.db)      StandardScaler + OHE Tensores       Deep MLP (64-32-16)            Loss/Accuracy Curves            Inferencia en Producción
  Queries de prevalencia          Data Augmentation (+20%)           1D-CNN (Conv1D + MaxPool)      Recall: 93.14% | ROC: 90.32%    https://iacun.lat/aca2/app/
```

#### PASO 1: Exploración y Gestión de Datos en Base de Datos Relacional SQL
* **Motor Relacional:** SQLite 3 (`database/cardiopredict.db`).
* **Tablas Creadas:**
  - `raw_patients`: Ingestión de los 918 pacientes originales con sus variables hemodinámicas y paraclínicas.
  - `curated_patients`: Tabla curada con descripciones clínicas estandarizadas (tipos de dolor torácico, electrocardiograma, pendientes ST) y cálculo de índices de severidad.
* **Consultas Analíticas SQL Clave:**
  - Auditoría de prevalencia por centro: Suiza (93.5% prevalencia), VA Long Beach (75.4%), Cleveland (45.9%), Hungría (36.2%).
  - Correlación SQL entre dolor torácico y riesgo: El dolor asintomático (Tipo 4) concentra el 79.03% de casos patológicos, mientras que la angina atípica solo el 13.87%.
  - Consultas de cribado para pacientes con crisis hipertensiva (`resting_bp >= 160 mmHg`).

#### PASO 2: Preparación, Pipeline y Data Augmentation
* **Tratamiento de Valores Faltantes:** Imputación robusta por mediana para variables fisiológicas continuas (`trestbps`, `chol`, `thalach`, `oldpeak`) y moda para categóricas (`restecg`, `exang`, `slope`), previniendo fugas de información (*data leakage*).
* **Pipeline de Transformación:** `ColumnTransformer` scikit-learn con `StandardScaler` y `OneHotEncoder(drop='first')`, resultando en un espacio de entrada vectorial de 15 dimensiones normalizadas.
* **Aumento de Datos Tabular (Data Augmentation):**
  - Para enriquecer la capacidad de generalización de la red neuronal y simular variabilidad biológica real, se aplicó perturbación gaussiana clínica controlada ($\sigma = 0.03$) sobre el espacio continuo del conjunto de entrenamiento.
  - Se incrementó el volumen de entrenamiento en un +20% (de 734 a 880 muestras), logrando mayor estabilidad en los gradientes.

#### PASO 3: Modelado con Redes Neuronales Artificiales (PyTorch)
Se diseñaron e implementaron dos arquitecturas de Deep Learning a la medida:
1. **Red Neuronal Artificial Profunda (Deep MLP):**
   - Entrada: 15 neuronas.
   - Capa Oculta 1: 64 neuronas con activación ReLU y regularización con **Dropout (0.25)**.
   - Capa Oculta 2: 32 neuronas con activación ReLU y **Dropout (0.20)**.
   - Capa Oculta 3: 16 neuronas con activación ReLU.
   - Capa de Salida: 1 neurona (Logits lineales procesados mediante función Sigmoide).
2. **Red Neuronal Convolucional Unidimensional (1D-CNN Médica):**
   - Entrada: Tensor 1D $(Batch, 1, 15)$.
   - Bloque Convolucional 1: `Conv1D` (16 filtros, kernel 3, padding 1) + ReLU + `MaxPool1D(2)`.
   - Bloque Convolucional 2: `Conv1D` (32 filtros, kernel 3, padding 1) + ReLU + `MaxPool1D(2)`.
   - Capa Totalmente Conectada: Linear (96 $\rightarrow$ 32) + ReLU + Dropout (0.20) + Linear (32 $\rightarrow$ 1).
* **Estrategia de Optimización:**
  - Función de Pérdida: `BCEWithLogitsLoss`.
  - Optimizador: **Adam** ($lr = 0.003$, weight decay = $1\times 10^{-4}$).
  - Programador Dinámico: `ReduceLROnPlateau` reduciendo a la mitad el learning rate tras mesetas en validación.

#### PASO 4: Evaluación Cuantitativa y Curvas de Aprendizaje
* **Protocolo de Validación:** Evaluación sobre el conjunto de prueba independiente de 184 pacientes (estratificado 80/20, idéntico al corte del ACA 1).
* **Curvas de Pérdida (Loss Curves):** La pérdida de entrenamiento descendió de 0.68 a 0.30, y la de validación se estabilizó en 0.44-0.47, demostrando rápida convergencia y ausencia de sobreajuste.
* **Métricas Oficiales Obtenidas en Prueba (N=184):**
  - **Sensibilidad Clínica (Recall):** **93.14%** (Detectó 95 de los 102 pacientes enfermos reales).
  - **Exactitud (Accuracy):** **82.61%** (152 aciertos totales).
  - **Área bajo la Curva ROC (ROC-AUC):** **90.32%** (excelente capacidad de discriminación diagnóstica).
  - **F1-Score Ponderado:** **85.59%**.
  - **Precisión:** **79.17%**.

#### PASO 5: Despliegue en Streamlit y Ecosistema Web
* **App en Producción:** [https://iacun.lat/aprendizajeautomatico/aca2/app/](https://iacun.lat/aprendizajeautomatico/aca2/app/)
* **Portal Central:** [https://iacun.lat/aprendizajeautomatico/aca2/](https://iacun.lat/aprendizajeautomatico/aca2/)
* **Características del Despliegue:**
  - Selector de arquitectura en tiempo real (Deep MLP vs. 1D-CNN).
  - Calibración interactiva de umbral de decisión (default 0.45).
  - Pestaña de auditoría SQL en vivo que permite ejecutar consultas sobre `cardiopredict.db`.
  - Inferencia instantánea (< 25 milisegundos) con semáforo de triaje y recomendaciones AHA/ACC.

---

### 4. Cuadro Comparativo Oficial: ACA 1 vs. ACA 2

| Criterio Metodológico | ACA 1: Machine Learning Clásico | ACA 2: Deep Learning & Redes Neuronales | Ganancia y Justificación Técnica |
| :--- | :---: | :---: | :--- |
| **Modelo Campeón** | Random Forest Classifier | **Red Neuronal Profunda (Deep MLP)** | Capacidad de aproximación universal no lineal. |
| **Sensibilidad (Recall)** | 85.29% (87/102 detectados) | **93.14% (95/102 detectados)** | **+7.85%** (Reduce falsos negativos de 15 a solo 7). |
| **Exactitud (Accuracy)** | 79.35% | **82.61%** | **+3.26%** de mejora global en diagnóstico. |
| **ROC-AUC** | 89.50% | **90.32%** | **+0.82%** mayor área de separación clínica. |
| **F1-Score** | 82.08% | **85.59%** | **+3.51%** mayor equilibrio precisión-sensibilidad. |
| **Gestión de Datos** | Carga directa en Pandas | **Base de Datos Relacional SQLite 3** | Cumplimiento pleno del estándar SQL corporativo. |
| **Aumento de Datos** | Ninguno (datos originales) | **Data Augmentation Tabular (+20%)** | Perturbación estocástica controlada con PyTorch. |

---

### 5. Conclusiones y Discusión Médica
1. **Impacto en Salud Pública:** En medicina de urgencias, la métrica reina es el **Recall (Sensibilidad)**. Al pasar del 85.29% del ACA 1 al **93.14%** en el ACA 2, la Red Neuronal logró salvar potencialmente a 8 pacientes adicionales que en el modelo tradicional habrían sido clasificados erróneamente como sanos.
2. **Generalización Demostrada:** Las curvas de Loss y Accuracy evidencian que el uso de Dropout y optimizador AdamW con decaimiento de peso previene eficazmente el sobreajuste incluso en datasets clínicos tabulares.
3. **Reproducibilidad y Accesibilidad:** La integración entre SQLite, el Notebook modular en Google Colab y la interfaz interactiva en Streamlit garantiza una experiencia de usuario fluida tanto para el evaluador académico como para el personal médico.

---

### 6. Referencias Bibliográficas (Normas APA 7)
* Janosi, A., Steinbrunn, R., Pfisterer, M., & Detrano, R. (1988). *Heart Disease Data Set*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease
* Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32, 8024-8035.
* World Health Organization. (2024). *Cardiovascular diseases (CVDs): Fact sheet*. WHO Guidelines Approved by the Guidelines Review Committee. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
* Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. http://www.deeplearningbook.org/
