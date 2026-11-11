# GUÍA DE PROMPTS PARA NOTEBOOKLM (PÓSTER INFOGRÁFICO - ACA 2)
## CardioPredict AI 2.0: Deep Learning, Redes Neuronales y Streamlit
**Especialización en Inteligencia Artificial — CUN**  
**Docente:** Dr. Hamilton Rivera Flor

---

### Instrucciones de Uso:
1. Sube como **Fuente principal** en NotebookLM el archivo `documento_fuente_poster_infografia_aca2.md` (o el PDF generado a partir de este).
2. En la sección de **Presentación** o **Póster / Informe**, utiliza los siguientes prompts para generar las 10 láminas infográficas de alta resolución con diseño profesional.

---

### Prompt Maestro para Presentación / Infografía (10 Láminas):

```text
Actúa como un Diseñador Senior de Infografías Científicas y Especialista en Inteligencia Artificial Médica. Utilizando ÚNICAMENTE la fuente proporcionada sobre "CardioPredict AI 2.0", genera una presentación infográfica de exactamente 10 láminas de alto impacto visual, concisa, profesional y estructurada bajo normas académicas, con títulos directos, iconos, diagramas de flujo y viñetas ejecutivas:

LÁMINA 1: PORTADA INSTITUCIONAL Y FICIHA TÉCNICA
- Título principal: CardioPredict AI 2.0: Redes Neuronales Artificiales y Triaje Clínico en Segundos.
- Subtítulo: Especialización en Inteligencia Artificial — Corporación Unificada Nacional de Educación Superior (CUN).
- Asignatura: Aprendizaje Automático (ACA 2) | Docente Evaluador: Dr. Hamilton Rivera Flor.
- Integrantes del equipo: Marcela Avila Beltrán, Jaime Andres Cabrera Avendaño, Cristhian Javier Garzon Jimenez, Johan Smith Valderrama Sanchez.
- Enlace destacado: https://iacun.lat/aprendizajeautomatico/aca2/

LÁMINA 2: SITUACIÓN PROBLEMA Y PREGUNTA ORIENTADORA
- Contexto de salud: Las enfermedades cardiovasculares causan 17.9 millones de muertes anuales (OMS).
- Desafío clínico: Reducir a cero los falsos negativos en triaje de urgencias.
- Pregunta Orientadora oficial del docente.
- Objetivo General: Desarrollar e implementar redes neuronales profundas (Deep MLP y 1D-CNN) con gestión SQL y despliegue interactivo en Streamlit.

LÁMINA 3: PILAR TEMÁTICO Y DATASET INTERNACIONAL (4 HOSPITALES)
- Pilar 1: Salud (Detección Asistida de Riesgo Cardiovascular).
- Dataset: UCI Heart Disease Dataset (Janosi et al., 1988) — 918 pacientes.
- Auditoría de los 4 hospitales: Cleveland (EE. UU.), Hungría (Budapest), Suiza (Zúrich) y VA Long Beach (California).
- Reto de imputación y estandarización sin fuga de datos (data leakage).

LÁMINA 4: PASO 1 - GESTIÓN Y CURADURÍA EN BASE DE DATOS SQL
- Motor Relacional: SQLite 3 (base de datos cardiopredict.db).
- Estructura de tablas: raw_patients (918 registros crudos) y curated_patients (datos clínicos curados).
- Consultas analíticas SQL clave: Prevalencia por hospital (Suiza 93.5% vs Hungría 36.2%), correlación entre tipo de dolor torácico (asintomático 79.03% riesgo) y cribado de hipertensión (>160 mmHg).

LÁMINA 5: PASO 2 - PIPELINE DE TENSORES Y DATA AUGMENTATION TABULAR
- Preprocesamiento modular: ColumnTransformer con medianas para continuas y moda para categóricas.
- Vector de entrada normalizado: 15 dimensiones fisiológicas.
- Innovación en Aumento de Datos (Data Augmentation): Perturbación gaussiana controlada (+20% de muestras sintéticas, 880 pacientes de entrenamiento) para estabilizar gradientes y enriquecer la generalización.

LÁMINA 6: PASO 3 - ARQUITECTURAS DE REDES NEURONALES (PYTORCH)
- Arquitectura 1 (Deep MLP a la medida): 15 entradas -> Capa 64 (ReLU + Dropout 0.25) -> Capa 32 (ReLU + Dropout 0.20) -> Capa 16 -> Salida Logits/Sigmoide.
- Arquitectura 2 (1D-CNN Médica): Bloques Conv1D (16 y 32 filtros, kernel 3) + MaxPool1D para extraer patrones cruzados de bioseñales.
- Función de pérdida ponderada BCEWithLogitsLoss y optimizador Adam con ReduceLROnPlateau.

LÁMINA 7: PASO 4 - CURVAS DE APRENDIZAJE Y CONVERGENCIA
- Curvas de Pérdida (Loss Curves): Descenso continuo de 0.68 a 0.30 en entrenamiento y estabilización en 0.44 en validación (cero sobreajuste).
- Curvas de Exactitud y Sensibilidad (Accuracy & Recall): Evolución estable a través de 50 épocas.
- Prevención de overfitting mediante regularización Dropout sistemática.

LÁMINA 8: RESULTADOS OFICIALES Y MATRIZ DE CONFUSIÓN (PRUEBA N=184)
- Métrica Médica Rectora: Sensibilidad (Recall) del 93.14% (95 de 102 pacientes enfermos detectados correctamente).
- Exactitud global: 82.61%.
- Área bajo la curva ROC (ROC-AUC): 90.32%.
- F1-Score Ponderado: 85.59%.
- Disminución crítica de falsos negativos de 15 a solo 7 pacientes.

LÁMINA 9: EVOLUCIÓN CIENTÍFICA ACA 1 VS ACA 2 (COMPARATIVA DIRECTA)
- Tabla comparativa lado a lado:
  * Modelo: Random Forest (ACA 1) vs Deep MLP (ACA 2).
  * Recall: 85.29% vs 93.14% (+7.85% de ganancia de vidas salvadas).
  * Exactitud: 79.35% vs 82.61% (+3.26%).
  * ROC-AUC: 89.50% vs 90.32%.
  * F1-Score: 82.08% vs 85.59%.
  * Datos: Pandas básico vs SQLite 3 + Data Augmentation.

LÁMINA 10: PASO 5 - DESPLIEGUE EN STREAMLIT, CONCLUSIONES Y REFERENCIAS APA 7
- Aplicación Web en vivo en la nube: https://iacun.lat/aprendizajeautomatico/aca2/app/
- Portal Central de Auditoría: https://iacun.lat/aprendizajeautomatico/aca2/
- Inferencia en milisegundos con semáforo de triaje y consola SQL interactiva.
- Conclusiones de salud pública y reproducibilidad.
- Referencias formales APA 7 (Janosi et al., 1988; Paszke et al., 2019; OMS, 2024; Goodfellow et al., 2016).
```
