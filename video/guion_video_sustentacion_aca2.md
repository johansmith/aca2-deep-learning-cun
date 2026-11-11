# GUION OFICIAL DE SUSTENTACIÓN EN VIDEO (3 A 5 MINUTOS)
## CARDIOPREDICT AI 2.0: REDES NEURONALES ARTIFICIALES Y DESPLIEGUE EN STREAMLIT
**Especialización en Inteligencia Artificial — Corporación Unificada Nacional de Educación Superior (CUN)**  
**Asignatura:** Aprendizaje Automático — ACA 2 | **Docente:** Dr. Hamilton Rivera Flor  
**Tiempo Total Estimado:** 4 minutos exactos (rango oficial permitido: 3 a 5 minutos).

---

### 👥 Ficha Técnica del Equipo de Investigación:
1. **Marcela Avila Beltrán**
2. **Jaime Andres Cabrera Avendaño**
3. **Cristhian Javier Garzon Jimenez**
4. **Johan Smith Valderrama Sanchez**

---

## MODALIDAD RECOMENDADA: Sustentación Conducida por Johan Smith Valderrama en Representación del Equipo

> **[Nota de Grabación]:** Esta modalidad unificada replica la fórmula que obtuvo felicitaciones en el ACA 1. Johan presenta la sustentación técnica fluida en representación de sus tres compañeros, proyectando las diapositivas de soporte y realizando la demostración en vivo de Streamlit.

---

### MINUTO 1: Portada, Problema Clínico y Gestión SQL (Paso 1)
**(0:00 - 1:00)**

* **Visual en Pantalla:** Lámina 1 (Portada con logos CUN, nombres de los 4 integrantes) $\rightarrow$ Lámina 2 (Situación Problema y Pregunta Orientadora).
* **Voz de Johan:**
  > *"Un cordial saludo, profesor Hamilton Rivera y compañeros de la Especialización en Inteligencia Artificial de la CUN. Mi nombre es Johan Smith Valderrama Sánchez y, en representación de nuestro equipo de investigación integrado por Marcela Ávila, Jaime Cabrera, Cristhian Garzón y mi persona, presento la sustentación oficial del proyecto ACA 2 de Aprendizaje Automático: **CardioPredict AI 2.0: Detección Asistida de Riesgo Cardiovascular mediante Redes Neuronales Artificiales y Despliegue en Streamlit**.*
  >
  > *Las enfermedades cardiovasculares cobran más de 17.9 millones de vidas al año. Tras el éxito del ACA 1 con modelos tradicionales, la pregunta orientadora de este corte fue: ¿cómo aprovechar arquitecturas profundas como **Deep MLP y Redes Convolucionales 1D** para extraer patrones clínicos complejos y reducir a su mínima expresión los falsos negativos médicos?*
  >
  > *Para dar respuesta rigurosa, el **Paso 1** consistió en estructurar los 918 pacientes de 4 hospitales internacionales dentro de una base de datos relacional SQL mediante **SQLite** (`cardiopredict.db`), ejecutando consultas analíticas de prevalencia y correlación clínica."*

---

### MINUTO 2: Pipeline de Datos y Data Augmentation Tabular (Paso 2)
**(1:00 - 2:00)**

* **Visual en Pantalla:** Lámina 4 (Esquema SQLite y Queries) $\rightarrow$ Lámina 5 (Pipeline de tensores y Data Augmentation).
* **Voz de Johan:**
  > *"En el **Paso 2**, diseñamos un pipeline modular con `ColumnTransformer` para estandarizar 15 dimensiones de signos vitales, aplicando imputación por medianas clínicas para evitar fugas de información (*data leakage*).*
  >
  > *Además, respondiendo a la exigencia de la guía, implementamos **Aumento de Datos (Data Augmentation) Tabular**. Mediante perturbación estocástica controlada con PyTorch, expandimos el conjunto de entrenamiento en un 20% (pasando de 734 a 880 muestras sintéticas biológicamente válidas), lo que estabilizó el descenso de gradiente y previno sesgos en la optimización."*

---

### MINUTO 3: Arquitecturas de Deep Learning en PyTorch (Paso 3)
**(2:00 - 3:00)**

* **Visual en Pantalla:** Lámina 6 (Diagrama de Capas de la Red Neuronal: Deep MLP y 1D-CNN) $\rightarrow$ Lámina 7 (Curvas de Pérdida Loss y Accuracy).
* **Voz de Johan:**
  > *"En el **Paso 3**, implementamos dos arquitecturas neuronales en PyTorch:*
  >
  > *1. **Deep MLP (Multilayer Perceptron a la medida):** Con capas jerárquicas de 64, 32 y 16 neuronas, activaciones ReLU y regularización estratégica con **Dropout al 25% y 20%** para blindar la red contra el sobreajuste.*
  > *2. **Red Convolucional 1D (1D-CNN Médica):** Que utiliza filtros de convolución y MaxPool1D para capturar interdependencias espaciales entre bioseñales como el ritmo cardíaco y la depresión del segmento ST.*
  >
  > *En el **Paso 4**, las curvas de aprendizaje muestran un descenso sostenido de la función de pérdida (`BCEWithLogitsLoss`) desde 0.68 hasta 0.30, con convergencia perfecta en validación gracias al optimizador Adam con tasa de aprendizaje adaptativa."*

---

### MINUTO 4: Resultados Médicos, Comparativa ACA 1 vs ACA 2 y Demo en Streamlit (Pasos 4 y 5)
**(3:00 - 4:00)**

* **Visual en Pantalla:** Lámina 8 (Métricas y Matriz de Confusión) $\rightarrow$ Navegador web en vivo con Streamlit en `https://iacun.lat/aprendizajeautomatico/aca2/app/`.
* **Voz de Johan:**
  > *"Al evaluar sobre el conjunto de prueba independiente de 184 pacientes, los resultados son contundentes:*
  >
  > *Nuestra Red Neuronal Deep MLP alcanzó una **Sensibilidad (Recall) del 93.14%**, detectando a 95 de los 102 pacientes enfermos reales, con una **Exactitud del 82.61%** y un **ROC-AUC del 90.32%**.*
  >
  > *Al comparar científicamente contra el ACA 1, superamos al Random Forest en todas las métricas, ganando un **+7.85% de capacidad diagnóstica** y reduciendo los falsos negativos de 15 a solo 7.*
  >
  > *Para cerrar, en el **Paso 5**, desplegamos la solución en la nube en nuestro servidor en `iacun.lat`. Como observamos en pantalla, la interfaz permite ingresar signos vitales en tiempo real, seleccionar la arquitectura neuronal, ajustar el umbral clínico y auditar directamente la base de datos SQL con un solo clic.*
  >
  > *Agradecemos su atención y los invitamos a probar la aplicación interactiva en nuestro portal. ¡Muchas gracias!"*

---

## ALTERNATIVA: Guion Dividido en 4 Turnos (1 minuto por integrante)

Si el equipo prefiere grabar con la intervención individual de cada uno:

| Minuto | Integrante | Tema Asignado | Diapositivas |
| :---: | :--- | :--- | :---: |
| **Minuto 1** | **Marcela Avila Beltrán** | Saludo oficial, presentación de los 4 integrantes, planteamiento del problema, pilar salud y gestión SQL (Paso 1). | Láminas 1, 2 y 4 |
| **Minuto 2** | **Jaime Andres Cabrera Avendaño** | Dataset UCI Heart Disease (4 hospitales), preprocesamiento con ColumnTransformer y técnica de Data Augmentation Tabular (Paso 2). | Láminas 3 y 5 |
| **Minuto 3** | **Cristhian Javier Garzon Jimenez** | Modelado en PyTorch: arquitectura Deep MLP, 1D-CNN, Dropout, optimizador AdamW y análisis de curvas de Loss/Accuracy (Pasos 3 y 4). | Láminas 6 y 7 |
| **Minuto 4** | **Johan Smith Valderrama Sanchez** | Métricas en prueba (Recall 93.14%), comparativa ACA 1 vs ACA 2, demostración en vivo de Streamlit y conclusiones institucionales (Paso 5). | Láminas 8, 9 y App Web |
