# 🎬 GUION OFICIAL DE SUSTENTACIÓN EN VIDEO (3:00 MINUTOS MÁXIMO)
## CARDIOPREDICT AI 2.0: SISTEMA INTELIGENTE DE TRIAJE CLÍNICO (DEEP LEARNING & SQL)
* **Institución:** Corporación Unificada Nacional de Educación Superior (CUN)
* **Programa:** Especialización en Inteligencia Artificial
* **Asignatura:** Aprendizaje Automático — ACA 2 | **Docente:** Dr. Hamilton Rivera Flor
* **Tiempo Total Cronometrado:** 3:00 minutos exactos (180 segundos).
* **Modalidad:** Sustentación Individual Conducida por Johan Smith Valderrama Sánchez en Representación del Grupo 12.
* **Integrantes del Grupo 12:**
  1. Marcela Avila Beltrán
  2. Jaime Andres Cabrera Avendaño
  3. Cristhian Javier Garzon Jimenez
  4. Johan Smith Valderrama Sanchez (Ponente)
* **Archivo PDF para Celular:** [Guion_Sustentacion_3min_CardioPredict_AI_2.pdf](https://iacun.lat/aprendizajeautomatico/aca2/video/Guion_Sustentacion_3min_CardioPredict_AI_2.pdf)

---

### 📱 INSTRUCCIONES PRÁCTICAS PARA LA GRABACIÓN / SUSTENTACIÓN
1. **En tu teléfono celular:** Abre el archivo PDF del guion para tenerlo a la mano como teleprompter personal.
2. **En la pantalla de tu computador:** Ten abierto el PDF del póster oficial (8 láminas) en pantalla completa (o proyéctalo).
3. **Ritmo de voz:** Habla a un ritmo seguro y pausado de aproximadamente 135 palabras por minuto.
4. **Transición:** Cambia de lámina en tu computador siguiendo la marca de tiempo indicada.

---

### ⏱️ [0:00 – 0:20] LÁMINA 1: Portada Institucional y Presentación del Equipo
* **Visual en Computador:** Proyectar Lámina 1 (CardioPredict AI 2.0, logos CUN, Dr. Hamilton Rivera Flor, Grupo 12).
* **Voz de Johan:**
> *"Un cordial saludo, profesor Hamilton y compañeros. En representación del **Grupo 12**, conformado por Marcela Avila, Jaime Cabrera, Cristhian Garzón y quien les habla, Johan Smith Valderrama, presento **CardioPredict AI 2.0**: un sistema inteligente de triaje clínico y detección de cardiopatía isquémica que integra **Deep Learning en PyTorch** y **gestión relacional en SQL** sobre datos multicéntricos."*

---

### ⏱️ [0:20 – 0:45] LÁMINA 2: El Falso Negativo como Enemigo Clínico: El Problema
* **Visual en Computador:** Proyectar Lámina 2 (Infografía de 3 columnas: Problema Real -> Factores de Riesgo -> Desafío de los Datos).
* **Voz de Johan:**
> *"Las enfermedades cardiovasculares son la primera causa de muerte mundial. En urgencias, hasta un 30% de pacientes presentan síntomas atípicos o silentes, y aquí **un falso negativo es fatal**: enviar a casa a un paciente infartado cuesta vidas.  
> Por eso, nuestro objetivo no fue solo predecir, sino construir un ecosistema con **Sensibilidad superior al 93% y cero falsos negativos tolerables**, capturando factores complejos como la depresión del segmento ST y crisis hipertensivas."*

---

### ⏱️ [0:45 – 1:10] LÁMINA 3: Unificación Multicéntrica y Gestión Relacional en SQLite
* **Visual en Computador:** Proyectar Lámina 3 (Diagrama de los 4 hospitales internacionales conectados a `cardiopredict.db`).
* **Voz de Johan:**
> *"Cumpliendo el Paso 1 de la rúbrica, abandonamos los archivos planos y unificamos **918 pacientes de 4 centros hospitalarios internacionales**: Cleveland, Zurich, Hungría y Long Beach, dentro de una base de datos relacional SQLite llamada `cardiopredict.db`.  
> Mediante tablas indexadas, vistas clínicas y un pipeline con `StandardScaler` y `One-Hot Encoding`, transformamos estas variables en tensores normalizados de 15 dimensiones listos para aprendizaje profundo."*

---

### ⏱️ [1:10 – 1:35] LÁMINA 4: Diagrama Integral de Solución: El Ecosistema
* **Visual en Computador:** Proyectar Lámina 4 (Flujo en 3 fases continuas: Ingesta SQL -> Núcleo PyTorch -> Triaje Web).
* **Voz de Johan:**
> *"Nuestra solución integra un flujo continuo en tres fases:  
> Primero, la **Ingesta y Curación SQL** con imputación clínica.  
> Segundo, el **Núcleo de Deep Learning en PyTorch**, donde entrenamos dos arquitecturas: una Red Convolucional 1D para bioseñales y nuestro modelo campeón, una Red Neuronal Densa Profunda.  
> Y tercero, el **Despliegue Web en Streamlit** bajo un servidor Linux en la nube en `iacun.lat`."*

---

### ⏱️ [1:35 – 2:00] LÁMINA 5: Topología de Red Profunda y Penalización Matemática
* **Visual en Computador:** Proyectar Lámina 5 (Grafo de la red 15-64-32-16 y cuadro rojo de penalización matemática asimétrica).
* **Voz de Johan:**
> *"La arquitectura de **CardioPredictDeepMLP** cuenta con tres capas ocultas de 64, 32 y 16 neuronas con activación ReLU y regularización Dropout para evitar sobreajuste.  
> Pero el verdadero diferenciador clínico es su matemática: usamos la función `BCEWithLogitsLoss` con una **penalización asimétrica de factor 1.2** y un **umbral calibrado en 0.45**. Esto castiga severamente el error de clasificar a un paciente enfermo como sano, priorizando la sensibilidad médica."*

---

### ⏱️ [2:00 – 2:20] LÁMINA 6: Salto Cuantitativo: Machine Learning vs. Deep Learning
* **Visual en Computador:** Proyectar Lámina 6 (Tabla comparativa: Recall 85% -> 93.14%, AUC 90.32%).
* **Voz de Johan:**
> *"Los resultados demuestran la superioridad del Deep Learning frente al corte anterior:  
> Elevamos la **Sensibilidad del 85.29% al 93.14%**, alcanzando un **Área ROC del 90.32%**.  
> En la práctica clínica real, detectamos a **95 de cada 102 pacientes con infarto**, reduciendo los falsos negativos de 15 a solo 7 casos; es decir, **más del 53% de reducción en errores letales** frente al ACA 1."*

---

### ⏱️ [2:20 – 2:40] LÁMINA 7: Inferencias en Tiempo Real: El Semáforo Clínico de Triaje
* **Visual en Computador:** Proyectar Lámina 7 (Dashboard Streamlit y semáforo tricolor Rojo/Amarillo/Verde).
* **Voz de Johan:**
> *"En producción, nuestra interfaz implementa un **Semáforo Clínico de 3 Niveles**:  
> **Rojo ($\ge 45\%$):** Prioridad I de alto riesgo para cateterismo urgente.  
> **Ámbar ($20\% - 44\%$):** Una zona gris de observación preventiva para casos ambiguos con dolor atípico que no deben darse de alta.  
> Y **Verde ($< 20\%$):** Bajo riesgo ambulatorio. Además, incluye presets de carga en un clic y auditoría SQL en vivo."*

---

### ⏱️ [2:40 – 3:00] LÁMINA 8: Conclusiones, Ética Algorítmica y Despliegue en Producción
* **Visual en Computador:** Proyectar Lámina 8 (Conclusiones, código QR y enlaces oficiales).
* **Voz de Johan:**
> *"En conclusión, CardioPredict AI 2.0 trasciende el experimento de laboratorio: combina ética algorítmica, rigor en bases de datos relacionales y Deep Learning en un servidor operativo 24/7.  
> Los invitamos a escanear el código QR o acceder a `https://iacun.lat/aprendizajeautomatico/aca2/` y consultar nuestro repositorio público en GitHub.  
> Muchas gracias por su atención."*
