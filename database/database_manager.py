"""
CardioPredict AI 2.0 - Módulo de Gestión de Base de Datos Relacional SQL
Corporación Unificada Nacional de Educación Superior (CUN)
Asignatura: Aprendizaje Automático - ACA 2
Docente: Dr. Hamilton Rivera Flor
Equipo: Marcela Avila, Jaime Cabrera, Cristhian Garzón, Johan Smith Valderrama
"""

import sqlite3
import pandas as pd
from pathlib import Path

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "cardiopredict.db"
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "heart_disease_limpio.csv"


def init_database():
    """Crea la estructura de base de datos SQLite y carga los datos de los 918 pacientes."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Tabla de pacientes crudos / originales (Fase de Ingestión SQL)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_patients (
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        age REAL,
        sex REAL,
        cp REAL,
        trestbps REAL,
        chol REAL,
        fbs REAL,
        restecg REAL,
        thalach REAL,
        exang REAL,
        oldpeak REAL,
        slope REAL,
        hospital TEXT,
        target INTEGER
    );
    """)

    # 2. Tabla de pacientes curados para analítica clínica (Fase de Curaduría SQL)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS curated_patients (
        patient_id INTEGER PRIMARY KEY,
        age INTEGER,
        sex_desc TEXT,
        chest_pain_type TEXT,
        resting_bp REAL,
        cholesterol REAL,
        fasting_bs_desc TEXT,
        rest_ecg_desc TEXT,
        max_hr REAL,
        exercise_angina_desc TEXT,
        st_depression REAL,
        st_slope_desc TEXT,
        hospital TEXT,
        has_heart_disease INTEGER,
        clinical_severity_score REAL
    );
    """)

    # Leer dataset base verificado de los 4 hospitales
    df = pd.read_csv(CLEAN_DATA_PATH)
    
    # Inserción en raw_patients si está vacía
    cursor.execute("SELECT COUNT(*) FROM raw_patients;")
    count_raw = cursor.fetchone()[0]

    if count_raw == 0:
        raw_records = []
        for _, row in df.iterrows():
            raw_records.append((
                float(row['age']),
                float(row['sex']),
                float(row['cp']),
                float(row['trestbps']),
                float(row['chol']),
                float(row['fbs']),
                float(row['restecg']),
                float(row['thalach']),
                float(row['exang']),
                float(row['oldpeak']),
                float(row['slope']),
                str(row.get('origen', 'cleveland')),
                int(row['target'])
            ))
        cursor.executemany("""
            INSERT INTO raw_patients (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, hospital, target)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, raw_records)
        print(f"✅ Inserción completada: {len(raw_records)} registros insertados en raw_patients.")

    # Inserción en curated_patients
    cursor.execute("SELECT COUNT(*) FROM curated_patients;")
    count_curated = cursor.fetchone()[0]

    if count_curated == 0:
        cp_map = {1.0: "Angina Típica", 2.0: "Angina Atípica", 3.0: "Dolor No Anginoso", 4.0: "Asintomático"}
        ecg_map = {0.0: "Normal", 1.0: "Anormalidad ST-T", 2.0: "Hipertrofia Ventricular"}
        slope_map = {1.0: "Ascendente", 2.0: "Plano", 3.0: "Descendente"}

        curated_records = []
        for idx, row in df.iterrows():
            p_id = int(idx + 1)
            age = int(row['age'])
            sex_desc = "Masculino" if row['sex'] == 1.0 else "Femenino"
            cp_desc = cp_map.get(row['cp'], "Otro")
            resting_bp = float(row['trestbps'])
            chol = float(row['chol'])
            fbs_desc = "> 120 mg/dl (Elevada)" if row['fbs'] == 1.0 else "<= 120 mg/dl (Normal)"
            ecg_desc = ecg_map.get(row['restecg'], "Normal")
            max_hr = float(row['thalach'])
            exang_desc = "Sí (Angina inducida)" if row['exang'] == 1.0 else "No"
            oldpeak = float(row['oldpeak'])
            slope_desc = slope_map.get(row['slope'], "Plano")
            hosp = str(row.get('origen', 'cleveland')).capitalize()
            target = int(row['target'])
            
            # Score de riesgo clínico sintético inicial para analítica
            score = (resting_bp / 140.0) + (1.5 if exang_desc == "Sí (Angina inducida)" else 0.0) + (oldpeak * 0.8)
            
            curated_records.append((
                p_id, age, sex_desc, cp_desc, resting_bp, chol, fbs_desc, ecg_desc,
                max_hr, exang_desc, oldpeak, slope_desc, hosp, target, round(score, 2)
            ))

        cursor.executemany("""
            INSERT INTO curated_patients (
                patient_id, age, sex_desc, chest_pain_type, resting_bp, cholesterol,
                fasting_bs_desc, rest_ecg_desc, max_hr, exercise_angina_desc,
                st_depression, st_slope_desc, hospital, has_heart_disease, clinical_severity_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, curated_records)
        print(f"✅ Inserción completada: {len(curated_records)} registros insertados en curated_patients.")

    conn.commit()
    conn.close()
    print(f"🏛️ Base de datos SQLite inicializada exitosamente en: {DB_PATH}")


def execute_sql_query(query: str, params=()):
    """Ejecuta una consulta SQL y retorna un DataFrame de pandas."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_hospital_prevalence_summary():
    """Consulta SQL para auditoría de prevalencia entre los 4 hospitales."""
    query = """
    SELECT 
        hospital AS Hospital,
        COUNT(*) AS Total_Pacientes,
        SUM(has_heart_disease) AS Casos_Positivos,
        COUNT(*) - SUM(has_heart_disease) AS Casos_Negativos,
        ROUND(AVG(has_heart_disease) * 100.0, 2) AS Prevalencia_Porcentaje,
        ROUND(AVG(age), 1) AS Edad_Promedio,
        ROUND(AVG(resting_bp), 1) AS Presion_Arterial_Media
    FROM curated_patients
    GROUP BY hospital
    ORDER BY Prevalencia_Porcentaje DESC;
    """
    return execute_sql_query(query)


def get_chest_pain_risk_summary():
    """Consulta SQL que evalúa el riesgo según el tipo de dolor torácico."""
    query = """
    SELECT 
        chest_pain_type AS Tipo_Dolor_Toracico,
        COUNT(*) AS Total_Pacientes,
        SUM(has_heart_disease) AS Con_Cardiopatia,
        ROUND(AVG(has_heart_disease) * 100.0, 2) AS Tasa_Riesgo_Pct,
        ROUND(AVG(st_depression), 2) AS Depresion_ST_Media
    FROM curated_patients
    GROUP BY chest_pain_type
    ORDER BY Tasa_Riesgo_Pct DESC;
    """
    return execute_sql_query(query)


if __name__ == "__main__":
    init_database()
    print("\n--- Resumen de Prevalencia por Hospital (SQL Query) ---")
    print(get_hospital_prevalence_summary())
    print("\n--- Riesgo por Tipo de Dolor Torácico (SQL Query) ---")
    print(get_chest_pain_risk_summary())
