"""
CardioPredict AI 2.0 - Aplicación Web Interactiva de Inferencia con Redes Neuronales
Corporación Unificada Nacional de Educación Superior (CUN)
Asignatura: Aprendizaje Automático - ACA 2
Docente: Dr. Hamilton Rivera Flor
Equipo de Investigación:
- Marcela Avila Beltrán
- Jaime Andres Cabrera Avendaño
- Cristhian Javier Garzon Jimenez
- Johan Smith Valderrama Sanchez
"""

import os
import json
import base64
import sqlite3
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import joblib

import torch
import torch.nn as nn

# Configuración de página de Streamlit
_fav_cand = os.path.join(os.path.dirname(__file__), "..", "favicon.png")
if not os.path.exists(_fav_cand):
    _fav_cand = os.path.join(os.path.dirname(__file__), "favicon.png")
_app_icon = Image.open(_fav_cand) if os.path.exists(_fav_cand) else "🫀"

st.set_page_config(
    page_title="CardioPredict AI 2.0 | Redes Neuronales CUN",
    page_icon=_app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para apariencia médica profesional idéntica a ACA 1
st.markdown("""
<style>
    .block-container {
        padding-top: 10px !important;
        padding-bottom: 2rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 10px !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.8rem !important;
        min-height: 1.8rem !important;
    }
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 16px 22px !important;
        border-radius: 12px;
        color: white;
        margin-top: 0 !important;
        margin-bottom: 16px !important;
    }
    .main-header h1 {
        font-size: 2.05rem !important;
        margin: 0 !important;
        color: white !important;
    }
    .main-header p {
        margin: 8px 0 0 0 !important;
        font-size: 1.02rem !important;
        opacity: 0.95 !important;
        color: white !important;
    }
    div[data-baseweb="select"] {
        min-height: 42px !important;
    }
    div[data-baseweb="select"] * {
        white-space: normal !important;
        text-overflow: clip !important;
        word-break: break-word !important;
        overflow: visible !important;
    }
    div[data-baseweb="select"] div[class*="ValueContainer"] {
        white-space: normal !important;
        line-height: 1.35 !important;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
    }

    /* Optimización Móvil y Adaptabilidad para Teléfonos Celulares */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 6px !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            padding-bottom: 1.5rem !important;
        }
        div[data-testid="stAppViewBlockContainer"] {
            padding-top: 6px !important;
        }
        .main-header {
            padding: 14px 16px !important;
            margin-bottom: 12px !important;
            border-radius: 10px !important;
        }
        .main-header h1 {
            font-size: 1.28rem !important;
            line-height: 1.25 !important;
        }
        .main-header p {
            font-size: 0.82rem !important;
            margin-top: 6px !important;
            line-height: 1.4 !important;
        }
        /* Apilamiento vertical adaptativo de columnas en móvil */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 10px !important;
        }
        div[data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        /* Pestañas táctiles con desplazamiento horizontal fluido */
        div[data-baseweb="tab-list"] {
            gap: 6px !important;
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            -webkit-overflow-scrolling: touch !important;
            scrollbar-width: none !important;
            padding-bottom: 4px !important;
        }
        div[data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none !important;
        }
        div[data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 0.84rem !important;
            white-space: nowrap !important;
        }
        /* Botones táctiles adaptativos de fácil pulsación */
        .stButton > button, div[data-testid="stForm"] button {
            min-height: 48px !important;
            font-size: 0.98rem !important;
            font-weight: 700 !important;
            width: 100% !important;
            border-radius: 8px !important;
        }
        /* Cajas de métricas responsivas */
        div[data-testid="stMetric"] {
            padding: 10px 12px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "database" / "cardiopredict.db"
ASSETS_DIR = BASE_DIR / "assets"

DEVICE = torch.device("cpu")


# -------------------------------------------------------------
# 1. Definición de Arquitecturas de PyTorch
# -------------------------------------------------------------
class CardioPredictDeepMLP(nn.Module):
    def __init__(self, input_dim: int = 15):
        super(CardioPredictDeepMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)


class CardioPredict1DCNN(nn.Module):
    def __init__(self, input_dim: int = 15):
        super(CardioPredict1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        flattened_dim = 32 * (input_dim // 2 // 2)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)


# -------------------------------------------------------------
# 2. Carga en Caché de Modelos y Preprocesador
# -------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    except Exception:
        # Auto-recuperación robusta ante diferencias de versión entre entornos scikit-learn
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer

        num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']

        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ]), cat_cols)
        ])
        conn = sqlite3.connect(DB_PATH)
        df_fit = pd.read_sql_query("SELECT age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope FROM raw_patients", conn)
        conn.close()
        preprocessor.fit(df_fit)
        joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    mlp_model = CardioPredictDeepMLP(input_dim=15)
    mlp_model.load_state_dict(torch.load(MODELS_DIR / "cardiopredict_deep_mlp.pt", map_location=DEVICE))
    mlp_model.eval()

    cnn_model = CardioPredict1DCNN(input_dim=15)
    cnn_model.load_state_dict(torch.load(MODELS_DIR / "cardiopredict_1d_cnn.pt", map_location=DEVICE))
    cnn_model.eval()

    return preprocessor, mlp_model, cnn_model


try:
    preprocessor, mlp_model, cnn_model = load_assets()
except Exception as e:
    st.error(f"Error cargando modelos o preprocesador: {e}")
    st.stop()


# -------------------------------------------------------------
# 3. Barra Lateral (Sidebar)
# -------------------------------------------------------------
with st.sidebar:
    _logo_file = os.path.join(os.path.dirname(__file__), "favicon.png")
    if not os.path.exists(_logo_file):
        _logo_file = os.path.join(os.path.dirname(__file__), "..", "favicon.png")
    if os.path.exists(_logo_file):
        with open(_logo_file, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode()
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            <img src="data:image/png;base64,{_b64}" alt="CUN" style="width:38px; height:38px; border-radius:8px; background:white; padding:3px; object-fit:contain; box-shadow:0 2px 6px rgba(0,0,0,0.2); flex-shrink:0;">
            <div>
                <div style="font-size:1.3rem; font-weight:800; line-height:1.2;">CardioPredict AI 2.0</div>
                <div style="font-size:0.8rem; opacity:0.85; font-weight:600;">ACA N° 2 - Deep Learning & SQL</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:2.2rem; margin-bottom:8px;'>🫀</div>", unsafe_allow_html=True)
        st.title("CardioPredict AI 2.0")
        st.caption("**ACA N° 2 - Redes Neuronales & Deep Learning**")
    st.markdown("---")
    
    st.markdown("**🎓 Institución:**")
    st.text("Corporación Unificada Nacional (CUN)\nEspecialización en Inteligencia Artificial")
    
    st.markdown("**👨‍🏫 Docente:**")
    st.text("Dr. Hamilton Rivera Flor")
    
    st.markdown("**👥 Integrantes del Equipo (Grupo 12):**")
    st.markdown("""
    * **Marcela Avila Beltrán**
    * **Jaime Andres Cabrera Avendaño**
    * **Cristhian Javier Garzon Jimenez**
    * **Johan Smith Valderrama Sanchez**
    """)
    st.markdown("---")

    st.subheader("⚙️ Modelo Neuronal Activo")
    model_choice = st.radio(
        "Seleccione la arquitectura de inferencia:",
        [
            "🧠 Deep MLP (Red Profunda a la Medida)",
            "🔬 1D-CNN (Red Convolucional de Bioseñales)"
        ],
        index=0
    )

    threshold = st.slider(
        "Umbral Clínico de Decisión (Threshold)",
        min_value=0.20, max_value=0.80, value=0.45, step=0.05,
        help="Calibrado en 0.45 para maximizar la Sensibilidad (Recall) médica, reduciendo falsos negativos."
    )

    st.markdown("---")
    st.markdown("""
    🔗 **Navegación Rápida:**  
    • [Portal Principal ACA 2](https://iacun.lat/aprendizajeautomatico/aca2/)  
    • [Portal ACA 1](https://iacun.lat/aprendizajeautomatico/aca1/)  
    • [Repositorio GitHub](https://github.com/johansmith/aca2-deep-learning-cun)
    """)


# -------------------------------------------------------------
# 4. Encabezado Principal y Pestañas
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🫀 CardioPredict AI 2.0: Sistema Inteligente de Triaje Clínico</h1>
    <p>
        Plataforma de Diagnóstico Asistido de Riesgo Cardiovascular mediante Redes Neuronales Artificiales (Deep Learning) y Gestión SQL — Especialización en IA CUN
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🩺 Inferencia y Triaje en Tiempo Real",
    "🏛️ Exploración y Consultas SQL (cardiopredict.db)",
    "📈 Métricas & Curvas de Aprendizaje",
    "📚 Evolución ACA 1 vs ACA 2 & Ficha Técnica"
])


# =============================================================
# TAB 1: INFERENCIA CLÍNICA CON RED NEURONAL
# =============================================================
with tabs[0]:
    st.markdown("### 📋 Casos de Prueba Rápidos (Presets)")
    st.caption("Seleccione un perfil clínico predefinido para autocompletar de inmediato los parámetros del paciente:")

    preset = st.selectbox(
        "Cargar perfil predefinido:",
        [
            "Personalizado (Ingrese sus propios datos)",
            "🚨 Caso Crítico de Urgencias (Alto Riesgo)",
            "🟡 Caso Ambiguo / Dolor Atípico (Riesgo Moderado)",
            "🟢 Paciente Sano de Control (Bajo Riesgo)"
        ],
        index=0
    )

    # Asignar valores por defecto según el preset seleccionado
    if "🚨 Caso Crítico" in preset:
        def_age, def_sex, def_cp = 63, "Masculino", "Asintomático (Tipo 4)"
        def_bp, def_chol, def_fbs = 160, 286, "Sí (> 120 mg/dl)"
        def_ecg, def_thalach, def_exang = "Hipertrofia Ventricular Izquierda (Tipo 2)", 108, "Sí"
        def_oldpeak, def_slope = 2.6, "Descendente (Tipo 3)"
    elif "🟢 Paciente Sano" in preset:
        def_age, def_sex, def_cp = 37, "Femenino", "Angina No Anginosa (Tipo 3)"
        def_bp, def_chol, def_fbs = 120, 195, "No (<= 120 mg/dl)"
        def_ecg, def_thalach, def_exang = "Normal (Tipo 0)", 175, "No"
        def_oldpeak, def_slope = 0.2, "Ascendente (Tipo 1)"
    elif "🟡 Caso Ambiguo" in preset:
        def_age, def_sex, def_cp = 54, "Masculino", "Angina Atípica (Tipo 2)"
        def_bp, def_chol, def_fbs = 135, 220, "No (<= 120 mg/dl)"
        def_ecg, def_thalach, def_exang = "Anormalidad Onda ST-T (Tipo 1)", 140, "No"
        def_oldpeak, def_slope = 1.0, "Plano (Tipo 2)"
    else:
        def_age, def_sex, def_cp = 55, "Masculino", "Asintomático (Tipo 4)"
        def_bp, def_chol, def_fbs = 135, 230, "No (<= 120 mg/dl)"
        def_ecg, def_thalach, def_exang = "Normal (Tipo 0)", 145, "No"
        def_oldpeak, def_slope = 1.2, "Plano (Tipo 2)"

    st.markdown("---")
    st.subheader("📋 Parámetros Fisiológicos del Paciente")
    st.caption("Ingrese los signos vitales y datos paraclínicos para inferencia en tiempo real con la Red Neuronal seleccionada.")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Edad (años)", min_value=18, max_value=100, value=def_age, step=1)
            sex = st.selectbox("Sexo Biológico", ["Masculino", "Femenino"], index=0 if def_sex == "Masculino" else 1)
            cp = st.selectbox("Tipo de Dolor Torácico (Chest Pain)", [
                "Angina Típica (Tipo 1)",
                "Angina Atípica (Tipo 2)",
                "Angina No Anginosa (Tipo 3)",
                "Asintomático (Tipo 4)"
            ], index=["Angina Típica (Tipo 1)", "Angina Atípica (Tipo 2)", "Angina No Anginosa (Tipo 3)", "Asintomático (Tipo 4)"].index(def_cp))
            trestbps = st.number_input("Presión Arterial en Reposo (mmHg)", min_value=80, max_value=220, value=def_bp, step=1)

        with col2:
            chol = st.number_input("Colesterol Sérico (mg/dl)", min_value=100, max_value=600, value=def_chol, step=1)
            fbs = st.selectbox("Glucemia en Ayunas > 120 mg/dl", ["No (<= 120 mg/dl)", "Sí (> 120 mg/dl)"], index=0 if "No" in def_fbs else 1)
            restecg = st.selectbox("Electrocardiograma en Reposo", [
                "Normal (Tipo 0)",
                "Anormalidad Onda ST-T (Tipo 1)",
                "Hipertrofia Ventricular Izquierda (Tipo 2)"
            ], index=["Normal (Tipo 0)", "Anormalidad Onda ST-T (Tipo 1)", "Hipertrofia Ventricular Izquierda (Tipo 2)"].index(def_ecg))
            thalach = st.number_input("Frecuencia Cardíaca Máxima (lpm)", min_value=60, max_value=220, value=def_thalach, step=1)

        with col3:
            exang = st.selectbox("Angina Inducida por Ejercicio", ["No", "Sí"], index=0 if def_exang == "No" else 1)
            oldpeak = st.number_input("Depresión del Segmento ST (Oldpeak)", min_value=-2.0, max_value=7.0, value=float(def_oldpeak), step=0.1)
            slope = st.selectbox("Pendiente del Segmento ST de Esfuerzo", [
                "Ascendente (Tipo 1)",
                "Plano (Tipo 2)",
                "Descendente (Tipo 3)"
            ], index=["Ascendente (Tipo 1)", "Plano (Tipo 2)", "Descendente (Tipo 3)"].index(def_slope))

        submitted = st.form_submit_button("🚀 Evaluar con Red Neuronal", use_container_width=True)

    if submitted:
        # Mapeo a codificación numérica del modelo
        sex_val = 1.0 if sex == "Masculino" else 0.0
        cp_map = {"Angina Típica (Tipo 1)": 1.0, "Angina Atípica (Tipo 2)": 2.0, "Angina No Anginosa (Tipo 3)": 3.0, "Asintomático (Tipo 4)": 4.0}
        cp_val = cp_map[cp]
        fbs_val = 1.0 if "Sí" in fbs else 0.0
        ecg_map = {"Normal (Tipo 0)": 0.0, "Anormalidad Onda ST-T (Tipo 1)": 1.0, "Hipertrofia Ventricular Izquierda (Tipo 2)": 2.0}
        ecg_val = ecg_map[restecg]
        exang_val = 1.0 if exang == "Sí" else 0.0
        slope_map = {"Ascendente (Tipo 1)": 1.0, "Plano (Tipo 2)": 2.0, "Descendente (Tipo 3)": 3.0}
        slope_val = slope_map[slope]

        # DataFrame con los nombres exactos esperados por el preprocesador
        patient_df = pd.DataFrame([{
            'age': float(age),
            'sex': sex_val,
            'cp': cp_val,
            'trestbps': float(trestbps),
            'chol': float(chol),
            'fbs': fbs_val,
            'restecg': ecg_val,
            'thalach': float(thalach),
            'exang': exang_val,
            'oldpeak': float(oldpeak),
            'slope': slope_val
        }])

        # Transformación con el ColumnTransformer
        X_proc = preprocessor.transform(patient_df)
        X_tensor = torch.tensor(X_proc, dtype=torch.float32).to(DEVICE)

        # Inferencia con la red neuronal seleccionada
        selected_net = mlp_model if "Deep MLP" in model_choice else cnn_model
        with torch.no_grad():
            prob = selected_net.predict_proba(X_tensor).item()

        is_high_risk = prob >= threshold
        risk_pct = prob * 100.0

        st.markdown("---")
        st.subheader("🎯 Resultado de la Inferencia Neuronal")

        res_col1, res_col2 = st.columns([1, 1.4])

        with res_col1:
            if prob >= threshold:
                st.error("### 🚨 ALTO RIESGO CARDIOVASCULAR DETECTADO (Prioridad I - Alerta Roja)")
                st.markdown(f"**Probabilidad Estimada por la Red:** `{risk_pct:.2f}%` (Umbral Crítico: `{threshold*100:.0f}%`)")
                st.progress(prob)
                st.markdown("""
                **Acción Clínica Recomendada (Prioridad I - Código Rojo):**
                * Derivación urgente a Unidad de Cuidados Coronarios / Hemodinamia para cateterismo y ecocardiograma.
                * Monitoreo continuo de enzimas cardíacas seriadas (Troponina I/T ultrasensible).
                * Ajuste farmacológico antianginoso, antiagregante y antihipertensivo inmediato.
                """)
            elif prob >= 0.20:
                st.warning("### 🟡 RIESGO MODERADO / ZONA GRIS CLÍNICA (Prioridad II - Alerta Amarilla)")
                st.markdown(f"**Probabilidad Estimada por la Red:** `{risk_pct:.2f}%` (Zona de Observación: `20%` - `{threshold*100:.0f}%`)")
                st.progress(prob)
                st.markdown("""
                **Acción Clínica Recomendada (Prioridad II - Observación y Monitoreo):**
                * Paciente con hallazgos atípicos o limítrofes (dolor atípico / ST sospechoso). No dar de alta definitiva.
                * Mantener en sala de observación y programar prueba de esfuerzo (ergometría) y Holter ECG 24h.
                * Repetir biomarcadores cardíacos a las 6 horas y seguimiento estricto de presión arterial.
                """)
            else:
                st.success("### 🟢 BAJO RIESGO CARDIOVASCULAR (Prioridad III - Sano / Control)")
                st.markdown(f"**Probabilidad Estimada por la Red:** `{risk_pct:.2f}%` (Bajo Riesgo: `< 20%`)")
                st.progress(prob)
                st.markdown("""
                **Acción Clínica Recomendada (Prioridad III - Control Ambulatorio):**
                * Control rutinario en atención primaria anual.
                * Promoción de estilos de vida saludables, actividad aeróbica regular y dieta cardioprotectora.
                * Seguimiento ambulatorio preventivo de perfil lipídico y glucemia.
                """)

        with res_col2:
            st.markdown(f"#### 🧠 Detalle Técnico de la Inferencia ({'Deep MLP' if 'Deep MLP' in model_choice else '1D-CNN'})")
            st.markdown(f"""
            * **Arquitectura:** `{type(selected_net).__name__}`
            * **Vector de entrada transformado:** `{X_proc.shape[1]}` dimensiones estandarizadas.
            * **Sensibilidad (Recall) del modelo en prueba:** **{('93.14%' if 'Deep MLP' in model_choice else '85.29%')}**
            * **Área bajo la Curva ROC (AUC):** **{('90.32%' if 'Deep MLP' in model_choice else '89.04%')}**
            * **Manejo de Falsos Negativos:** Penalización ajustada en función de pérdida `BCEWithLogitsLoss`.
            """)


# =============================================================
# TAB 2: GESTIÓN Y CONSULTAS SQL
# =============================================================
with tabs[1]:
    st.subheader("🏛️ Gestión de Datos Mediante Base de Datos Relacional SQLite")
    st.markdown("""
    En cumplimiento del **Paso 1 de la Rúbrica Oficial del ACA 2**, el conjunto de datos de 918 pacientes de los 4 hospitales
    ha sido estructurado, indexado y gestionado dentro de la base de datos relacional `database/cardiopredict.db`.
    """)

    sql_col1, sql_col2 = st.columns([1, 2])

    with sql_col1:
        st.markdown("#### ⚡ Consultas SQL Predefinidas")
        query_option = st.selectbox(
            "Seleccione una consulta analítica:",
            [
                "1. Prevalencia de Cardiopatía por Hospital",
                "2. Tasa de Riesgo según Tipo de Dolor Torácico",
                "3. Pacientes con Crisis Hipertensiva (PA > 160 mmHg)",
                "4. Distribución de Edad y Cardiopatía",
                "5. Explorar Primeros 10 Registros Curados"
            ]
        )

    queries_dict = {
        "1. Prevalencia de Cardiopatía por Hospital": """
            SELECT 
                hospital AS Hospital,
                COUNT(*) AS Total_Pacientes,
                SUM(has_heart_disease) AS Casos_Positivos,
                COUNT(*) - SUM(has_heart_disease) AS Casos_Sanos,
                ROUND(AVG(has_heart_disease) * 100.0, 2) AS Prevalencia_Pct,
                ROUND(AVG(age), 1) AS Edad_Media,
                ROUND(AVG(resting_bp), 1) AS Presion_Media
            FROM curated_patients
            GROUP BY hospital
            ORDER BY Prevalencia_Pct DESC;
        """,
        "2. Tasa de Riesgo según Tipo de Dolor Torácico": """
            SELECT 
                chest_pain_type AS Tipo_Dolor,
                COUNT(*) AS Total,
                SUM(has_heart_disease) AS Enfermos,
                ROUND(AVG(has_heart_disease) * 100.0, 2) AS Riesgo_Pct,
                ROUND(AVG(st_depression), 2) AS Depresion_ST_Media
            FROM curated_patients
            GROUP BY chest_pain_type
            ORDER BY Riesgo_Pct DESC;
        """,
        "3. Pacientes con Crisis Hipertensiva (PA > 160 mmHg)": """
            SELECT 
                patient_id AS ID,
                age AS Edad,
                sex_desc AS Sexo,
                resting_bp AS Presion_Arterial,
                chest_pain_type AS Dolor,
                has_heart_disease AS Cardiopatia,
                hospital AS Hospital
            FROM curated_patients
            WHERE resting_bp >= 160
            ORDER BY resting_bp DESC
            LIMIT 15;
        """,
        "4. Distribución de Edad y Cardiopatía": """
            SELECT 
                CASE 
                    WHEN age < 40 THEN '< 40 años'
                    WHEN age BETWEEN 40 AND 49 THEN '40 - 49 años'
                    WHEN age BETWEEN 50 AND 59 THEN '50 - 59 años'
                    WHEN age BETWEEN 60 AND 69 THEN '60 - 69 años'
                    ELSE '>= 70 años'
                END AS Rango_Etario,
                COUNT(*) AS Total,
                SUM(has_heart_disease) AS Casos_Positivos,
                ROUND(AVG(has_heart_disease) * 100.0, 2) AS Tasa_Cardiopatia_Pct
            FROM curated_patients
            GROUP BY Rango_Etario
            ORDER BY Rango_Etario;
        """,
        "5. Explorar Primeros 10 Registros Curados": """
            SELECT * FROM curated_patients LIMIT 10;
        """
    }

    selected_sql = queries_dict[query_option]

    with sql_col2:
        st.markdown("#### 📝 Código SQL Ejecutado")
        st.code(selected_sql, language="sql")

    try:
        conn = sqlite3.connect(DB_PATH)
        df_result = pd.read_sql_query(selected_sql, conn)
        conn.close()
        st.dataframe(df_result, use_container_width=True)
    except Exception as e:
        st.error(f"Error ejecutando consulta SQL: {e}")


# =============================================================
# TAB 3: MÉTRICAS Y CURVAS DE APRENDIZAJE
# =============================================================
with tabs[2]:
    st.subheader("📈 Rendimiento Cuantitativo y Curvas de Aprendizaje (Paso 4)")
    st.markdown("""
    Evaluación multimétrica sobre el conjunto de prueba independiente de **184 pacientes (Cleveland, Hungría, Suiza y VA)**
    demostrando la capacidad de generalización y prevención de sobreajuste de las redes neuronales.
    """)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Sensibilidad Médica (Recall)", "93.14%", "+7.85% vs ACA 1")
    mcol2.metric("Área ROC (ROC-AUC)", "90.32%", "+0.82% vs ACA 1")
    mcol3.metric("Exactitud General (Accuracy)", "82.61%", "+3.26% vs ACA 1")
    mcol4.metric("F1-Score Ponderado", "85.59%", "+3.51% vs ACA 1")

    st.markdown("---")
    gcol1, gcol2 = st.columns(2)

    with gcol1:
        st.markdown("#### Curvas de Pérdida (Loss) por Época")
        if (ASSETS_DIR / "loss_curves.png").exists():
            st.image(str(ASSETS_DIR / "loss_curves.png"), use_container_width=True)

    with gcol2:
        st.markdown("#### Curvas de Exactitud y Sensibilidad Clínica")
        if (ASSETS_DIR / "accuracy_curves.png").exists():
            st.image(str(ASSETS_DIR / "accuracy_curves.png"), use_container_width=True)

    gcol3, gcol4 = st.columns(2)

    with gcol3:
        st.markdown("#### Matriz de Confusión en Prueba (N=184)")
        if (ASSETS_DIR / "confusion_matrix_deep_learning.png").exists():
            st.image(str(ASSETS_DIR / "confusion_matrix_deep_learning.png"), use_container_width=True)

    with gcol4:
        st.markdown("#### Curva ROC Comparativa (Deep MLP vs 1D-CNN)")
        if (ASSETS_DIR / "roc_curve_deep_learning.png").exists():
            st.image(str(ASSETS_DIR / "roc_curve_deep_learning.png"), use_container_width=True)


# =============================================================
# TAB 4: EVOLUCIÓN ACA 1 VS ACA 2 & FICHA TÉCNICA
# =============================================================
with tabs[3]:
    st.subheader("🔬 Evolución Metodológica: Del Machine Learning Clásico a Redes Neuronales Profundas")
    st.markdown("""
    En este segundo corte, el equipo avanzó desde algoritmos supervisados tradicionales (Random Forest, Regresión Logística, Árboles)
    hacia **arquitecturas profundas de Deep Learning** con capas ocultas jerárquicas, aumentación de datos tabular y regularización con Dropout.
    """)

    if (ASSETS_DIR / "head_to_head_comparison.png").exists():
        st.image(str(ASSETS_DIR / "head_to_head_comparison.png"), use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### 🏛️ Ficha Técnica Institucional
    * **Institución:** Corporación Unificada Nacional de Educación Superior (CUN)
    * **Programa:** Especialización en Inteligencia Artificial — Escuela de Ingeniería
    * **Asignatura:** Aprendizaje Automático | **Docente:** Dr. Hamilton Rivera Flor
    * **Pilar:** Proyecto 1: Salud (Detección y Triaje Asistido de Cardiopatías)
    * **Dataset:** *UCI Heart Disease Dataset* (Janosi et al., 1988) — 918 pacientes (4 hospitales internacionales)
    * **Integrantes del Equipo:**
      1. Marcela Avila Beltrán
      2. Jaime Andres Cabrera Avendaño
      3. Cristhian Javier Garzon Jimenez
      4. Johan Smith Valderrama Sanchez
    """)

    st.markdown("---")
    st.markdown("""
    ### 📑 Entregables Oficiales del Proyecto
    * 📑 **Póster Académico Oficial (PDF):** [Visualizar en línea](https://iacun.lat/aprendizajeautomatico/aca2/poster/CardioPredict_AI_2_Poster_Infografia.pdf) | [Descargar archivo original](https://iacun.lat/aprendizajeautomatico/aca2/poster/CardioPredict_AI_2_Poster_Infografia.pdf)
    * 💻 **Repositorio GitHub Oficial:** [johansmith/aca2-deep-learning-cun](https://github.com/johansmith/aca2-deep-learning-cun)
    * 🌐 **Portal Central ACA 2:** [https://iacun.lat/aprendizajeautomatico/aca2/](https://iacun.lat/aprendizajeautomatico/aca2/)
    """)
