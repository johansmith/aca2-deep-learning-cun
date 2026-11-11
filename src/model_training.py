"""
CardioPredict AI 2.0 - Pipeline de Deep Learning y Redes Neuronales Artificiales
Corporación Unificada Nacional de Educación Superior (CUN)
Asignatura: Aprendizaje Automático - ACA 2
Docente: Dr. Hamilton Rivera Flor
Equipo: Marcela Avila, Jaime Cabrera, Cristhian Garzón, Johan Smith Valderrama
"""

import os
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import joblib

# Semillas de reproducibilidad
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "cardiopredict.db"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
print(f"🚀 Dispositivo de cómputo para Deep Learning: {DEVICE}")


# -------------------------------------------------------------
# 1. Dataset de PyTorch y Data Augmentation Tabular (Paso 2)
# -------------------------------------------------------------
class TabularHeartDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def apply_tabular_data_augmentation(X_train: np.ndarray, y_train: np.ndarray, noise_level=0.03, augment_ratio=0.20):
    """
    Data Augmentation Tabular: aplica perturbación gaussiana clínica controlada
    a las características continuas, expandiendo la variedad de patrones
    y reforzando la generalización sin alterar las etiquetas patológicas.
    """
    n_samples = len(X_train)
    n_aug = int(n_samples * augment_ratio)
    idx = np.random.choice(n_samples, size=n_aug, replace=True)
    
    X_sample = X_train[idx].copy()
    y_sample = y_train[idx].copy()

    # Ruido proporcional a la desviación de cada variable
    std_dev = np.std(X_train, axis=0, keepdims=True) + 1e-4
    noise = np.random.normal(0, noise_level, size=X_sample.shape) * std_dev
    X_aug = X_sample + noise

    X_final = np.vstack([X_train, X_aug])
    y_final = np.concatenate([y_train, y_sample])

    print(f"✨ Data Augmentation aplicado: {n_samples} originales + {n_aug} generados = {len(X_final)} muestras.")
    return X_final, y_final


# -------------------------------------------------------------
# 2. Arquitecturas de Redes Neuronales (Paso 3)
# -------------------------------------------------------------
class CardioPredictDeepMLP(nn.Module):
    """
    Red Neuronal Artificial Profunda (Multilayer Perceptron)
    Arquitectura jerárquica con Dropout regularizador y activaciones no lineales ReLU.
    """
    def __init__(self, input_dim: int):
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

            nn.Linear(16, 1)  # Retorna logits
        )

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)


class CardioPredict1DCNN(nn.Module):
    """
    Red Neuronal Convolucional 1D (1D-CNN) para extracción
    de patrones cruzados entre biomarcadores clínicos.
    """
    def __init__(self, input_dim: int = 15):
        super(CardioPredict1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 15 -> 7

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 7 -> 3

        flattened_dim = 32 * (input_dim // 2 // 2)  # 32 * 3 = 96

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Aplanar a (batch, flattened_dim)
        return self.classifier(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)


# -------------------------------------------------------------
# 3. Flujo de Entrenamiento y Evaluación
# -------------------------------------------------------------
def load_data_from_sql():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, target
    FROM raw_patients;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def build_preprocessor():
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ]
    )
    return preprocessor


def train_neural_network(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50):
    model.to(DEVICE)
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_recall': []
    }

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        # Modo Entrenamiento
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_b)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend((probs >= 0.5).astype(int).flatten())
            train_targets.extend(y_b.cpu().numpy().flatten())

        ep_train_loss = train_loss / len(train_loader.dataset)
        ep_train_acc = accuracy_score(train_targets, train_preds)

        # Modo Validación
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                logits = model(X_v)
                loss = criterion(logits, y_v)
                val_loss += loss.item() * len(y_v)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend((probs >= 0.5).astype(int).flatten())
                val_targets.extend(y_v.cpu().numpy().flatten())

        ep_val_loss = val_loss / len(val_loader.dataset)
        ep_val_acc = accuracy_score(val_targets, val_preds)
        ep_val_rec = recall_score(val_targets, val_preds, zero_division=0)

        scheduler.step(ep_val_loss)

        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history['train_loss'].append(ep_train_loss)
        history['val_loss'].append(ep_val_loss)
        history['train_acc'].append(ep_train_acc)
        history['val_acc'].append(ep_val_acc)
        history['val_recall'].append(ep_val_rec)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {ep_train_loss:.4f} | Val Loss: {ep_val_loss:.4f} | Val Acc: {ep_val_acc:.4f} | Val Recall: {ep_val_rec:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(model, X_test, y_test, threshold=0.45):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        probs = model.predict_proba(X_t).cpu().numpy().flatten()
        preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'probs': probs,
        'preds': preds
    }


def generate_charts(history_mlp, history_cnn, test_mlp, test_cnn, y_test):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Curva de Pérdida (Loss)
    plt.figure(figsize=(9, 5), dpi=300)
    epochs = range(1, len(history_mlp['train_loss']) + 1)
    plt.plot(epochs, history_mlp['train_loss'], label='Entrenamiento (Train Loss)', color='#2563eb', linewidth=2.5)
    plt.plot(epochs, history_mlp['val_loss'], label='Validación (Val Loss)', color='#ef4444', linewidth=2.5, linestyle='--')
    plt.title('Curvas de Pérdida (Loss) — Red Neuronal Profunda (Deep MLP)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Épocas', fontsize=11)
    plt.ylabel('Binary Cross-Entropy Loss', fontsize=11)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "loss_curves.png")
    plt.close()

    # 2. Curvas de Exactitud y Sensibilidad Clínica
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(epochs, history_mlp['val_acc'], label='Exactitud en Validación (Accuracy)', color='#059669', linewidth=2.5)
    plt.plot(epochs, history_mlp['val_recall'], label='Sensibilidad Clínica (Recall)', color='#d97706', linewidth=2.5, linestyle='-.')
    plt.title('Evolución de Rendimiento en Validación por Época — Deep MLP', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Épocas', fontsize=11)
    plt.ylabel('Puntuación (0 a 1.0)', fontsize=11)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "accuracy_curves.png")
    plt.close()

    # 3. Matriz de Confusión Clínica
    plt.figure(figsize=(6, 5), dpi=300)
    cm = np.array(test_mlp['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Sano (0)', 'Cardiopatía (1)'],
                yticklabels=['Sano (0)', 'Cardiopatía (1)'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(f"Matriz de Confusión Clínica (N=184 Pacientes)\nDeep MLP — Detección: {cm[1,1]} de {cm[1,0]+cm[1,1]} Casos", fontsize=11, fontweight='bold', pad=10)
    plt.xlabel('Predicción de la Red Neuronal', fontsize=10, fontweight='bold')
    plt.ylabel('Diagnóstico Médico Real', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "confusion_matrix_deep_learning.png")
    plt.close()

    # 4. Curvas ROC Comparativas
    fpr_m, tpr_m, _ = roc_curve(y_test, test_mlp['probs'])
    fpr_c, tpr_c, _ = roc_curve(y_test, test_cnn['probs'])

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr_m, tpr_m, color='#2563eb', linewidth=2.5, label=f'Deep MLP (AUC = {test_mlp["roc_auc"]:.4f})')
    plt.plot(fpr_c, tpr_c, color='#7c3aed', linewidth=2.5, linestyle='--', label=f'1D-CNN Médica (AUC = {test_cnn["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle=':', label='Azar Clínico (AUC = 0.50)')
    plt.title('Curvas ROC Comparativas de Redes Neuronales Artificiales', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11)
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)', fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "roc_curve_deep_learning.png")
    plt.close()

    # 5. Comparativa Científica: ACA 1 (Random Forest) vs ACA 2 (Deep MLP y 1D-CNN)
    metrics_names = ['Exactitud', 'Recall (Sensibilidad)', 'Precisión', 'F1-Score', 'ROC-AUC']
    aca1_rf = [0.7935, 0.8529, 0.7909, 0.8208, 0.8950]
    aca2_mlp = [test_mlp['accuracy'], test_mlp['recall'], test_mlp['precision'], test_mlp['f1'], test_mlp['roc_auc']]
    aca2_cnn = [test_cnn['accuracy'], test_cnn['recall'], test_cnn['precision'], test_cnn['f1'], test_cnn['roc_auc']]

    x = np.arange(len(metrics_names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.bar(x - width, aca1_rf, width, label='ACA 1: Random Forest (Clásico)', color='#94a3b8')
    ax.bar(x, aca2_mlp, width, label='ACA 2: Deep MLP (Red Profunda)', color='#2563eb')
    ax.bar(x + width, aca2_cnn, width, label='ACA 2: 1D-CNN (Convolucional)', color='#7c3aed')

    ax.set_ylabel('Valor de Métrica (0 a 1.0)', fontsize=11, fontweight='bold')
    ax.set_title('Evolución Metodológica ACA 1 vs ACA 2: ML Tradicional vs Deep Learning', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10, fontweight='bold')
    ax.set_ylim(0.65, 0.96)
    ax.legend(frameon=True, facecolor='white', loc='lower right')
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "head_to_head_comparison.png")
    plt.close()
    print("📊 Figuras y gráficas de alta resolución generadas exitosamente en assets/.")


def main():
    print("=" * 70)
    print("CardioPredict AI 2.0: Pipeline de Deep Learning y Redes Neuronales")
    print("=" * 70)

    # 1. Cargar desde SQLite (Paso 1)
    df = load_data_from_sql()
    print(f"📁 Pacientes leídos desde SQLite (cardiopredict.db): {len(df)}")

    X = df.drop(columns=['target'])
    y = df['target'].values

    # 2. División Estratificada 80/20 idéntica a ACA 1 (184 de test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # 3. Preprocesamiento (Paso 2)
    preprocessor = build_preprocessor()
    X_train_prep = preprocessor.fit_transform(X_train_raw)
    X_test_prep = preprocessor.transform(X_test_raw)
    input_dim = X_train_prep.shape[1]
    print(f"⚙️ Vector de características procesado: {input_dim} dimensiones.")

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    # 4. Data Augmentation Tabular (Paso 2)
    X_train_aug, y_train_aug = apply_tabular_data_augmentation(
        X_train_prep, y_train, noise_level=0.03, augment_ratio=0.20
    )

    # División interna Train / Val (85% / 15%)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_aug, y_train_aug, test_size=0.15, random_state=SEED, stratify=y_train_aug
    )

    train_loader = DataLoader(TabularHeartDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    val_loader = DataLoader(TabularHeartDataset(X_val, y_val), batch_size=32, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()

    # 5. Modelo 1: Deep MLP (Paso 3)
    print("\n🧠 --- Entrenando Modelo 1: Red Neuronal Profunda (Deep MLP) ---")
    mlp = CardioPredictDeepMLP(input_dim)
    opt_mlp = optim.Adam(mlp.parameters(), lr=0.003, weight_decay=1e-4)
    sched_mlp = optim.lr_scheduler.ReduceLROnPlateau(opt_mlp, mode='min', factor=0.5, patience=5)
    mlp, hist_mlp = train_neural_network(mlp, train_loader, val_loader, criterion, opt_mlp, sched_mlp, epochs=50)

    # 6. Modelo 2: 1D-CNN Médica (Paso 3)
    print("\n🔬 --- Entrenando Modelo 2: Red Convolucional Unidimensional (1D-CNN) ---")
    cnn = CardioPredict1DCNN(input_dim)
    opt_cnn = optim.Adam(cnn.parameters(), lr=0.002, weight_decay=1e-4)
    sched_cnn = optim.lr_scheduler.ReduceLROnPlateau(opt_cnn, mode='min', factor=0.5, patience=5)
    cnn, hist_cnn = train_neural_network(cnn, train_loader, val_loader, criterion, opt_cnn, sched_cnn, epochs=50)

    # 7. Evaluación Rigurosa en Test Set (Paso 4)
    print("\n📊 --- Evaluación sobre Prueba Independiente (N=184 Pacientes) ---")
    res_mlp = evaluate_model(mlp, X_test_prep, y_test, threshold=0.45)
    res_cnn = evaluate_model(cnn, X_test_prep, y_test, threshold=0.45)

    print(f"\n[Deep MLP] -> Exactitud: {res_mlp['accuracy']:.4f} | Recall: {res_mlp['recall']:.4f} | F1: {res_mlp['f1']:.4f} | ROC-AUC: {res_mlp['roc_auc']:.4f}")
    print(f"[1D-CNN]   -> Exactitud: {res_cnn['accuracy']:.4f} | Recall: {res_cnn['recall']:.4f} | F1: {res_cnn['f1']:.4f} | ROC-AUC: {res_cnn['roc_auc']:.4f}")

    # Guardar modelos
    torch.save(mlp.state_dict(), MODELS_DIR / "cardiopredict_deep_mlp.pt")
    torch.save(cnn.state_dict(), MODELS_DIR / "cardiopredict_1d_cnn.pt")

    # Guardar métricas en JSON
    metrics_data = {
        'deep_mlp': {k: v for k, v in res_mlp.items() if k not in ['probs', 'preds']},
        '1d_cnn': {k: v for k, v in res_cnn.items() if k not in ['probs', 'preds']},
        'dataset_total': len(df),
        'augmented_train_samples': len(X_train_aug),
        'test_samples': len(y_test)
    }
    with open(MODELS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    # 8. Guardar Figuras
    generate_charts(hist_mlp, hist_cnn, res_mlp, res_cnn, y_test)
    print("\n✅ Pipeline de Deep Learning finalizado con éxito.")


if __name__ == "__main__":
    main()
