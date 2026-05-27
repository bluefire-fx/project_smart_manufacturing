import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 0. Finale Hyperparameter für die SVM Klassifizierung
PARAMETER_C = 1000
PARAMETER_GAMMA = "scale"

# 1. Dataframe erstellen
df = pd.read_csv("data/raw/smart-manufacturing-iot-cloud-monitoring-dataset.csv")

# 2. Metadata formatieren
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)

# 3. Zentrale feature liste definieren
x_feature_sensors = [
    "temperature",
    "vibration",
]

x = df[x_feature_sensors]
y = df["anomaly_flag"]

# 4. Train-Test-Split (80% Training für die Modellentwicklung, 20% für den finalen Test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

x_train = x_train.copy()
x_test = x_test.copy()

# 5. SAUBERES WINSORIZING (Grenzen werden NUR vom Trainings-Set berechnet)
winsor_bounds = {}
for col in x_feature_sensors:
    mean = x_train[col].mean()
    std = x_train[col].std()

    lower = mean - 3 * std
    upper = mean + 3 * std
    winsor_bounds[col] = (lower, upper)

    # Begrenzung fehlerfrei auf beide Sets anwenden
    x_train[col] = x_train[col].clip(lower=lower, upper=upper)
    x_test[col] = x_test[col].clip(lower=lower, upper=upper)

# 6. SAUBERE SKALIERUNG (Verhindert Data Leakage)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 7. Finaler SVM Classifier mit Wahrscheinlichkeitsberechnung für das Ranking
# HINWEIS ZUR HYPERPARAMETER-WAHL (gamma='scale' vs. 'auto'):
# Da die Daten mittels StandardScaler auf eine Varianz von 1.0 normiert wurden
# und nur 2 Features genutzt werden, sind beide Gamma-Optionen mathematisch identisch:
# gamma = 1 / (n_features * Var(X)) = 1 / (2 * 1) = 0.5.
# Dies erklärt die exakt identischen Performance-Werte in der GridSearch-Heatmap.
final_model = SVC(
    kernel="rbf",
    C=PARAMETER_C,
    gamma=PARAMETER_GAMMA,
    class_weight="balanced",
    probability=True,  # Aktiviert für die Erstellung der Dringlichkeits-Rangliste
    cache_size=2000,  # Beschleunigt das interne 5-Fold-Platt-Scaling im RAM
    random_state=42,
)

print(
    f"Trainiere das finale C={PARAMETER_C} Modell inkl. Wahrscheinlichkeiten auf den Trainingsdaten..."
)
final_model.fit(x_train_scaled, y_train)

# 8. Pipeline-Package schnüren
production_artifacts = {
    "winsor_bounds": winsor_bounds,  # Wichtig für die korrekte Transformation neuer Live-Daten
    "scaler": scaler,
    "model": final_model,
    "features": x_feature_sensors,
    "parameter_c": PARAMETER_C,
    "parameter_gamma": PARAMETER_GAMMA,
}

# 9. Auf die Festplatte schreiben
output_path = "models/svm_anomaly_detector.joblib"
joblib.dump(production_artifacts, output_path)
print(f"✓ Artefakte erfolgreich in '{output_path}' gespeichert!")
