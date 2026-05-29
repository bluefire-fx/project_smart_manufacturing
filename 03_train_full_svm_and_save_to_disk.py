import joblib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
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

# ROC Plot vom final_model
# 1. Wahrscheinlichkeiten für die Anomalie-Klasse (Index 1) auf dem Test-Set vorhersagen
# x_test_scaled und y_test kommen aus deinem bestehenden Train-Test-Split
y_scores = final_model.predict_proba(x_test_scaled)[:, 1]

# 2. Rote Kringel absichern: Typen explizit für basedpyright konvertieren
y_test_arr = np.asarray(y_test, dtype=int)
y_scores_arr = np.asarray(y_scores, dtype=float)

# 3. Falsch-Positiv-Rate (fpr) und Richtig-Positiv-Rate (tpr) berechnen
fpr, tpr, _ = roc_curve(y_test_arr, y_scores_arr)
roc_auc = auc(fpr, tpr)

# 4. Figure und Axes initialisieren
fig, ax = plt.subplots(figsize=(8, 6))

# Farbdefinitionen aus deinem bestehenden Design
COLOR_ROC = "#2E8B57"  # Seegrün für dein optimales Modell
COLOR_BASE = "#B22222"  # Karmesinrot für die Zufallslinie

# 5. Die ROC-Kurve zeichnen
ax.plot(
    fpr, tpr, color=COLOR_ROC, linewidth=2.5, label=f"SVM Modell (AUC = {roc_auc:.4f})"
)

# Die diagonale Referenzlinie zeichnen (Zufallsklassifikator / "Münzwurf")
ax.plot(
    [0, 1],
    [0, 1],
    color=COLOR_BASE,
    linestyle="--",
    linewidth=1.5,
    label="Zufälliges Raten (AUC = 0.5000)",
)

# 6. Design und Achsenbeschriftungen verfeinern
ax.set_title(
    "ROC-Kurve (Receiver Operating Characteristic)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel(
    "Falsch-Positiv-Rate (False Positive Rate / Alpha)", fontsize=11, labelpad=10
)
ax.set_ylabel(
    "Richtig-Positiv-Rate (True Positive Rate / Sensitivität)", fontsize=11, labelpad=10
)

# Achsenbereiche leicht erweitern, damit die Kurve nicht am Rand klebt
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

# Gitter und Legende positionieren
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(
    loc="lower right", fontsize=10, frameon=True, facecolor="white", edgecolor="none"
)

# 7. Grafiken hochauflösend im passenden Projektordner speichern
target_folder = "images"
filename_base = f"{target_folder}/svm_roc_curve"

fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")

print(f"✓ ROC-Kurve erfolgreich gespeichert unter: {filename_base}.png/.pdf")

# fig.tight_layout()
# plt.show()

# Beispiel Anwendung
# 1. Wahrscheinlichkeiten mit den skalierten Daten berechnen
df["anomaly_probability"] = final_model.predict_proba(x_test_scaled)[:, 1]

# 2. Den jeweils allerletzten Log-Eintrag pro Maschine isolieren (Dein Code ab hier ist top!)
latest_factory_state = (
    df.sort_values("timestamp").groupby("machine_id").last().reset_index()
)

# Top 10 Risiko-Maschinen filtern
prioritization_table = latest_factory_state[
    ["machine_id", "anomaly_probability", "temperature", "vibration"]
]

prioritization_table = prioritization_table.sort_values(
    by="anomaly_probability", ascending=False
).head(10)  # Zeigt die Top 10

# 3. Maschinen nach dringlichkeit auflisten
print("=== 📋 REPRODUZIERBARE PRIORISIERUNGSLISTE FÜR DIE INSTANDHALTUNG ===")
for idx, row in prioritization_table.reset_index(drop=True).iterrows():
    status = "🚨 ALARM (Anomalie)" if row["anomaly_probability"] > 0.1 else "✅ Normal"
    print(
        f"Priorität {idx + 1:02d} | Maschine ID: {int(row['machine_id']):02d} | "
        f"Risiko: {row['anomaly_probability'] * 100:5.1f}% | Status: {status}"
    )
