# %% read the csv file
import os

import matplotlib.pyplot as plt
import pandas as pd

# -1. Stichproben, oder eben nicht
DOWNSAMPLE_SVM = None  # Voller Datensatz -> Bilder landen direkt im Bilder Ordner
# DOWNSAMPLE_SVM = 0.1  # 10 % Der sample verwenden, schneller zum testen und entwickeln -> Unterordner samples

# 0. Ordner erstellen
IMAGE_FOLDER = "images"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

SAMPLE_FOLDER = f"{IMAGE_FOLDER}/samples"
if not os.path.exists(SAMPLE_FOLDER):
    os.makedirs(SAMPLE_FOLDER)

# 1. dataframe erstellen
df = pd.read_csv("data/raw/smart-manufacturing-iot-cloud-monitoring-dataset.csv")

# 2. Metadaten formatieren
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)

# 3. WINSORIZING (alles über 3 Sigma Standardabweichung auf den Wert ebenda drücken)
for col in ["vibration", "temperature"]:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = df[col].clip(lower=mean - 3 * std, upper=mean + 3 * std)

# 4. Define a central feature list
x_feature_sensors = [
    "temperature",
    "vibration",
    # "humidity",
    # "pressure",
    # "energy_consumption",
]

# AUSGABE DES TRAININGS MIT auschließlich den sensordaten von "humidity", "pressure", "energy_consumption",
# NICHT brauchbar für das modell => NICHT hilfreich alle 5 zu nehmen => Daher NUR 2: "temperature", "vibration"
# Sample size: 20000
# Starte paralleles Hyperparameter-Tuning ohne Data Leakage...
# ✓ Optimales Parameter-Set: {'C': 100, 'gamma': 'auto'}
# === 🚀 ERGEBNISSE: OPTIMIERTE ANOMALIE-KLASSIFIKATION (Tuned SVC) ===
#               precision    recall  f1-score   support
#            0       0.91      0.49      0.63      3639
#            1       0.09      0.50      0.15       361
#     accuracy                           0.49      4000
#    macro avg       0.50      0.49      0.39      4000
# weighted avg       0.83      0.49      0.59      4000
# === 📊 GRID SEARCH DETAIL-ERGEBNISSE ===


# %% run the svm classifier and perform a gridsearch to find optimize hyperparameters
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Take a sample
if DOWNSAMPLE_SVM:
    sample_size = int(len(df.index) * DOWNSAMPLE_SVM)
else:
    sample_size = len(df.index)

print(f"Sample size: {sample_size}")
df_sample = df.sample(n=sample_size, random_state=42)

x_sample = df_sample[x_feature_sensors]
y_sample = df_sample["anomaly_flag"]

# 2. train test split (before scaling)
x_train_sample, x_test_sample, y_train_sample, y_test_sample = train_test_split(
    x_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
)

# 3. SAUBERE SKALIERUNG: Verhindert Data Leakage
scaler = StandardScaler()
# fit_transform NUR auf Trainingsdaten
x_train_sample_scaled = scaler.fit_transform(x_train_sample)
# NUR transform auf Testdaten
x_test_sample_scaled = scaler.transform(x_test_sample)

# 4. Erweitertes Parameter-Grid für die RBF-SVM
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100, 1000, 10000],
    "gamma": [
        "scale",
        "auto",
    ],
}

# 5. Scoring Metriken definieren
scoring_metrics = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}

# 6. GridSearchCV initialisieren
print("Starte paralleles Hyperparameter-Tuning ohne Data Leakage...")
grid_search = GridSearchCV(
    estimator=SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        cache_size=2000,
        random_state=42,
    ),
    param_grid=param_grid,
    cv=3,
    scoring=scoring_metrics,
    refit="f1",  # type: ignore
    n_jobs=-1,
)

# Tuning mit skalierten Trainingsdaten ausführen
grid_search.fit(x_train_sample_scaled, y_train_sample)

# 6. Bestes Modell testen
y_pred = grid_search.best_estimator_.predict(x_test_sample_scaled)

print(f"\n✓ Optimales Parameter-Set: {grid_search.best_params_}")
print("\n=== 🚀 ERGEBNISSE: OPTIMIERTE ANOMALIE-KLASSIFIKATION (Tuned SVC) ===")
print(classification_report(y_test_sample, y_pred))
print("\n=== 📊 GRID SEARCH DETAIL-ERGEBNISSE ===")
# Grid-Ergebnisse in ein DataFrame laden
cv_results = pd.DataFrame(grid_search.cv_results_)

# %% ee

# Nur die relevanten Spalten für die Übersicht auswählen
columns_to_show = [
    "param_C",
    "param_gamma",
    "mean_test_f1",
    "std_test_f1",
    "mean_test_precision",
    "std_test_precision",
    "mean_test_recall",
    "std_test_recall",
]
grid_summary = cv_results[columns_to_show]

grid_summary


# %%

import numpy as np
import pandas as pd

# 1. Bestimme die exakte Anzahl an Normalen und Anomalien in den Trainingsdaten
# (Da das die Basis für die Kreuzvalidierungs-Splits ist)
P_total = np.sum(y_train_sample == 1)  # Alle echten Anomalien (Positives)
N_total = np.sum(y_train_sample == 0)  # Alle echten Normalfälle (Negatives)

# Da wir cv=3 nutzen, berechnen wir die durchschnittliche Anzahl pro Validierungs-Fold
P = P_total / 3
N = N_total / 3

# 2. Ergebnisse des Multi-Metric-GridSearch laden
cv_results = pd.DataFrame(grid_search.cv_results_)

# 3. Mathematische Rekonstruktion der Konfusionsmatrix-Komponenten
# Recall = TP / P  =>  TP = Recall * P
cv_results["Anzahl_TP"] = (cv_results["mean_test_recall"] * P).round(1)

# FN = P - TP
cv_results["Anzahl_FN"] = (P - cv_results["Anzahl_TP"]).round(1)

# Precision = TP / (TP + FP)  =>  FP = (TP / Precision) - TP
# (Vermeidung von Division durch Null, falls Precision mal 0 ist)
cv_results["Anzahl_FP"] = (
    np.where(
        cv_results["mean_test_precision"] > 0,
        (cv_results["Anzahl_TP"] / cv_results["mean_test_precision"])
        - cv_results["Anzahl_TP"],
        0,
    )
).round(1)

# TN = N - FP
cv_results["Anzahl_TN"] = (N - cv_results["Anzahl_FP"]).round(1)

# 4. Spalten für die finale Übersicht auswählen und sortieren
matrix_columns = [
    "param_C",
    "param_gamma",
    "mean_test_f1",
    "Anzahl_TN",  # Richtig Normal
    "Anzahl_FP",  # Falscher Alarm
    "Anzahl_FN",  # Verpasste Anomalie
    "Anzahl_TP",  # Richtig Anomalie
]

hyper_confusion_table = cv_results[matrix_columns].sort_values(
    by="mean_test_f1", ascending=False
)

# 5. Schicke Ausgabe im Terminal
print("\n=== 🔮 DIE REKONSTRUIERTE HYPER-KONFUSIONS-MATRIX PRO VALIDIERUNGS-FOLD ===")
print(
    f"(Basis pro Validierungs-Set: Ca. {N_total / 3:.1f} Normalfälle & {P_total / 3:.1f} Anomalien)\n"
)

# Spaltennamen für die Anzeige lesbarer machen
hyper_confusion_table.columns = [
    "C",
    "Gamma",
    "F1-Score",
    "TN (Richtig Normal)",
    "FP (Falscher Alarm)",
    "FN (Verpasst)",
    "TP (Erkannt)",
]

print(hyper_confusion_table.to_string(index=False))
# %%

# 1. Daten vorbereiten (C numerisch machen)
hyper_confusion_table["C"] = hyper_confusion_table["C"].astype(float)

valid_gammas = [
    g for g in ["scale", "auto"] if not hyper_confusion_table.query("Gamma == @g").empty
]
num_plots = len(valid_gammas)

# Subplots nebeneinander erzeugen
fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(6 * num_plots, 5.5))

if num_plots == 1:
    axes = [axes]

COLOR_FP = "#4682B4"  # Stahlblau
COLOR_FN = "#B22222"  # Karmesinrot
COLOR_CHOICE = "#2E8B57"  # Seegrün

# Schleife über die Gamma-Kanäle
for ax_left, gamma_val in zip(axes, valid_gammas):
    df_plot = (
        hyper_confusion_table.query("Gamma == @gamma_val").sort_values(by="C").copy()
    )

    c_values = df_plot["C"].values
    fp_values = df_plot["FP (Falscher Alarm)"].values
    fn_values = df_plot["FN (Verpasst)"].values

    # --- 🔀 ERSTELLEN DER ZWEITEN Y-ACHSE ---
    ax_right = ax_left.twinx()

    # --- 📈 VERLÄUFE ZEICHNEN ---
    # Linke Achse: Falsche Alarme (Stahlblau)
    line1 = ax_left.plot(
        c_values,
        fp_values,
        marker="o",
        linewidth=2.5,
        color=COLOR_FP,
        label="Falsche Alarme (FP)",
    )
    ax_left.fill_between(c_values, fp_values, color=COLOR_FP, alpha=0.07)

    # Rechte Achse: Verpasste Anomalien (Karmesinrot)
    line2 = ax_right.plot(
        c_values,
        fn_values,
        marker="s",
        linewidth=3.0,
        color=COLOR_FN,
        label="Verpasste Anomalien (FN)",
    )

    # --- 🎯 DER VISUELLE ANKER (C=10 Trennlinie) ---
    ax_left.axvline(10.0, color=COLOR_CHOICE, linestyle="--", alpha=0.8, linewidth=1.5)

    # --- 🏷️ TEXT-LABELS AN DEN PUNKTEN ---
    for i, c_val in enumerate(c_values):
        fp = fp_values[i]
        fn = fn_values[i]

        # FP-Labels nutzen das Koordinatensystem der linken Achse
        ax_left.text(
            c_val,
            fp + 12,
            f"{fp:.1f}",
            color=COLOR_FP,
            fontsize=9,
            ha="center",
            fontweight="bold",
        )

        # FN-Labels nutzen das Koordinatensystem der rechten Achse
        if fn > 0 or c_val == 10.0:
            ax_right.text(
                c_val,
                fn + 0.08,
                f"FN: {fn:.1f}",
                color=COLOR_FN,
                fontsize=9,
                ha="center",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor=COLOR_FN,
                    alpha=0.9,
                    linewidth=1,
                ),
            )

    # --- 🛠️ ACHSEN-DESIGN LINKS (FP) ---
    ax_left.set_xscale("log")
    ax_left.set_xlabel("Hyperparameter C (logarithmisch)", fontsize=11)
    ax_left.set_ylabel(
        "Falsche Alarme (FP)", color=COLOR_FP, fontsize=11, fontweight="bold"
    )
    ax_left.tick_params(axis="y", labelcolor=COLOR_FP)
    ax_left.set_ylim(-15, max(fp_values) * 1.15)
    ax_left.grid(True, which="both", linestyle="--", alpha=0.3)

    # --- 🛠️ ACHSEN-DESIGN RECHTS (FN) ---
    ax_right.set_ylabel(
        "Verpasste Anomalien (FN)", color=COLOR_FN, fontsize=11, fontweight="bold"
    )
    ax_right.tick_params(axis="y", labelcolor=COLOR_FN)
    ax_right.set_ylim(
        -0.1, max(fn_values) * 1.25
    )  # Feine Skalierung speziell für die kleinen FN-Werte

    # Titel setzen
    ax_left.set_title(
        f"Fehlertyp-Abwägung für gamma='{gamma_val}'",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # Gemeinsame Legende für beide Achsen bauen
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_left.legend(
        lines,
        labels,
        loc="upper center",
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="none",
    )

# Horizontalen Abstand anpassen, damit die rechten Achsenbeschriftungen Platz haben
fig.subplots_adjust(wspace=0.45)

# 💾 ALS VEKTORGRAFIK UND PNG SPEICHERN
if DOWNSAMPLE_SVM:
    filename_base = f"{SAMPLE_FOLDER}/svm_error_tradeoff_twinx"
else:
    filename_base = f"{IMAGE_FOLDER}/svm_error_tradeoff_twinx"

fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")

fig.tight_layout()
plt.show()

# %%

# 1. Daten vorbereiten (C numerisch machen)
cv_results["param_C"] = cv_results["param_C"].astype(float)

# Gültige Gamma-Werte ermitteln
valid_gammas = [
    g for g in param_grid["gamma"] if not cv_results.query("param_gamma == @g").empty
]
num_plots = len(valid_gammas)

# 2. Subplots erstellen
fig, axes = plt.subplots(
    nrows=1, ncols=num_plots, figsize=(6 * num_plots, 5), sharey=True
)

if num_plots == 1:
    axes = [axes]

# Schleife über die Gamma-Werte
for ax, gamma_val in zip(axes, valid_gammas):
    subset = cv_results.query("param_gamma == @gamma_val").sort_values("param_C").copy()

    if "mean_test_f1" in subset.columns:
        mean_col, std_col = "mean_test_f1", "std_test_f1"
    else:
        mean_col, std_col = "mean_test_score", "std_test_score"

    subset[mean_col] = subset[mean_col].astype(float)
    subset[std_col] = subset[std_col].astype(float)

    # X-Werte im log10-Raum für die lineare Distanzberechnung der Sekante
    log_C = np.log10(subset["param_C"].values)
    f1_values = subset[mean_col].values

    # --- 📐 MATHEMATISCHE ELLBOGEN-METHODE (Zukunftssicher ohne np.cross) ---
    # Start- und Endpunkt der Kurve greifen
    x1, y1 = log_C[0], f1_values[0]
    x2, y2 = log_C[-1], f1_values[-1]

    # Sekante (Referenzlinie) im Plot einzeichnen
    ax.plot(
        [subset["param_C"].iloc[0], subset["param_C"].iloc[-1]],
        [y1, y2],
        color="gray",
        linestyle=":",
        alpha=0.7,
        label="Sekante (Basis-Trend)",
    )

    # Distanz jedes Punktes zur Sekante im 2D-Raum berechnen
    # Formel: |(y2-y1)x0 - (x2-x1)y0 + x2*y1 - y2*x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
    normalisator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    distances = []
    for i in range(len(log_C)):
        x0, y0 = log_C[i], f1_values[i]
        zaehler = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        distances.append(zaehler / normalisator)

    # Index des Punktes mit der maximalen Distanz zur Geraden
    elbow_idx = np.argmax(distances)

    elbow_C = subset["param_C"].iloc[elbow_idx]
    elbow_score = subset[mean_col].iloc[elbow_idx]
    # ------------------------------------------------------------------------

    # F1-Kurve plotten
    ax.plot(
        subset["param_C"],
        subset[mean_col],
        marker="o",
        linestyle="-",
        linewidth=2,
        color="#1f77b4",
        label="Mittlerer F1-Score",
    )

    # Standardabweichungs-Korridor
    ax.fill_between(
        subset["param_C"],
        subset[mean_col] - subset[std_col],
        subset[mean_col] + subset[std_col],
        color="#1f77b4",
        alpha=0.15,
    )

    # Das gewählte Ellbogen-Optimum markieren (Grüner Kreis)
    ax.plot(
        elbow_C,
        elbow_score,
        marker="o",
        color="green",
        markersize=12,
        markeredgewidth=2,
        markerfacecolor="none",
        linestyle="None",
        label=f"Ellbogen-Optimum (C={elbow_C})",
    )

    # Textlabel für den Ellbogen
    ax.text(
        elbow_C,
        elbow_score - 0.04,
        f"C={elbow_C}\nF1={elbow_score:.3f}",
        color="green",
        fontweight="bold",
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.8
        ),
    )

    # Achsen-Konfiguration
    ax.set_xscale("log")
    ax.set_xlabel("Hyperparameter C (log)", fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title(f"gamma = '{gamma_val}'", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0.6, 1.05)

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

# Abstand zwischen den Plots einstellen
fig.subplots_adjust(wspace=0.3)
fig.tight_layout()

# ========================================================================
# 💾 HOCHAUFLÖSENDES SPEICHERN (Direkt vor plt.show() einfügen)
# ========================================================================
# Dateiname dynamisch generieren, damit kein Plot überschrieben wird
if DOWNSAMPLE_SVM:
    filename_base = f"{SAMPLE_FOLDER}/f1_performance_elbow_gamma_comparison"
else:
    filename_base = f"{IMAGE_FOLDER}/f1_performance_elbow_gamma_comparison"

# 1. Als hochauflösendes PNG für Office/Web speichern
fig.savefig(
    f"{filename_base}.png",
    dpi=300,  # 300 DPI ist Druckqualität
    bbox_inches="tight",  # Verhindert, dass abgeschnittene Achsenbeschriftungen entstehen
    transparent=False,  # Weißer Hintergrund (besser für Dokumente)
    facecolor="white",
)

# 2. Als verlustfreie PDF-Vektorgrafik für maximale Zoom-Qualität speichern
fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")

print(f"Grafiken erfolgreich gespeichert als: {filename_base}.png/.pdf")
# ========================================================================

# Plot anzeigen
plt.show()


# # %%
# import seaborn as sns

# # Ergebnisse in eine Matrix-Form bringen
# scores_matrix = cv_results.pivot(
#     index="param_C", columns="param_gamma", values="mean_test_score"
# )

# fig, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(scores_matrix, annot=True, fmt=".4f", cmap="viridis", ax=ax)
# ax.set_title("GridSearchCV F1-Scores Heatmap")
# ax.set_xlabel("Gamma")
# ax.set_ylabel("C")
# plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Sicherstellen, dass wir das richtige DataFrame nutzen (hier 'results')
# Wir erstellen eine Kopie, um die Formatierung für den Plot anzupassen
df_heatmap = cv_results.copy()

# 2. Spaltennamen für den Score dynamisch ermitteln (F1-Score bevorzugt)
if "mean_test_f1" in df_heatmap.columns:
    score_col = "mean_test_f1"
elif "mean_test_score" in df_heatmap.columns:
    score_col = "mean_test_score"
else:
    raise KeyError("Keine passenden Score-Spalten im DataFrame gefunden!")

# 3. Parameter-Typen für eine saubere Achsenbeschriftung formatieren
# Wenn C ein Float bleibt, formatieren wir es schön als String (z.B. 10000.00 -> '10000')
df_heatmap["param_C"] = df_heatmap["param_C"].astype(float).map(lambda x: f"{x:g}")
df_heatmap["param_gamma"] = df_heatmap["param_gamma"].astype(str)

# 4. Matrix für die Heatmap via Pivot-Tabelle erstellen
scores_matrix = df_heatmap.pivot(
    index="param_C", columns="param_gamma", values=score_col
)

# WICHTIG: Da C jetzt ein String ist, müssen wir die Zeilen manuell nach dem
# mathematischen Wert sortieren, sonst ordnet Pandas sie alphabetisch an ('10', '100', '10000', '2')
scores_matrix = scores_matrix.reindex(index=sorted(scores_matrix.index, key=float))

# 5. Figure und Axes initialisieren
fig, ax = plt.subplots(figsize=(8, 6))

# 6. Heatmap zeichnen
# Wir nutzen das professionelle 'viridis'-Thema oder alternativ 'YlGnBu'
sns.heatmap(
    scores_matrix,
    annot=True,  # Zahlen in den Kästchen anzeigen
    fmt=".4f",  # 4 Nachkommastellen wie in deinem Original
    cmap="viridis",  # Harmonische Farbskala (gelb=super, blau=schlecht)
    linewidths=0.5,  # Dezente Trennlinien zwischen den Kästchen
    linecolor="white",
    cbar_kws={"label": "Mittlerer F1-Score"},  # Beschriftung der Farblegende
    ax=ax,
)

# 7. Design und Beschriftungen verfeinern
ax.set_title(
    "GridSearchCV Hyperparameter-Analyse (F1-Score)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Hyperparameter: Gamma", fontsize=11, labelpad=10)
ax.set_ylabel("Hyperparameter: C", fontsize=11, labelpad=10)

# 8. Grafik hochauflösend speichern (PNG + PDF)
if DOWNSAMPLE_SVM:
    filename_base = f"{SAMPLE_FOLDER}/svm_hyperparameter_heatmap"
else:
    filename_base = f"{IMAGE_FOLDER}/svm_hyperparameter_heatmap"

fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")

fig.tight_layout()
plt.show()


# %% Beispiel Anwendung
# 1. WICHTIG: Die Daten exakt wie im Training mit dem existierenden Scaler transformieren
x_entire_scaled = scaler.transform(df[x_feature_sensors])

# 2. Wahrscheinlichkeiten mit den skalierten Daten berechnen
df["anomaly_probability"] = grid_search.best_estimator_.predict_proba(x_entire_scaled)[
    :, 1
]

# 3. Den jeweils allerletzten Log-Eintrag pro Maschine isolieren (Dein Code ab hier ist top!)
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

print("=== 📋 REPRODUZIERBARE PRIORISIERUNGSLISTE FÜR DIE INSTANDHALTUNG ===")
for idx, row in prioritization_table.reset_index(drop=True).iterrows():
    status = "🚨 ALARM (Anomalie)" if row["anomaly_probability"] > 0.1 else "✅ Normal"
    print(
        f"Priorität {idx + 1:02d} | Maschine ID: {int(row['machine_id']):02d} | "
        f"Risiko: {row['anomaly_probability'] * 100:5.1f}% | Status: {status}"
    )
