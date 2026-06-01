# %% read the csv file
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -1. Stichproben, oder eben nicht
DOWNSAMPLE_SVM = None  # Voller Datensatz -> Bilder landen direkt im Bilder Ordner
# DOWNSAMPLE_SVM = 0.2  # 20 % Der sample verwenden, schneller zum testen und entwickeln -> Unterordner samples

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
param_grid = [
    {
        "kernel": ["rbf"],
        "C": [0.01, 0.1, 1, 10, 100, 1000, 10000],
        "gamma": [
            "scale",
            "auto",
        ],
    },
    {
        "kernel": ["poly"],
        "C": [0.001, 0.01, 0.1, 1],
        "degree": [3],
        "gamma": ["scale", "auto"],
    },
]

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
        class_weight="balanced",
        probability=True,
        cache_size=2000,
        random_state=42,
        max_iter=500000,
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
    "param_kernel",
    "param_degree",
    "mean_test_f1",
    "std_test_f1",
    "mean_test_precision",
    "std_test_precision",
    "mean_test_recall",
    "std_test_recall",
]
cv_results[columns_to_show]


# %%
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
    "param_kernel",
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
    "Kernel",
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

import matplotlib.pyplot as plt
import numpy as np

# 1. Daten standardisieren und Typen sichern
hyper_confusion_table["C"] = hyper_confusion_table["C"].astype(float)
hyper_confusion_table["Kernel"] = hyper_confusion_table["Kernel"].astype(str)
hyper_confusion_table["Gamma"] = hyper_confusion_table["Gamma"].astype(str)

available_kernels = hyper_confusion_table["Kernel"].unique().tolist()
gammas = ["scale", "auto"]

COLOR_FP = "#4682B4"  # Stahlblau
COLOR_FN = "#B22222"  # Karmesinrot
COLOR_CHOICE = "#2E8B57"  # Seegrün

# 2. Äußere Schleife: Erzeugt pro Kernel-Typ ein eigenes Bild
for kernel_val in available_kernels:
    df_kernel = hyper_confusion_table.query("Kernel == @kernel_val")
    valid_gammas = [g for g in gammas if not df_kernel.query("Gamma == @g").empty]
    num_plots = len(valid_gammas)

    if num_plots == 0:
        continue

    fig, axes = plt.subplots(
        nrows=1, ncols=num_plots, figsize=(6 * num_plots, 5.5), sharey=False
    )

    if num_plots == 1:
        axes = [axes]

    # --- 🔍 GLOBALEN Y-ZOOM PRO KERNEL BERECHNEN ---
    # Wir ermitteln die Min/Max-Werte über ALLE Gammas dieses Kernels hinweg,
    # damit die linke und rechte Achse innerhalb des Bildes vergleichbar bleiben.
    global_fp = df_kernel["FP (Falscher Alarm)"].values
    global_fn = df_kernel["FN (Verpasst)"].values

    # Dynamische Grenzen für FP (Linke Achse)
    fp_min, fp_max = float(np.min(global_fp)), float(np.max(global_fp))
    fp_puffer = (fp_max - fp_min) * 0.15 if (fp_max - fp_min) > 0 else 10.0
    dynamic_fp_min = max(0.0, fp_min - fp_puffer)
    dynamic_fp_max = fp_max + fp_puffer

    # Dynamische Grenzen für FN (Rechte Achse)
    fn_min, fn_max = float(np.min(global_fn)), float(np.max(global_fn))
    fn_puffer = (fn_max - fn_min) * 0.15 if (fn_max - fn_min) > 0 else 2.0
    dynamic_fn_min = max(0.0, fn_min - fn_puffer)
    dynamic_fn_max = fn_max + fn_puffer
    # -----------------------------------------------

    # 3. Innere Schleife: Über die Gamma-Kanäle iterieren
    for ax_left, gamma_val in zip(axes, valid_gammas):
        df_filtered = df_kernel.query("Gamma == @gamma_val")

        df_plot = (
            df_filtered.groupby("C", as_index=False)
            .agg(
                {
                    "FP (Falscher Alarm)": "mean",
                    "FN (Verpasst)": "mean",
                    "F1-Score": "max",
                }
            )
            .sort_values(by="C")
        )

        c_values = df_plot["C"].values
        fp_values = df_plot["FP (Falscher Alarm)"].values
        fn_values = df_plot["FN (Verpasst)"].values

        # --- 🔀 TWINX ACHSE ---
        ax_right = ax_left.twinx()

        # --- 📈 VERLÄUFE ZEICHNEN ---
        line1 = ax_left.plot(
            c_values,
            fp_values,
            marker="o",
            linewidth=2.5,
            color=COLOR_FP,
            label="Falsche Alarme (FP)",
        )
        ax_left.fill_between(c_values, fp_values, color=COLOR_FP, alpha=0.07)

        line2 = ax_right.plot(
            c_values,
            fn_values,
            marker="s",
            linewidth=3.0,
            color=COLOR_FN,
            label="Verpasste Anomalien (FN)",
        )

        # --- 🎯 VISUELLER ANKER (Nur bei RBF zeichnen) ---
        if kernel_val == "rbf":
            ax_left.axvline(
                10.0, color=COLOR_CHOICE, linestyle="--", alpha=0.8, linewidth=1.5
            )

        # --- 🏷️ TEXT-LABELS MIT DYNAMISCHEM OFFSET ---
        text_offset_fp = (dynamic_fp_max - dynamic_fp_min) * 0.03
        text_offset_fn = (dynamic_fn_max - dynamic_fn_min) * 0.04

        for i, c_val in enumerate(c_values):
            fp = fp_values[i]
            fn = fn_values[i]

            # FP-Label
            ax_left.text(
                c_val,
                fp + text_offset_fp,
                f"{fp:.0f}",
                color=COLOR_FP,
                fontsize=9,
                ha="center",
                fontweight="bold",
            )

            # FN-Label
            if fn > 0 or (kernel_val == "rbf" and c_val == 10.0):
                ax_right.text(
                    c_val,
                    fn + text_offset_fn,
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

        # --- 🛠️ STYLING LINKS (FP) ---
        ax_left.set_xscale("log")
        ax_left.set_xlabel("Hyperparameter C (logarithmisch)", fontsize=11)
        ax_left.set_ylabel(
            "Falsche Alarme (FP)", color=COLOR_FP, fontsize=11, fontweight="bold"
        )
        ax_left.tick_params(axis="y", labelcolor=COLOR_FP)

        # ANWENDUNG DYNAMISCHER ZOOM LINKS
        ax_left.set_ylim(dynamic_fp_min, dynamic_fp_max)
        ax_left.grid(True, which="both", linestyle="--", alpha=0.3)

        # --- 🛠️ STYLING RECHTS (FN) ---
        ax_right.set_ylabel(
            "Verpasste Anomalien (FN)", color=COLOR_FN, fontsize=11, fontweight="bold"
        )
        ax_right.tick_params(axis="y", labelcolor=COLOR_FN)

        # ANWENDUNG DYNAMISCHER ZOOM RECHTS
        ax_right.set_ylim(dynamic_fn_min, dynamic_fn_max)

        # Titel für den jeweiligen Subplot
        ax_left.set_title(
            f"Kernel: {kernel_val.upper()} | gamma='{gamma_val}'",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        # Kombinierte Legende
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

    fig.subplots_adjust(wspace=0.45)
    fig.tight_layout()

    # 💾 BILD SPEICHERN
    target_folder = SAMPLE_FOLDER if DOWNSAMPLE_SVM else IMAGE_FOLDER
    filename_base = f"{target_folder}/svm_error_tradeoff_{kernel_val}"
    fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")
    print(f"✓ Grafik für Kernel '{kernel_val}' erfolgreich gespeichert.")

    plt.show()

# %%
# 1. Daten vorbereiten und Typen sichern
cv_results["param_C"] = cv_results["param_C"].astype(float)
cv_results["param_kernel"] = cv_results["param_kernel"].astype(str)
cv_results["param_gamma"] = cv_results["param_gamma"].astype(str)

available_kernels = cv_results["param_kernel"].unique().tolist()
gammas = ["scale", "auto"]

# Äußere Schleife: Erzeugt pro Kernel ein komplett eigenes Bild
for kernel_val in available_kernels:
    # Filtere alle Daten für den aktuellen Kernel
    df_kernel = cv_results.query("param_kernel == @kernel_val")

    valid_gammas = [g for g in gammas if not df_kernel.query("param_gamma == @g").empty]
    num_plots = len(valid_gammas)

    if num_plots == 0:
        continue

    # 2. Subplots initialisieren
    fig, axes = plt.subplots(
        nrows=1, ncols=num_plots, figsize=(6 * num_plots, 5), sharey=True
    )

    if num_plots == 1:
        axes = [axes]

    # --- 🔍 DYNAMISCHEN Y-ZOOM BERECHNEN ---
    # Wir suchen die absoluten Min/Max-Werte des F1-Scores für DIESEN Kernel heraus
    if "mean_test_f1" in df_kernel.columns:
        global_mean_col = "mean_test_f1"
    else:
        global_mean_col = "mean_test_score"

    y_min_data = float(df_kernel[global_mean_col].min())
    y_max_data = float(df_kernel[global_mean_col].max())

    # Dynamischer Puffer von 0.05 (1%-Punkte) nach oben und unten
    # max(0.0, ...) fängt negative Skalen ab, min(1.01, ...) verhindert zu weites Hinausschießen oben
    dynamic_y_min = max(0.0, y_min_data - 0.01)
    dynamic_y_max = min(1.01, y_max_data + 0.01)
    # ---------------------------------------

    # Innere Schleife: Über die Gamma-Werte iterieren
    for ax, gamma_val in zip(axes, valid_gammas):
        df_filtered = df_kernel.query("param_gamma == @gamma_val")

        if "mean_test_f1" in df_filtered.columns:
            mean_col, std_col = "mean_test_f1", "std_test_f1"
        else:
            mean_col, std_col = "mean_test_score", "std_test_score"

        df_plot = (
            df_filtered.groupby("param_C", as_index=False)
            .agg({mean_col: "max", std_col: "mean"})
            .sort_values("param_C")
        )

        c_values = df_plot["param_C"].values
        f1_values = df_plot[mean_col].values
        std_values = df_plot[std_col].values

        log_C = np.log10(c_values)

        # --- 📐 MATHEMATISCHE ELLBOGEN-METHODE ---
        x1, y1 = log_C[0], f1_values[0]
        x2, y2 = log_C[-1], f1_values[-1]

        # Sekante einzeichnen
        ax.plot(
            [c_values[0], c_values[-1]],
            [y1, y2],
            color="gray",
            linestyle=":",
            alpha=0.7,
            label="Sekante (Basis-Trend)",
        )

        normalisator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances = []
        for i in range(len(log_C)):
            x0, y0 = log_C[i], f1_values[i]
            zaehler = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            distances.append(zaehler / normalisator if normalisator > 0 else 0.0)

        elbow_idx = np.argmax(distances)
        elbow_C = c_values[elbow_idx]
        elbow_score = f1_values[elbow_idx]
        # ------------------------------------------------------------------------

        # F1-Kurve plotten
        ax.plot(
            c_values,
            f1_values,
            marker="o",
            linestyle="-",
            linewidth=2,
            color="#1f77b4",
            label="Mittlerer F1-Score",
        )

        # Standardabweichungs-Korridor
        ax.fill_between(
            c_values,
            f1_values - std_values,
            f1_values + std_values,
            color="#1f77b4",
            alpha=0.15,
        )

        # Das gewählte Ellbogen-Optimum markieren
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

        # Textlabel für den Ellbogen (Nutzt nun einen kleineren Offset, da wir herangezoomed sind)
        text_offset = (dynamic_y_max - dynamic_y_min) * 0.08
        ax.text(
            elbow_C,
            elbow_score - text_offset,
            f"C={elbow_C}\nF1={elbow_score:.3f}",
            color="green",
            fontweight="bold",
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="green",
                alpha=0.8,
            ),
        )

        # Achsen-Konfiguration mit dem neuen dynamischen Zoom
        ax.set_xscale("log")
        ax.set_xlabel("Hyperparameter C (log)", fontsize=11)
        ax.set_ylabel("F1-Score", fontsize=11)
        ax.set_title(
            f"Kernel: {kernel_val.upper()} | gamma='{gamma_val}'",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        # HIER WIRD DER ZUGEWIESENE DYNAMISCHE ZOOM ANGEWANDT
        ax.set_ylim(dynamic_y_min, dynamic_y_max)

        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)

    fig.subplots_adjust(wspace=0.3)
    fig.tight_layout()

    # 💾 HOCHAUFLÖSENDES SPEICHERN
    target_folder = SAMPLE_FOLDER if DOWNSAMPLE_SVM else IMAGE_FOLDER
    filename_base = f"{target_folder}/f1_performance_elbow_{kernel_val}"

    fig.savefig(
        f"{filename_base}.png",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")
    print(f"✓ Grafiken für '{kernel_val}' erfolgreich gespeichert.")

    plt.show()

# %%
# 1. Sicherstellen, dass wir das richtige DataFrame nutzen und Typen sichern
df_heatmap_master = cv_results.copy()
df_heatmap_master["param_kernel"] = df_heatmap_master["param_kernel"].astype(str)
df_heatmap_master["param_gamma"] = df_heatmap_master["param_gamma"].astype(str)

# Spaltennamen für den Score dynamisch ermitteln
if "mean_test_f1" in df_heatmap_master.columns:
    score_col = "mean_test_f1"
elif "mean_test_score" in df_heatmap_master.columns:
    score_col = "mean_test_score"
else:
    raise KeyError("Keine passenden Score-Spalten im DataFrame gefunden!")

# Verfügbare Kernels im aktuellen Suchraum ermitteln
available_kernels = df_heatmap_master["param_kernel"].unique().tolist()

# 2. Äußere Schleife: Generiert pro Kernel eine eigene Heatmap
for kernel_val in available_kernels:
    # Filter für den aktuellen Kernel setzen
    df_kernel = df_heatmap_master.query("param_kernel == @kernel_val").copy()

    # WICHTIG: Falls 'degree' oder andere Parameter Duplikate erzeugen, aggregieren wir nach
    # param_C und param_gamma und picken uns den mathematisch besten F1-Score heraus.
    df_grouped = df_kernel.groupby(["param_C", "param_gamma"], as_index=False).agg(
        {score_col: "max"}
    )

    # Parameter-Typen erst NACH der Aggregation für eine saubere Achsenbeschriftung formatieren
    df_grouped["param_C_str"] = (
        df_grouped["param_C"].astype(float).map(lambda x: f"{x:g}")
    )

    # 3. Matrix für die Heatmap via Pivot-Tabelle fehlerfrei erstellen
    scores_matrix = df_grouped.pivot(
        index="param_C_str", columns="param_gamma", values=score_col
    )

    # WICHTIG: Da die Index-Keys Strings sind, reindizieren wir sie nach ihrem echten mathematischen Wert
    # Das verhindert, dass '10' vor '2' sortiert wird (alphabetisches Sortierungsproblem)
    sorted_index_keys = sorted(scores_matrix.index, key=float)
    scores_matrix = scores_matrix.reindex(index=sorted_index_keys)

    # 4. Figure und Axes initialisieren
    fig, ax = plt.subplots(figsize=(8, 6))

    # 5. Heatmap zeichnen
    sns.heatmap(
        scores_matrix,
        annot=True,  # Zahlen in den Kästchen anzeigen
        fmt=".4f",  # 4 Nachkommastellen wie im Original
        cmap="viridis",  # Gelb = exzellent, Blau = suboptimal
        linewidths=0.5,  # Trennlinien zwischen den Kästchen
        linecolor="white",
        cbar_kws={"label": "Mittlerer F1-Score"},
        ax=ax,
    )

    # 6. Design und Beschriftungen verfeinern (Inklusive dynamischem Kernel-Namen im Titel)
    ax.set_title(
        f"GridSearchCV Hyperparameter-Analyse ({kernel_val.upper()}-Kernel)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Hyperparameter: Gamma", fontsize=11, labelpad=10)
    ax.set_ylabel("Hyperparameter: C", fontsize=11, labelpad=10)

    # 7. Grafik hochauflösend und getrennt nach Kernel speichern
    target_folder = SAMPLE_FOLDER if DOWNSAMPLE_SVM else IMAGE_FOLDER
    filename_base = f"{target_folder}/svm_hyperparameter_heatmap_{kernel_val}"

    fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")

    print(
        f"✓ Heatmap für '{kernel_val}' erfolgreich gespeichert unter: {filename_base}.png/.pdf"
    )

    fig.tight_layout()
    plt.show()
