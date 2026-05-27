# %% impots
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% dataframe erstellen
df_eda = pd.read_csv("data/raw/smart-manufacturing-iot-cloud-monitoring-dataset.csv")
df_eda.head()

# %% namen aller spalten
df_eda.columns.astype(str).to_list()

# %% Gibt es fehlende Werte?
if df_eda.isna().values.any():
    print(
        "Ja => Wir müssen die fehlenen Werte behandeln, zum Beispiel den Mittelwert als Ersatzwert nehmen"
    )
else:
    print(
        "No => Wir müssen die SVM mit class_weight='balanced' trainieren, oder durch downsampling oder doppelung das ungleichgewicht ausgleichen"
    )

# %% Diskreptanz Statistik
df_eda.describe().T

# %% global correlation map
# Relevante Spalten für die Gesamtmatrix auswählen
# Wir nehmen die Sensoren, die Metadaten, die Flags und unser potenziellen target auf
correlation_cols = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption",
    "anomaly_flag",
    "predicted_remaining_life",
    "downtime_risk",
    "maintenance_required",
]

# 1. Die Korrelationsmatrix berechnen
corr_matrix = df_eda.loc[:, correlation_cols].corr()

# 3. Plot-Objekte im OO-Stil erstellen
fig, ax = plt.subplots(figsize=(10, 8))

# 4. Heatmap zeichnen
sns.heatmap(
    corr_matrix,
    annot=True,  # Zahlenwerte in die Boxen schreiben
    fmt=".3f",  # 3 Nachkommastellen für die winzigen Werte
    cmap="coolwarm",  # Blau (negativ), Weiß (null), Rot (positiv)
    vmin=-1,
    vmax=1,  # Skala von -1 bis +1 festlegen
    square=True,  # Quadratische Boxen
    linewidths=0.5,  # Dünne Trennlinien
    linecolor="lightgray",
    ax=ax,
)

# 5. OO-Layout anpassen
ax.set_title("Globale Korrelation als Heatmap", fontsize=16, pad=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

fig.tight_layout()
plt.show()

# %%
print(
    "Aus der Heatmap ist zu sehen, dass es eine potentiell brauchbare Korrelationen für die Sensoren 'temperature' und 'vibration' gibt."
)
print(
    "Die Sensoren 'humidity' 'pressure' und 'energy_consumption' zeigen keine korrelationen."
)
print(
    "Es gibt eine bemerkenswerte 1 zu 1 Korrelation zwischen dem 'anomaly_flag' und dem 'downtime_risk'. Hier muss aussortiert werden um data leakage zu eliminieren!"
)
print(
    "Das Feld 'predicted_reaininglife' hätte auch interessante korrelationen, aber der name ist einfach zu verdächtig, als dass das schonmal ein output war."
)
print(
    "'maintenance_required' ist scheinbar eine abgeschwächte version vom 'anomaly_flag' bzw 'downtime_risk'"
)

corr_matrix

# %%
# Spalten auswählen
sensor_cols = ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]

# Daten standardisieren: (Wert - Mittelwert) / Standardabweichung
df_scaled = (df_eda[sensor_cols] - df_eda[sensor_cols].mean()) / df_eda[
    sensor_cols
].std()

# Horizontale Boxplots zeichnen
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_scaled, orient="h", ax=ax, palette="Set2")

ax.set_title("Standardisierte Sensorwerte (Z-Score)")
ax.set_xlabel("Abweichung vom Mittelwert (in Standardabweichungen)")
plt.show()

# %%
# 1. Plot-Objekte im OO-Stil erstellen
fig, ax = plt.subplots(figsize=(8, 4))

# 2. Histogramm auf Achse zeichnen
sns.histplot(
    df_eda.loc[:, "temperature"],
    kde=True,
    color="crimson",
    edgecolor="black",
    alpha=0.6,
    ax=ax,
)

# 3. Titel und Achsbeschriftung
ax.set_title("Normalverteilung: Temperatur", fontsize=14, pad=15)
ax.set_xlabel("'temperature' in °F", fontsize=11)
ax.set_ylabel("Anzahl Messwerte", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)

# 4. Anzeigen
fig.tight_layout()
plt.show()

# %%
# 1. Setup the figure and axis
fig, ax = plt.subplots(figsize=(8, 4))

# 2. Histogramm zeichnen
sns.histplot(
    df_eda.loc[:, "vibration"],
    kde=True,
    color="darkorange",
    edgecolor="black",
    alpha=0.6,
    ax=ax,
)

# 3. Titel und Achsbeschriftung
ax.set_title("Normalverteilung: Vibrationen", fontsize=14, pad=15)
ax.set_xlabel("'vibration' in ??? Einheiten", fontsize=11)
ax.set_ylabel("Anzahl Messwerte", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)

# 4. Anzeigen
fig.tight_layout()
plt.show()

# %%
# 1. Plot-Objekte im OO-Stil erstellen
fig, ax = plt.subplots(figsize=(8, 4))

# 2. Histogramm plotten
# 20-25 bins zeigen, ob der Verlauf flach oder eher spitz ist
ax.hist(df_eda["pressure"], bins=20, color="crimson", edgecolor="black", alpha=0.7)

# 3. Titel und Achsbeschriftung
ax.set_title("Gleichverteilung: Feuchtigkeit", fontsize=14, pad=15)
ax.set_xlabel("'humidity' in ??? Einheiten", fontsize=11)
ax.set_ylabel("Anzahl Messwerte", fontsize=11)

# Explizite integer X-ticks an den Quartilgrenzen
ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.tight_layout()
plt.show()

# %%
# 1 Plot-Objekte im OO-Stil erstellen
fig, ax = plt.subplots(figsize=(8, 4))

# 2. Histogramm zeichnen
# 20-25 bins zeigen, ob der Verlauf flach oder eher spitz ist
ax.hist(df_eda["pressure"], bins=20, color="indigo", edgecolor="black", alpha=0.7)

# 3. Diagramm- und Achsenobjekte erstellen
ax.set_title("Gleichverteilung: Druck", fontsize=14, pad=15)
ax.set_xlabel("'pressure' in ??? Einheiten", fontsize=11)
ax.set_ylabel("Anzahl Messwerte", fontsize=11)

# Explizite integer X-ticks an den Quartilgrenzen
ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.tight_layout()
plt.show()

# %%
# 1. Plot-Objekte im OO-Stil erstellen
fig, ax = plt.subplots(figsize=(8, 4))

# 2. Histogramm zeichnen
ax.hist(
    df_eda["energy_consumption"],
    bins=20,
    color="teal",
    edgecolor="black",
    alpha=0.7,
)

# 3. Titel und Achsbeschriftung
ax.set_title("Gleichverteilung: Energieverbrauch", fontsize=14, pad=15)
ax.set_xlabel("'energy_consumption' in ??? Einheiten", fontsize=11)
ax.set_ylabel("Frequency Count", fontsize=11)

ax.grid(axis="y", linestyle="--", alpha=0.7)

fig.tight_layout()
plt.show()

# %%
# 1. Create a 2-row, 3-column grid of axes (the bottom-right spot will remain blank)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# --- Row 1: The Uniform Distributions ---
# 1. Energy
axes[0, 0].hist(
    df_eda["energy_consumption"], bins=20, color="teal", edgecolor="black", alpha=0.7
)
axes[0, 0].set_title("1. Energy Consumption (Uniform)")
axes[0, 0].set_ylabel("Frequency")

# 2. Pressure
axes[0, 1].hist(
    df_eda["pressure"], bins=20, color="indigo", edgecolor="black", alpha=0.7
)
axes[0, 1].set_title("2. System Pressure (Uniform)")

# 3. Humidity
axes[0, 2].hist(
    df_eda["humidity"], bins=20, color="crimson", edgecolor="black", alpha=0.7
)
axes[0, 2].set_title("3. Ambient Humidity (Uniform)")

# --- Row 2: The Normal Distributions ---
# 4. Vibration
axes[1, 0].hist(
    df_eda["vibration"], bins=20, color="darkorange", edgecolor="black", alpha=0.7
)
axes[1, 0].set_title("4. System Vibration (Normal)")
axes[1, 0].set_ylabel("Frequency")

# 5. Temperature
axes[1, 1].hist(
    df_eda["temperature"], bins=20, color="darkred", edgecolor="black", alpha=0.7
)
axes[1, 1].set_title("5. Machine Temperature (Normal)")

# 6. Hide the empty 6th subplot so the layout looks clean
axes[1, 2].axis("off")

# 4. Global styling using OO syntax
fig.suptitle(
    "Sensor Feature Profiles: Uniform vs. Normal Distributions",
    fontsize=16,
    weight="bold",
    y=0.98,
)
fig.tight_layout()

plt.show()

# %%
# See exactly which failure types happen under which machine status
status_breakdown = pd.crosstab(df_eda["machine_status"], df_eda["failure_type"])
status_breakdown
# %%

# 1. Calculate percentages for each status row
# (divides each cell by the total sum of that row)
status_percentages = (
    pd.crosstab(df_eda["machine_status"], df_eda["failure_type"], normalize="index")
    * 100
)

# 2. Setup the plot
fig, ax = plt.subplots(figsize=(10, 6))

# 3. Plot as a stacked horizontal bar chart
status_percentages.plot(
    kind="barh",
    stacked=True,
    ax=ax,
    edgecolor="black",
    color=["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#1f77b4"],
)

# 4. Object-Oriented Styling
ax.set_title(
    "Operational Logic: Failure Types by Machine Status (%)", fontsize=14, pad=15
)
ax.set_xlabel("Percentage of Rows within Status (%)")
ax.set_ylabel("Machine Status Code")
ax.set_yticklabels(["Status 0 (Idle)", "Status 1 (Healthy)", "Status 2 (Alert)"])

# Place legend cleanly outside
ax.legend(title="Failure Type", bbox_to_anchor=(1.02, 1), loc="upper left")

fig.tight_layout()
plt.show()

# %% anomaly_flag & downtime_risk (Donuts mit Center-Text & Tabellen)
import matplotlib.pyplot as plt

# 1. Absolute Häufigkeiten berechnen und sortieren
counts_anomaly = df_eda["anomaly_flag"].value_counts().sort_index()
counts_risk = df_eda["downtime_risk"].value_counts().sort_index()

# 2. Erstelle eine Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# --- LINKES DIAGRAMM: anomaly_flag ---
wedges_anom, _ = ax1.pie(
    counts_anomaly,
    startangle=90,
    colors=["#4682B4", "#B22222"],
    wedgeprops={"edgecolor": "black", "width": 0.4},  # Donut-Breite
)
# Spaltenname fett in die Mitte des Donuts schreiben
ax1.text(
    0,
    0,
    "anomaly_flag",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="#333333",
)
ax1.set_title(
    "Starkes Ungleichgewicht der Klassen\n\nProblem für das SVM Modell\nStichproben, doppelte Einträge oder...\nden 'balanced' paramater?",
    fontsize=12,
    pad=15,
)

# Tabelle links
labels_anom = ["0 (Keine Anomalie)", "1 (Anomalie vorhanden)"]
data_anom = [
    [lbl, f"{val:,}", f"{(val / counts_anomaly.sum()) * 100:.2f}%"]
    for lbl, val in zip(labels_anom, counts_anomaly)
]
table_anom = ax1.table(
    cellText=data_anom,
    colLabels=["Klasse", "Anzahl", "Anteil"],
    loc="bottom",
    cellLoc="center",
)
table_anom.scale(1, 1.4)
ax1.axis("off")


# --- RECHTES DIAGRAMM: downtime_risk ---
risk_labels = [f"Risiko: {float(val):.1f}" for val in counts_risk.index]
risk_colors = ["#2AAE6F", "#85C1E9", "#F7DC6F", "#F5B041", "#EC7063"][
    : len(counts_risk)
]

wedges_risk, _ = ax2.pie(
    counts_risk,
    startangle=90,
    colors=risk_colors,
    wedgeprops={"edgecolor": "black", "width": 0.4},
)
# Spaltenname fett in die Mitte des Donuts schreiben
ax2.text(
    0,
    0,
    "downtime_risk",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="#333333",
)
ax2.set_title(
    "Starkes Ungleichgewicht der Klassen\n\nProblem für das SVM Modell\nStichproben, doppelte Einträge oder...\nden 'balanced' paramater?",
    fontsize=12,
    pad=15,
)

# Tabelle rechts
data_risk = [
    [lbl, f"{val:,}", f"{(val / counts_risk.sum()) * 100:.2f}%"]
    for lbl, val in zip(risk_labels, counts_risk)
]
table_risk = ax2.table(
    cellText=data_risk,
    colLabels=["Klasse", "Anzahl", "Anteil"],
    loc="bottom",
    cellLoc="center",
)
table_risk.scale(1, 1.4)
ax2.axis("off")

# 5. Layout optimieren und anzeigen
fig.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.show()
# %%
# 1. Generate the cross-tab data
comparison = pd.crosstab(df_eda["anomaly_flag"], df_eda["downtime_risk"] > 0)

# 2. Setup the plot
fig, ax = plt.subplots(figsize=(6, 5))

# 3. Plot a heatmap of the overlap
sns.heatmap(
    comparison,
    annot=True,  # Put the exact numbers inside the boxes
    fmt="d",  # Format numbers as integers (no scientific notation)
    cmap="Blues",  # Color scheme
    cbar=False,  # Hide the color bar side-legend for a cleaner look
    linewidths=2,
    linecolor="black",
    ax=ax,
)

# 4. Labeling via Object-Oriented syntax
ax.set_title("Feature Redundancy Verification", fontsize=14, pad=15)
ax.set_xlabel("Downtime Risk > 0 (True/False)", fontsize=11)
ax.set_ylabel("Anomaly Flag (0/1)", fontsize=11)

fig.tight_layout()
plt.show()

# 5. Display the plot
plt.show()

# %%
# 1. Setup the figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# 2. Create a horizontal boxplot
ax.boxplot(
    df_eda["predicted_remaining_life"],
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="blue"),
    medianprops=dict(color="red", linewidth=2),
)

# 3. Customize using OO syntax
ax.set_title(
    "Distribution of Predicted Remaining Life (NEEDEs Scaling/normalization!)",
    fontsize=14,
    pad=15,
)
ax.set_xlabel("Remaining Life (Cycles/Days)")
ax.set_yticklabels([])  # Hides the default numeric Y-axis label

fig.tight_layout()
plt.show()


# %%
# Check how many unique machines are in the dataset
print(f"Total Unique Machines: {df_eda['machine_id'].nunique()}")

# Check the exact timeframe of your study project
# (Converting to datetime first ensures accurate min/max calculations)
df_eda["timestamp"] = pd.to_datetime(df_eda["timestamp"])
print(f"Data Starts: {df_eda['timestamp'].min()}")
print(f"Data Ends:   {df_eda['timestamp'].max()}")

# %%
# 1. Sichere Kopie des Originalen DataFrames erstellen
df_prep = df_eda.copy()

# 2. Zeitstempel explizit als Datetime formatieren (falls noch nicht geschehen)
df_prep["timestamp"] = pd.to_datetime(df_prep["timestamp"])

# 3. Features und Zielvariablen trennen
# Wir definieren hier schon einmal die Liste der reinen Sensor-Features
sensor_features = [
    "energy_consumption",
    "pressure",
    "humidity",
    "vibration",
    "temperature",
]

# %%
# 1. Schwellenwert definieren (3 Standardabweichungen)
z_threshold = 3

columns_to_check = ["vibration", "temperature"]

print("--- Ausreißer-Analyse (3-Sigma-Regel) ---")
for col in columns_to_check:
    mean = df_prep[col].mean()
    std = df_prep[col].std()

    # Obere und untere Grenze berechnen
    lower_bound = mean - (z_threshold * std)
    upper_bound = mean + (z_threshold * std)

    # Ausreißer zählen (Verwendung der lesbaren .query() Methode)
    outliers = df_prep.query(f"{col} < {lower_bound} or {col} > {upper_bound}")
    outlier_count = len(outliers)

    print(
        f"Spalte '{col}': {outlier_count} Zeilen liegen außerhalb von [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    # Best Practice: Werte an den Grenzen kappen (Winsorisierung)
    # Statt Daten zu löschen, setzen wir extreme Ausreißer auf die Grenze fest.
    df_prep[col] = df_prep[col].clip(lower=lower_bound, upper=upper_bound)

print("\n-> Ausreißer wurden erfolgreich auf die 3-Sigma-Grenzen gekappt (Clipping).")
# %%
from sklearn.preprocessing import StandardScaler

# 1. Liste aller 5 Sensor-Features definieren
sensor_features = [
    "energy_consumption",
    "pressure",
    "humidity",
    "vibration",
    "temperature",
]

# 2. Den StandardScaler initialisieren
scaler = StandardScaler()

# 3. Die Skalierung auf die gekappten Daten anwenden
df_prep[sensor_features] = scaler.fit_transform(df_prep[sensor_features])

# 4. Zur Überprüfung die neuen Min/Max-Werte anzeigen lassen
print(df_prep[sensor_features].describe().loc[["mean", "std", "min", "max"]])
# %%
import matplotlib.pyplot as plt

# 1. Erstelle ein 2x3 Grid für die 5 Sensor-Spalten
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# Farbschema für die transformierten Daten definieren
colors = ["#008080", "#4B0082", "#DC143C", "#FF8C00", "#8B0000"]

# --- Reihe 1: Die gleichverteilten, skalierten Sensoren ---
# 1. Energy Consumption
axes[0, 0].hist(
    df_prep["energy_consumption"],
    bins=25,
    color=colors[0],
    edgecolor="black",
    alpha=0.7,
)
axes[0, 0].set_title("1. Energy Consumption (Standardized)")
axes[0, 0].set_ylabel("Anzahl Einträge")

# 2. Pressure
axes[0, 1].hist(
    df_prep["pressure"], bins=25, color=colors[1], edgecolor="black", alpha=0.7
)
axes[0, 1].set_title("2. System Pressure (Standardized)")

# 3. Humidity
axes[0, 2].hist(
    df_prep["humidity"], bins=25, color=colors[2], edgecolor="black", alpha=0.7
)
axes[0, 2].set_title("3. Ambient Humidity (Standardized)")

# --- Reihe 2: Die normalverteilten, skalierten Sensoren ---
# 4. Vibration
axes[1, 0].hist(
    df_prep["vibration"], bins=25, color=colors[3], edgecolor="black", alpha=0.7
)
axes[1, 0].set_title("4. System Vibration (Standardized)")
axes[1, 0].set_ylabel("Anzahl Einträge")

# 5. Temperature
axes[1, 1].hist(
    df_prep["temperature"], bins=25, color=colors[4], edgecolor="black", alpha=0.7
)
axes[1, 1].set_title("5. Machine Temperature (Standardized)")

# 6. Den leeren 6. Plot unsichtbar machen
axes[1, 2].axis("off")

# --- Globale Anpassungen (Objektorientiert) ---
fig.suptitle(
    "Sensor-Features NACH der Standardisierung (Mittelwert = 0, Varianz = 1)",
    fontsize=16,
    weight="bold",
    y=0.98,
)

# Einheitliche X-Achsen-Beschriftung, um die gleiche Skalierung zu betonen
for row in axes:
    for ax in row:
        if ax != axes[1, 2]:  # Nicht für den versteckten Plot
            ax.set_xlabel("Standardabweichungen (Z-Score)")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.tight_layout()
plt.show()
# %%
from sklearn.preprocessing import LabelEncoder

# --- 1. Variante A: Label Encoding für failure_type (Perfekt, falls es das Target wird) ---
le = LabelEncoder()
df_prep["failure_type_encoded"] = le.fit_transform(df_prep["failure_type"])

# Zeigt dir das Mapping an (z.B. Normal -> 2, Overheating -> 3, etc.)
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("--- Label Encoding Mapping für 'failure_type' ---")
for key, val in mapping.items():
    print(f"  {key} -> {val}")


# --- 2. Variante B: One-Hot-Encoding (Perfekt, falls sie Features werden) ---
# Wir erstellen Dummies für beide Spalten, behalten aber die Originalspalten im df_eda zur Übersicht
df_prep = pd.get_dummies(
    df_prep,
    columns=["failure_type", "machine_status"],
    prefix=["fail", "status"],
    dtype=int,  # Erstellt direkt 0 und 1 statt True/False
    drop_first=False,  # Wir behalten alle, um beim Daten-Ausschluss flexibel zu sein
)

print("\n--- Neue One-Hot-Encoded Spalten erfolgreich hinzugefügt ---")
# Filtert und zeigt nur die neu generierten Dummy-Spaltennamen an
dummy_cols = [col for col in df_prep.columns if col.startswith(("fail_", "status_"))]
print(dummy_cols)
# %%
# Initialisiere den Scaler speziell für dieses Feature
life_scaler = StandardScaler()

# Skaliere die Spalte und überschreibe sie im vorbereiteten DataFrame
df_prep["predicted_remaining_life"] = life_scaler.fit_transform(
    df_prep[["predicted_remaining_life"]]
)

print("--- predicted_remaining_life nach der Skalierung ---")
print(df_prep["predicted_remaining_life"].describe().loc[["mean", "std", "min", "max"]])
# %%

# 1. Setup: Ein Grid mit 1 Zeile und 2 Spalten
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# 2. Plot vor der Skalierung (aus dem originalen DataFrame df_eda)
ax1.boxplot(
    df_eda["predicted_remaining_life"],
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="lightgray", color="gray"),
    medianprops=dict(color="black", linewidth=2),
)
ax1.set_title("Vorher: Originale Skala (1 bis 499)")
ax1.set_xlabel("Tage / Zyklen")
ax1.set_yticklabels([])

# 3. Plot nach der Skalierung (aus dem vorbereiteten DataFrame df_prep)
ax2.boxplot(
    df_prep["predicted_remaining_life"],
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="blue"),
    medianprops=dict(color="red", linewidth=2),
)
ax2.set_title("Nachher: Standardisierte Skala (Z-Score)")
ax2.set_xlabel("Standardabweichungen um den Mittelwert (0)")
ax2.set_yticklabels([])

# 4. Globale Titel und Layout
fig.suptitle(
    "Transformation der Restlebensdauer (Predicted Remaining Life)",
    fontsize=14,
    weight="bold",
    y=1.05,
)
fig.tight_layout()

plt.show()

# %%
df_eda["failure_type"].value_counts()
# %%
# 1. Filter out 'Normal' to see only actual failures
failures_only = df_eda.query("failure_type != 'Normal'")["failure_type"].value_counts()

# 2. Setup the plot objects
fig, ax = plt.subplots(figsize=(10, 4))

# 3. Create a horizontal bar chart for the failures
ax.barh(failures_only.index, failures_only.values, edgecolor="black")

# 4. Clean up the labels using Object-Oriented style
ax.set_title(
    'Distribution of Machine Failures (Excluding "Normal" Status)', fontsize=14, pad=15
)
ax.set_xlabel("Number of Incidents")
ax.invert_yaxis()  # Puts the highest count (Vibration) at the top

# 5. Add exact count text next to each bar
for index, value in enumerate(failures_only.values):
    ax.text(value + 20, index, f"{value:,}", va="center", weight="bold")

fig.tight_layout()
plt.show()

# %% maintenance_required
# 1. Get the frequency counts
counts = df_eda["maintenance_required"].value_counts()
print(counts)

# 2. Create the figure and axis objects
fig, ax = plt.subplots(figsize=(6, 6))

# 3. Plot the pie chart on the 'ax' object
ax.pie(
    counts,
    labels=["0 (Majority)", "1 (Minority)"],
    autopct="%1.1f%%",  # Shows the percentage (e.g., 80.0%)
    startangle=90,  # Rotates the chart for a cleaner look
    colors=["#ff9999", "#66b3ff"],  # Distinct colors
    wedgeprops={"edgecolor": "black"},  # Adds a clean border
)

# 4. Set titles and layout using the object-oriented style
ax.set_title(
    "Imbalanced class distibution\n\n problem for future SVM model\n sampling, double entries or synthetic filler?",
    fontsize=14,
    pad=20,
)
fig.tight_layout()
