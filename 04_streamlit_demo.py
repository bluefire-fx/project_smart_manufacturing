import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Konfiguration der Seite
st.set_page_config(
    page_title="IoT Smart Manufacturing Dashboard", page_icon="🏭", layout="centered"
)

logo_url = (
    "https://upload.wikimedia.org/wikipedia/commons/2/21/HS-Aalen-Logo-rgb_RZ.svg"
)

# 2. Wir nutzen Spalten, um das Logo elegant rechts oben zu platzieren
col_title, col_logo = st.columns([3, 1])

with col_title:
    st.title("🏭 Smart Manufacturing: Live IoT-Anomalie-Erkennung")

with col_logo:
    # Zeigt das Logo skaliert an
    st.image(logo_url, width="stretch")


# 3. Modell und Scaler sicher laden
@st.cache_resource  # Verhindert, dass das Modell bei jedem Klick neu geladen wird
def load_model_artifacts():
    try:
        artifacts = joblib.load("models/svm_anomaly_detector.joblib")
        return artifacts
    except FileNotFoundError:
        st.error(
            "❌ Modell-Datei nicht gefunden! Bitte führen Sie zuerst Ihr Trainings-Skript aus."
        )
        return None


artifacts = load_model_artifacts()

if artifacts:
    winsor_bounds = artifacts["winsor_bounds"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    features = artifacts["features"]
    parameter_c = artifacts["parameter_c"]

    st.markdown(f"""
        Dieses Dashboard demonstriert ein optimiertes **Support Vector Machine (SVM) Modell ($C={parameter_c}$)**
    zur Echtzeit-Überwachung von Produktionsmaschinen.
    Ändern Sie die Sensorwerte in der Seitenleiste, um den Zustand der Maschine zu simulieren.
    """)

    # 4. Seitenleiste für die Live-Eingabe (Simulation der IoT-Sensoren)
    st.sidebar.header("📊 Sensor-Live-Daten (Simulation)")
    st.sidebar.markdown("Stellen Sie hier die aktuellen Maschinenwerte ein:")

    # Dynamische Slider basierend auf Ihren Features
    user_inputs = {}
    if "temperature" in features:
        user_inputs["temperature"] = st.sidebar.slider(
            "Temperatur (°C)", min_value=10.0, max_value=120.0, value=50.0, step=0.1
        )
    if "vibration" in features:
        user_inputs["vibration"] = st.sidebar.slider(
            "Vibration (mm/s)", min_value=0.0, max_value=20.0, value=2.5, step=0.05
        )

    # 5. Daten für das Modell aufbereiten
    # Input in DataFrame umwandeln (wichtig für die Feature-Reihenfolge)
    input_df = pd.DataFrame([user_inputs])[features]

    # ==========================================
    # NEU: WINSORIZATION (AUSREISSER-BEGRENZUNG)
    # ==========================================
    # Wir begrenzen die Live-Eingaben auf die gelernten Winsor-Grenzen aus dem Training
    for col in features:
        if col in winsor_bounds:
            lower_bound = float(winsor_bounds[col][0])
            upper_bound = float(winsor_bounds[col][1])
            input_df[col] = input_df[col].clip(lower_bound, upper_bound)
    # ==========================================

    # Skalierung mit dem gespeicherten Scaler (Data-Leakage-Schutz!)
    input_scaled = scaler.transform(input_df)

    # 6. Vorhersage und Wahrscheinlichkeiten berechnen
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[
        0
    ]  # Array-Index [0] für die erste Zeile fixiert
    anomaly_prob = probabilities[1] * 100

    # ==========================================
    # NEU: VIRTUELLE ZEITACHSE (SESSION STATE)
    # ==========================================
    # Streamlit vergisst Variablen bei jedem Klick. 'session_state' speichert sie dauerhaft.
    if "history" not in st.session_state:
        # Wir starten mit einer leeren Liste für die Historie
        st.session_state.history = []

    # Wir speichern den aktuellen Datenpunkt ab
    current_point = {
        "Temperatur": user_inputs.get("temperature", 0),
        "Vibration": user_inputs.get("vibration", 0),
        "Anomalie-Risiko (%)": anomaly_prob,
        "Status": "Kritisch" if prediction == 1 else "Normal",
    }
    st.session_state.history.append(current_point)

    # Um den Arbeitsspeicher nicht zu sprengen, behalten wir nur die letzten 30 Messpunkte
    if len(st.session_state.history) > 30:
        st.session_state.history.pop(0)

    # Die Historie in eine Tabelle für den Plot umwandeln
    history_df = pd.DataFrame(st.session_state.history)
    # ==========================================

    # 7. Visuelle Ausgabe der Ergebnisse
    st.subheader("🔮 Aktueller System-Status")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("🚨 ANOMALIE DETEKTIERT!")
            st.metric(
                label="Zustand",
                value="Kritisch",
                delta="- Gefahr",
                delta_color="inverse",
            )
        else:
            st.success("✅ SYSTEM STABIL")
            st.metric(label="Zustand", value="Normal", delta="Optimal")

    with col2:
        st.metric(label="Anomalie-Wahrscheinlichkeit", value=f"{anomaly_prob:.1f} %")
        st.progress(int(anomaly_prob))

    # ==========================================
    # NEU: DAS LIVE-LINIENDIAGRAMM
    # ==========================================
    st.subheader("📈 Sensor- und Risiko-Verlauf (Virtuelle Zeitachse)")
    st.markdown(
        "Bewegen Sie die Slider, um Datenpunkte auf der Zeitachse hinzuzufügen:"
    )

    # Wir zeichnen ein Liniendiagramm, das das Anomalie-Risiko über die Zeit anzeigt
    # Streamlit bringt dafür eine extrem einfache, schicke Funktion mit:
    st.line_chart(data=history_df, y="Anomalie-Risiko (%)", width="stretch")

    # Button zum Zurücksetzen des Verlaufs
    if st.button("🔄 Verlauf zurücksetzen"):
        st.session_state.history = []
        st.rerun()
    # ==========================================

    # Info-Box für die Prüfer
    st.markdown("---")
    with st.expander("🔬 Technische Modelldetails (für die Dokumentation)"):
        st.write(f"**Verwendete Features:** {', '.join(features)}")
        st.write("**Modell-Typ:** RBF-Support Vector Classifier (SVC)")
        st.write(
            "**Eingestellte Hyperparameter:** $C=100.0$, $\\gamma=\\text{scale}$, Class Weight = *balanced*"
        )
        st.write(
            "Die Sensordaten werden vor der Übergabe an das Modell live durch einen vor-trainierten StandardScaler transformiert."
        )
