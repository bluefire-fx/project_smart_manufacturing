# ML-Projekt Smart Manufacturing Data (Kaggle)

Dieses Repository enthält das Machine-Learning-Projekt für Predictive Analytics an der HS Aalen. 
Für das Paket- und Umgebungsmanagement wird das moderne Tool **uv** verwendet.

## 🛠️ Voraussetzungen

Stelle sicher, dass du `uv` auf deinem System installiert hast. Falls nicht, installiere es kurz:

- **macOS/Linux:** `curl -LsSf https://astral.sh | sh`
- **Windows (PowerShell):** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh | iex"`

## 🚀 Erste Schritte

Folge diesen Schritten, um das Projekt lokal aufzusetzen:

### 1. Repository klonen
```bash
git clone https://github.com
cd REPOSITORNAME
```

### 2. Virtuelle Umgebung erstellen und Abhängigkeiten installieren
`uv` erstellt die Umgebung und installiert alle Pakete aus der `pyproject.toml` mit einem einzigen Befehl:
```bash
uv sync
```

### 3. Umgebung aktivieren
- **macOS/Linux:** `source .venv/bin/activate`
- **Windows:** `.venv\Scripts\activate`

---

## 📂 Datensatz herunterladen

Da der Datensatz nicht im Repository liegt, kannst du ihn mit folgendem Befehl direkt in den richtigen Ordner laden und entpacken:

```bash
uv run python ./data/download.py
```


---

## 💻 Notebook philosophie

Im git repo werden keine ipynb dateien verwaltet. Als alternative dienen Notebooks im Helium stil. Zwischen den beiden Stilen kann aber problemlos konvertiert werden:

Helium => Jupyter Notebook
```bash
uv run ipynb-py-convert 00_converter_example.py ipynb-py-convert 00_converter_example.ipynb
```
Jupyter Notebook => Helium
```bash
uv run ipynb-py-convert 00_converter_example.ipynb ipynb-py-convert 00_converter_example.py
```
