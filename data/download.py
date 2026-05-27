import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, Union
from urllib.parse import ParseResult, urlparse, urlunparse
from zipfile import BadZipFile, ZipFile

KAGGLE_URL = urlparse(
    "https://www.kaggle.com/api/v1/datasets/download/ziya07/smart-manufacturing-iot-cloud-monitoring-dataset"
)
DOWNLOAD_FOLDER = Path("data") / "raw"
ZIP_PATH = DOWNLOAD_FOLDER.joinpath(KAGGLE_URL.path.split("/").pop()).with_suffix(
    ".zip"
)
CSV_PATH = ZIP_PATH.with_suffix(".csv")


T = TypeVar("T")  # Typ für Erfolg (Ok)
E = TypeVar("E")  # Typ für Fehler (Err)


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E


# Der Result-Typ ist eine Union aus beiden Zuständen
Result = Union[Ok[T], Err[E]]


def download(url: ParseResult, path: Path) -> Result[Path, str]:
    zip_path = path.joinpath(url.path.split("/").pop()).with_suffix(".zip")
    # 1. Zielordner erstellen falls nicht vorhanden
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Prüfen, ob die zip Datei bereits heruntergeladen wurde
    if zip_path.exists():
        return Ok(zip_path)

    # 3. Download-Logik mit Fehlerbehandlung
    try:
        # Kaggle erfordert oft einen User-Agent, um 403-Fehler zu vermeiden
        req = urllib.request.Request(
            urlunparse(url), headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())
        return Ok(zip_path)
    except urllib.error.HTTPError as e:
        return Err(f"HTTP-Fehler beim Download: {e.code} - {e.reason}")
    except urllib.error.URLError as e:
        return Err(f"Netzwerkfehler beim Download: {e.reason}")
    except Exception as e:
        return Err(f"Unerwarteter Fehler beim Download: {e}")


def extract_smart_manufacturing_data(
    path: Path, consume: bool = False
) -> Result[str, str]:

    csv_path = path.with_suffix(".csv")
    file_in_zip = "smart_manufacturing_data.csv"

    if csv_path.exists():
        return Ok("CSV ist bereits extrahiert")

    # Entpacken mit Fehlerbehandlung
    try:
        # ZIP-Datei öffnen
        with ZipFile(path, "r") as z:
            # Prüfen, ob die Datei im ZIP existiert
            if file_in_zip not in z.namelist():
                return Err(
                    "Datei 'smart_manufacturing_data.csv' nicht im ZIP gefunden."
                )

            # Inhalt lesen und direkt als gewünschte CSV schreiben
            data = z.read("smart_manufacturing_data.csv")
            csv_path.write_bytes(data)

            if consume:
                path.unlink()
            return Ok("Extraktion erfolgreich")

    except BadZipFile:
        # Lösche die defekte ZIP, damit beim nächsten Lauf ein neuer Download startet
        if path.exists():
            path.unlink()
        return Err(f"Fehler: Die Datei {path} ist beschädigt (kein gültiges ZIP).")
    except Exception as e:
        return Err(f"Fehler beim Entpacken: {e}")


match download(KAGGLE_URL, DOWNLOAD_FOLDER):
    case Ok(zip_path):
        print(zip_path)
        match extract_smart_manufacturing_data(zip_path, False):
            case Ok(msg):
                print(msg)
            case Err(msg):
                print(msg)
    case Err(msg):
        print(msg)
