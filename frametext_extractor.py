import cv2
import numpy as np
import pytesseract
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Tuple, Optional
import openai
import re
import sys
import argparse  # Für Kommandozeilenargumente
import tiktoken
from functools import wraps
import time
from dotenv import load_dotenv
import requests  # Für erweiterte Fehlerbehandlung

# Load environment variables from a .env file if it exists
load_dotenv()

# Optional: Decorator to retry on certain exceptions
def retry_on_exception(max_retries=3, delay=2, exceptions=(Exception,)):
    """
    Ein Decorator, der eine Funktion bei bestimmten Exceptions erneut versucht.

    :param max_retries: Maximale Anzahl von Versuchen.
    :param delay: Verzögerung zwischen den Versuchen in Sekunden.
    :param exceptions: Tuple von Exceptions, die behandelt werden sollen.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    logging.warning(
                        f"Exception {e} in {func.__name__}. Versuch {retries}/{max_retries} nach {current_delay} Sekunden."
                    )
                    time.sleep(current_delay)
                    current_delay *= 2  # Exponentielles Backoff
            logging.error(f"Maximale Anzahl von Versuchen ({max_retries}) für {func.__name__} überschritten.")
            raise
        return wrapper
    return decorator

def set_tesseract_path(tesseract_path: Optional[str] = None):
    """
    Setzt den Pfad für Tesseract-OCR basierend auf Umgebungsvariable, CLI-Argument oder Standardpfaden.
    Wirft FileNotFoundError, wenn Tesseract nicht gefunden wird.

    :param tesseract_path: Optional benutzerdefinierter Pfad zur Tesseract-OCR ausführbaren Datei.
    """
    # Überprüfe zuerst die Umgebungsvariable
    tesseract_env_path = os.getenv("TESSERACT_PATH")
    if tesseract_env_path and os.path.exists(tesseract_env_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_env_path
        logging.info(f"Verwende benutzerdefinierten Tesseract-Pfad aus der Umgebung: {tesseract_env_path}")
        return

    # Dann überprüfe das CLI-Argument
    if tesseract_path:
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logging.info(f"Verwende benutzerdefinierten Tesseract-Pfad aus dem CLI-Argument: {tesseract_path}")
            return
        else:
            logging.warning(f"Tesseract nicht am angegebenen CLI-Pfad gefunden: {tesseract_path}")

    # Fallback-Strategie
    if os.name == 'nt':
        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win_path):
            pytesseract.pytesseract.tesseract_cmd = default_win_path
            logging.info(f"Tesseract-Pfad auf den standardmäßigen Windows-Standort gesetzt: {default_win_path}")
            return
    else:
        tesseract_shutil = shutil.which("tesseract")
        if tesseract_shutil:
            pytesseract.pytesseract.tesseract_cmd = tesseract_shutil
            logging.info(f"Tesseract in PATH gefunden: {tesseract_shutil}")
            return

    # Letzter Versuch: Suche in bekannten Standardpfaden
    standard_paths = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract"
    ]
    for path in standard_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logging.info(f"Tesseract-Pfad auf Standardpfad gesetzt: {path}")
            return

    logging.error("Tesseract-OCR nicht gefunden. Bitte installiere Tesseract oder setze den richtigen Pfad.")
    raise FileNotFoundError("Tesseract-OCR nicht gefunden.")

def extract_text_with_pytesseract(frame: np.ndarray) -> str:
    """
    Extrahiert Text aus einem gegebenen Frame mittels Tesseract-OCR.

    :param frame: Das Bildframe, aus dem Text extrahiert werden soll.
    :return: Extrahierter Text als String.
    """
    config = '--psm 6 --oem 1'
    try:
        text = pytesseract.image_to_string(frame, config=config)
        logging.debug(f"Extrahierter Text: {text}")
        return text
    except pytesseract.TesseractError as e:
        logging.error(f"Tesseract OCR Fehler: {e}")
        return ""
    except Exception as e:
        logging.error(f"Unerwarteter Fehler während der OCR: {e}")
        return ""

def detect_movement(prev_frame: np.ndarray, curr_frame: np.ndarray, base_threshold: float = 0.05) -> bool:
    """
    Erkennt Bewegung zwischen zwei aufeinanderfolgenden Frames.

    :param prev_frame: Der vorherige Graustufen-Frame.
    :param curr_frame: Der aktuelle Graustufen-Frame.
    :param base_threshold: Basis-Schwellenwert für die Bewegungserkennung.
    :return: True, wenn Bewegung erkannt wird, sonst False.
    """
    # Rauschunterdrückung mittels GaussianBlur
    prev_blur = cv2.GaussianBlur(prev_frame, (5, 5), 0)
    curr_blur = cv2.GaussianBlur(curr_frame, (5, 5), 0)
    diff = cv2.absdiff(prev_blur, curr_blur)
    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    non_zero_count = np.count_nonzero(diff_thresh)
    total_pixels = diff_thresh.size
    average_intensity = np.mean(diff_thresh) / 255  # Normalisiert auf [0,1]

    # Dynamischer Schwellenwert basierend auf der durchschnittlichen Intensität
    dynamic_threshold = base_threshold * (1 + average_intensity)
    movement = (non_zero_count / total_pixels) > dynamic_threshold
    logging.debug(f"Bewegung erkannt: {movement} (Schwellenwert: {dynamic_threshold:.4f}, Durchschnittliche Intensität: {average_intensity:.4f})")
    return movement

def preprocess_frame(frame: np.ndarray, scale_factor: int = 2, adaptive_threshold: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Verarbeitet einen Frame durch Skalierung, Umwandlung in Graustufen und Anwendung eines Schwellenwerts.

    :param frame: Das Originalbildframe.
    :param scale_factor: Faktor zur Skalierung des Frames (Upscaling).
    :param adaptive_threshold: Ob adaptives Thresholding anstelle von Otsu verwendet werden soll.
    :return: Tuple des binarisierten Frames und des Graustufenframes.
    """
    try:
        # Upscaling statt Downscaling
        resized_frame = cv2.resize(frame, (frame.shape[1] * scale_factor, frame.shape[0] * scale_factor), interpolation=cv2.INTER_LINEAR)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        if adaptive_threshold:
            thresh_frame = cv2.adaptiveThreshold(
                gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            logging.debug("Adaptives Thresholding angewendet.")
        else:
            _, thresh_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logging.debug("Otsu'sches Thresholding angewendet.")
        return thresh_frame, gray_frame
    except cv2.error as e:
        logging.error(f"OpenCV Fehler während der Vorverarbeitung: {e}")
        logging.debug(f"Frame Abmessungen: {frame.shape}, dtype: {frame.dtype}")
        return None, None
    except Exception as e:
        logging.error(f"Unerwarteter Fehler während der Frame-Vorverarbeitung: {e}")
        return None, None

def process_frame(thresh_frame: np.ndarray) -> str:
    """
    Führt die Textextraktion auf einem vorverarbeiteten Frame durch.

    :param thresh_frame: Das binarisierte Bildframe.
    :return: Extrahierter Text als String.
    """
    if thresh_frame is not None:
        return extract_text_with_pytesseract(thresh_frame)
    return ""

def estimate_tokens(text: str, model: str = "deepseek-chat") -> int:
    """
    Schätzt die Anzahl der Tokens in einem gegebenen Text basierend auf dem verwendeten Modell.

    :param text: Der zu analysierende Text.
    :param model: Das Modell, das für die Tokenisierung verwendet wird.
    :return: Geschätzte Anzahl der Tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.error(f"Unbekanntes Modell für die Token-Schätzung: {model}")
        raise
    return len(encoding.encode(text))

def process_video_optimized(
    video_path: str,
    frame_interval: float = 1.0,  # Zeitintervall in Sekunden
    scale_factor: int = 2,
    motion_threshold: float = 0.05,
    max_workers: Optional[int] = 4,  # Standard auf 4 gesetzt
    supported_formats: Optional[List[str]] = None,  # Erweiterte unterstützte Formate
    adaptive_threshold: bool = False  # Option für adaptives Thresholding
) -> List[Tuple[int, str]]:
    """
    Verarbeitet ein Video, extrahiert Text aus relevanten Frames basierend auf Bewegungserkennung.

    :param video_path: Pfad zur Videodatei.
    :param frame_interval: Zeitintervall in Sekunden zur Auswahl der Frames.
    :param scale_factor: Faktor zur Skalierung der Frames.
    :param motion_threshold: Schwellenwert für die Bewegungserkennung.
    :param max_workers: Maximale Anzahl von Arbeitern für parallele Verarbeitung.
    :param supported_formats: Liste der unterstützten Videoformate.
    :param adaptive_threshold: Ob adaptives Thresholding verwendet werden soll.
    :return: Liste von Tupeln (Frame-Nummer, extrahierter Text).
    """
    if supported_formats is None:
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

    extracted_text: List[Tuple[int, str]] = []

    try:
        if not os.path.exists(video_path):
            logging.error(f"Videodatei '{video_path}' existiert nicht.")
            return extracted_text

        if not os.path.isfile(video_path):
            logging.error(f"'{video_path}' ist keine gültige Datei.")
            return extracted_text

        # Überprüfe das Videoformat
        _, ext = os.path.splitext(video_path)
        if ext.lower() not in supported_formats:
            logging.error(f"Nicht unterstütztes Videoformat '{ext}'. Unterstützte Formate sind: {supported_formats}")
            return extracted_text

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logging.error("Fehler beim Öffnen der Videodatei.")
                return extracted_text

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                logging.error("Konnte die FPS des Videos nicht bestimmen.")
                return extracted_text

            frame_interval_frames = max(1, int(fps * frame_interval))

            logging.info(f"Video FPS: {fps}")
            logging.info(f"Verarbeite jedes {frame_interval_frames}. Frame (entspricht {frame_interval} Sekunden).")

            frame_num = 0
            frames_to_process = []
            prev_gray = None  # Initialisierung für Bewegungserkennung

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % frame_interval_frames == 0:
                    thresh_frame, gray_frame = preprocess_frame(
                        frame, scale_factor=scale_factor, adaptive_threshold=adaptive_threshold
                    )
                    if thresh_frame is not None and gray_frame is not None:
                        if prev_gray is None or detect_movement(prev_gray, gray_frame, motion_threshold):
                            frames_to_process.append((frame_num, thresh_frame))
                            logging.debug(f"Frame {frame_num} zur OCR-Verarbeitung hinzugefügt.")
                        prev_gray = gray_frame

                frame_num += 1

            logging.info(f"Alle relevanten Frames ({len(frames_to_process)}) für die OCR-Verarbeitung gesammelt.")

            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_frame = {executor.submit(process_frame, thresh): frame_num
                                   for frame_num, thresh in frames_to_process}
                logging.info("Starte OCR-Verarbeitung mit ThreadPoolExecutor.")

                for future in as_completed(future_to_frame):
                    frame_num = future_to_frame[future]
                    try:
                        text = future.result()
                        if text.strip():
                            results.append((frame_num, text.strip()))
                            logging.debug(f"OCR für Frame {frame_num} abgeschlossen.")
                    except pytesseract.TesseractError as e:
                        logging.error(f"Tesseract-Fehler bei Frame {frame_num}: {e}")
                    except cv2.error as e:
                        logging.error(f"OpenCV-Fehler bei Frame {frame_num}: {e}")
                    except Exception as e:
                        logging.error(f"Unerwarteter Fehler bei Frame {frame_num}: {e}")

            # Sortiere Ergebnisse nach Frame-Nummer
            results.sort(key=lambda x: x[0])

            # Füge die extrahierten Texte hinzu
            extracted_text = results

        finally:
            cap.release()

    except Exception as e:
        logging.error(f"Fehler bei der Videoverarbeitung: {e}")

    return extracted_text  # Rückgabe der Liste von Tupeln (Frame-Nummer, Text)

def extract_output_content(text: str) -> str:
    """
    Extrahiert den Inhalt innerhalb der <output>-Tags aus dem gegebenen Text.

    :param text: Der vollständige zu analysierende Text.
    :return: Zusammengefasster Inhalt innerhalb der <output>-Tags oder der vollständige Text, falls keine Tags gefunden wurden.
    """
    # Robuste Extraktion der <output>-Tags
    matches = re.findall(r'<output>(.*?)</output>', text, re.DOTALL | re.IGNORECASE)
    if matches:
        # Mehrere Blöcke zusammenfassen
        return "\n".join(m.strip() for m in matches)
    else:
        # Versuch, den vollständigen Text zu verwenden
        logging.warning("Keine <output>-Tags in der LLM-Antwort gefunden. Versuch, den vollständigen Text zu verwenden.")
        return text.strip()

@retry_on_exception(
    max_retries=5,
    delay=2,
    exceptions=(
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.ServiceUnavailableError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        openai.error.OpenAIError,
    )
)
def correct_text_with_llm(text: str, api_key: str) -> str:
    """
    Korrigiert den gegebenen Text mittels eines LLM (Language Model) und gibt den bereinigten Text zurück.

    :param text: Der zu korrigierende Text.
    :param api_key: API-Schlüssel für die DeepSeek API.
    :return: Korrigierter Text als String.
    """
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="deepseek-chat",  # **WICHTIG: DeepSeek-Modell verwenden**
            messages=[
                {"role": "system", "content": (
                    "Du bist ein hilfreicher Assistent. Bereinige den folgenden Text, indem du alle unnötigen Zeichen "
                    "wie Sondersymbole, doppelte Leerzeichen, zufällige Zahlen oder Strings entfernst. Korrigiere Rechtschreibfehler, "
                    "überprüfe die Großschreibung und formatiere den Text für bessere Lesbarkeit. Entferne alles, was nicht relevant "
                    "für den Inhalt ist (z.B. Zeitstempel oder irrelevante Metadaten), während die Bedeutung und Struktur "
                    "des Originaltexts erhalten bleibt. Umgib deine Ausgabe mit <output>-Tags."
                )},
                {"role": "user", "content": text},
            ],
            stream=False
        )
        raw_response = response.choices[0].message.content
        corrected_text = extract_output_content(raw_response)
        logging.debug(f"Korrigierter Text: {corrected_text}")
        return corrected_text
    except openai.error.RateLimitError as e:
        logging.error(f"RateLimitError mit der DeepSeek API: {e}")
        raise  # Re-raise für den Decorator
    except openai.error.OpenAIError as e:
        logging.error(f"DeepSeek API Fehler: {e}")
        raise  # Re-raise für den Decorator
    except Exception as e:
        logging.error(f"Unerwarteter Fehler während der LLM-Korrektur: {e}")
        raise  # Re-raise für den Decorator

def process_and_correct_text(
    video_path: str,
    api_key: str,
    max_tokens: int = 7000,
    tesseract_path: Optional[str] = None,
    frame_interval: float = 1.0,
    scale_factor: int = 2,
    motion_threshold: float = 0.05,
    max_workers: Optional[int] = 4,
    supported_formats: Optional[List[str]] = None,
    adaptive_threshold: bool = False
) -> str:
    """
    Verarbeitet das Video, extrahiert und korrigiert den Text.

    :param video_path: Pfad zur Videodatei.
    :param api_key: API-Schlüssel für die DeepSeek API.
    :param max_tokens: Maximale Anzahl von Tokens pro Text-Chunk (inklusive Puffer).
    :param tesseract_path: Benutzerdefinierter Pfad zur Tesseract-OCR ausführbaren Datei.
    :param frame_interval: Zeitintervall in Sekunden zur Auswahl der Frames.
    :param scale_factor: Faktor zur Skalierung der Frames.
    :param motion_threshold: Schwellenwert für die Bewegungserkennung.
    :param max_workers: Maximale Anzahl von Arbeitern für parallele Verarbeitung.
    :param supported_formats: Liste der unterstützten Videoformate.
    :param adaptive_threshold: Ob adaptives Thresholding verwendet werden soll.
    :return: Finaler korrigierter Text.
    """
    set_tesseract_path(tesseract_path=tesseract_path)  # Korrigierte Übergabe des Arguments
    extracted_text = process_video_optimized(
        video_path=video_path,
        frame_interval=frame_interval,
        scale_factor=scale_factor,
        motion_threshold=motion_threshold,
        max_workers=max_workers,
        supported_formats=supported_formats,
        adaptive_threshold=adaptive_threshold
    )

    if not extracted_text:
        logging.warning("Kein Text extrahiert. Der korrigierte Text wird leer sein.")
        return ""

    corrected_chunks = []
    current_chunk = ""

    logging.info("Starte Textkorrektur mit LLM.")

    with ThreadPoolExecutor(max_workers=4) as executor:  # Begrenzung der max_workers auf 4
        futures = []
        for frame_num, text in extracted_text:
            sentences = re.split(r'(?<=[.!?])\s+', text)  # Satzbasierte Trennung
            for sentence in sentences:
                if not sentence:
                    continue
                token_count = estimate_tokens(current_chunk + " " + sentence) + 500
                if token_count > max_tokens:
                    if current_chunk.strip():
                        futures.append(executor.submit(correct_text_with_llm, current_chunk.strip(), api_key))
                        logging.debug("Text-Chunk zur Korrektur eingereicht.")
                    current_chunk = sentence  # Starte einen neuen Chunk mit dem aktuellen Satz
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

        if current_chunk.strip():
            futures.append(executor.submit(correct_text_with_llm, current_chunk.strip(), api_key))
            logging.debug("Letzter Text-Chunk zur Korrektur eingereicht.")

        for future in as_completed(futures):
            try:
                corrected_text = future.result()
                # Überprüfe, ob die Token-Anzahl innerhalb des Limits liegt
                if estimate_tokens(corrected_text) <= max_tokens:
                    corrected_chunks.append(corrected_text)
                    logging.debug("Ein Text-Chunk wurde korrigiert und hinzugefügt.")
                else:
                    logging.warning("Korrigierter Text überschreitet das maximale Token-Limit und wurde nicht hinzugefügt.")
            except Exception as e:
                logging.error(f"Fehler bei der Korrektur eines Text-Chunks: {e}")

    final_text = " ".join(corrected_chunks)
    logging.info("Textkorrektur abgeschlossen.")
    return final_text

def validate_api_key(api_key: str) -> bool:
    """
    Validiere den API-Schlüssel durch einen Testaufruf.

    :param api_key: Zu validierender API-Schlüssel.
    :return: True, wenn der Schlüssel gültig ist, sonst False.
    """
    if not api_key:
        logging.error("API-Schlüssel ist nicht gesetzt.")
        return False
    # Optional: Führe einen Testaufruf durch, um die Gültigkeit des Schlüssels zu überprüfen
    try:
        openai.api_key = api_key
        openai.Model.list(limit=1)
        return True
    except openai.error.AuthenticationError:
        logging.error("Ungültiger API-Schlüssel für die DeepSeek API.")
    except Exception as e:
        logging.error(f"Fehler bei der Validierung des API-Schlüssels: {e}")
    return False

def configure_logging(log_level: str = "INFO", log_file: str = "video_processing.log"):
    """
    Konfiguriert die Logging-Einstellungen.

    :param log_level: Das Logging-Level als String.
    :param log_file: Der Dateipfad für die Logdatei.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Bestehende Handler entfernen, um doppelte Logs zu vermeiden
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating File Handler (DEBUG und höher ins File)
    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def main():
    """
    Hauptfunktion zur Verarbeitung der Kommandozeilenargumente und Start des Text-Extraktions- und Korrekturprozesses.
    """
    parser = argparse.ArgumentParser(description="Video-Text-Extraktion und -Korrektur")
    parser.add_argument('video_path', type=str, help='Pfad zur Videodatei')
    parser.add_argument('output_text_path', type=str, help='Pfad zur Ausgabedatei für den korrigierten Text')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Setze das Logging-Level')
    parser.add_argument('--frame-interval', type=float, default=1.0, help='Zeitintervall in Sekunden zur Auswahl der Frames')
    parser.add_argument('--scale-factor', type=int, default=2, help='Faktor zur Skalierung der Frames (Upscaling)')
    parser.add_argument('--motion-threshold', type=float, default=0.05, help='Schwellenwert für die Bewegungserkennung')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximale Anzahl von Arbeitern für die parallele Verarbeitung')
    parser.add_argument('--supported-formats', type=str, nargs='*', default=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'], help='Liste der unterstützten Videoformate')
    parser.add_argument('--max-tokens', type=int, default=7000, help='Maximale Anzahl von Tokens pro Text-Chunk (inklusive Puffer)')
    parser.add_argument('--tesseract-path', type=str, default=None, help='Benutzerdefinierter Pfad zur Tesseract-OCR ausführbaren Datei')
    parser.add_argument('--adaptive-threshold', action='store_true', help='Verwende adaptives Thresholding statt Otsu')
    args = parser.parse_args()

    # Konfiguriere Logging basierend auf dem Kommandozeilenargument
    configure_logging(log_level=args.log_level)

    api_key = os.getenv("DEEPSEEK_API_KEY")  # Sicheres Abrufen des API-Schlüssels aus einer Umgebungsvariable

    if not validate_api_key(api_key):
        logging.error("Bitte setze die Umgebungsvariable 'DEEPSEEK_API_KEY' mit deinem tatsächlichen DeepSeek API-Schlüssel.")
        sys.exit(1)

    try:
        final_text = process_and_correct_text(
            video_path=args.video_path,
            api_key=api_key,
            max_tokens=args.max_tokens,
            tesseract_path=args.tesseract_path,
            frame_interval=args.frame_interval,
            scale_factor=args.scale_factor,
            motion_threshold=args.motion_threshold,
            max_workers=args.max_workers,
            supported_formats=args.supported_formats,
            adaptive_threshold=args.adaptive_threshold
        )

        with open(args.output_text_path, 'w', encoding='utf-8') as f:
            f.write(final_text)

        logging.info(f"Verarbeitung abgeschlossen. Korrigierter Text wurde in '{args.output_text_path}' gespeichert.")
    except FileNotFoundError as e:
        logging.error(f"Datei nicht gefunden: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fehler im Hauptprozess: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
