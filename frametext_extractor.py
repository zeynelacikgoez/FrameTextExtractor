import cv2
import numpy as np
import pytesseract
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Tuple, Optional
import openai
import re
import sys
import argparse  # For command line arguments
import tiktoken
from functools import wraps
import time
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# Optional: Decorator to retry on certain exceptions
def retry_on_exception(max_retries=3, delay=2, exceptions=(Exception,)):
    """
    A decorator that retries a function upon certain exceptions.

    :param max_retries: Maximum number of retry attempts.
    :param delay: Delay between attempts in seconds.
    :param exceptions: Tuple of exceptions to handle.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    logging.warning(f"Exception {e} in {func.__name__}. Attempt {retries}/{max_retries} after {delay} seconds.")
                    time.sleep(delay)
            logging.error(f"Max number of attempts ({max_retries}) exceeded for {func.__name__}.")
            raise
        return wrapper
    return decorator

def set_tesseract_path(tesseract_path: Optional[str] = None):
    """
    Sets the path for Tesseract-OCR based on environment variable, CLI argument, oder default paths.
    Raises FileNotFoundError if Tesseract is not found.

    :param tesseract_path: Optional custom path to Tesseract-OCR executable.
    """
    # Check environment variable first
    tesseract_env_path = os.getenv("TESSERACT_PATH")
    if tesseract_env_path and os.path.exists(tesseract_env_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_env_path
        logging.info(f"Using custom Tesseract path from environment: {tesseract_env_path}")
        return

    # Then check CLI argument
    if tesseract_path:
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logging.info(f"Using custom Tesseract path from CLI argument: {tesseract_path}")
            return
        else:
            logging.error(f"Tesseract not found at the provided CLI path: {tesseract_path}")
            raise FileNotFoundError(f"Tesseract not found at the provided path: {tesseract_path}")

    # Fallback strategy
    if os.name == 'nt':
        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win_path):
            pytesseract.pytesseract.tesseract_cmd = default_win_path
            logging.info(f"Tesseract path set to default Windows location: {default_win_path}")
            return
        else:
            raise FileNotFoundError(f"Tesseract not found at the default Windows location: {default_win_path}")
    else:
        tesseract_shutil = shutil.which("tesseract")
        if tesseract_shutil:
            pytesseract.pytesseract.tesseract_cmd = tesseract_shutil
            logging.info(f"Tesseract found in PATH: {tesseract_shutil}")
            return
        else:
            raise FileNotFoundError("Tesseract-OCR not found in PATH on this system.")

def extract_text_with_pytesseract(frame: np.ndarray) -> str:
    """
    Extracts text from a given frame using Tesseract-OCR.

    :param frame: The image frame from which to extract text.
    :return: Extracted text as a string.
    """
    config = '--psm 6 --oem 1'
    try:
        text = pytesseract.image_to_string(frame, config=config)
        logging.debug(f"Extracted text: {text}")
        return text
    except pytesseract.TesseractError as e:
        logging.error(f"Tesseract OCR error: {e}")
        return ""
    except Exception as e:
        logging.error(f"Unexpected error during OCR: {e}")
        return ""

def detect_movement(prev_frame: np.ndarray, curr_frame: np.ndarray, base_threshold: float = 0.05) -> bool:
    """
    Detects movement between two consecutive frames.

    :param prev_frame: The previous grayscale frame.
    :param curr_frame: The current grayscale frame.
    :param base_threshold: Base threshold for motion detection.
    :return: True if movement is detected, otherwise False.
    """
    diff = cv2.absdiff(prev_frame, curr_frame)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size
    average_intensity = np.mean(diff) / 255  # Normalized to [0,1]

    # Dynamic threshold based on average intensity
    dynamic_threshold = base_threshold * (1 + average_intensity)
    movement = (non_zero_count / total_pixels) > dynamic_threshold
    logging.debug(f"Movement detected: {movement} (Threshold: {dynamic_threshold:.4f}, Avg intensity: {average_intensity:.4f})")
    return movement

def preprocess_frame(frame: np.ndarray, scale_factor: int = 2, adaptive_threshold: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Processes a frame by scaling, converting to grayscale, and applying a threshold.

    :param frame: The original image frame.
    :param scale_factor: Factor to reduce the size of the frame.
    :param adaptive_threshold: Whether to use adaptive thresholding instead of Otsu.
    :return: Tuple of the binarized frame and the grayscale frame.
    """
    try:
        resized_frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        if adaptive_threshold:
            thresh_frame = cv2.adaptiveThreshold(
                gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            logging.debug("Applied adaptive thresholding.")
        else:
            _, thresh_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logging.debug("Applied Otsu's thresholding.")
        return thresh_frame, gray_frame
    except cv2.error as e:
        logging.error(f"OpenCV error during preprocessing: {e}")
        logging.debug(f"Frame dimensions: {frame.shape}, dtype: {frame.dtype}")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error during frame preprocessing: {e}")
        return None, None

def process_frame(thresh_frame: np.ndarray) -> str:
    """
    Performs text extraction on a preprocessed frame.

    :param thresh_frame: The binarized image frame.
    :return: Extracted text as a string.
    """
    if thresh_frame is not None:
        return extract_text_with_pytesseract(thresh_frame)
    return ""

def estimate_tokens(text: str, model: str = "deepseek-chat") -> int:
    """
    Estimates the number of tokens in a given text based on the used model.

    :param text: The text to be analyzed.
    :param model: The model used for tokenization.
    :return: Estimated number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.error(f"Unknown model for token estimation: {model}")
        raise
    return len(encoding.encode(text))

def process_video_optimized(
    video_path: str,
    frame_interval: float = 1.0,  # Time interval in seconds
    scale_factor: int = 2,
    motion_threshold: float = 0.05,
    max_workers: Optional[int] = 4,  # Set default to 4
    supported_formats: Optional[List[str]] = None,  # Extend supported formats
    adaptive_threshold: bool = False  # Option to use adaptive thresholding
) -> List[Tuple[int, str]]:
    """
    Processes a video, extracts text from relevant frames based on motion detection.

    :param video_path: Path to the video file.
    :param frame_interval: Time interval in seconds for selecting frames.
    :param scale_factor: Factor to reduce the size of the frames.
    :param motion_threshold: Threshold for motion detection.
    :param max_workers: Maximum number of workers for parallel processing.
    :param supported_formats: List of supported video formats.
    :param adaptive_threshold: Whether to use adaptive thresholding.
    :return: List of tuples (frame number, extracted text).
    """
    if supported_formats is None:
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

    extracted_text: List[Tuple[int, str]] = []

    try:
        if not os.path.exists(video_path):
            logging.error(f"Video file '{video_path}' does not exist.")
            return extracted_text

        if not os.path.isfile(video_path):
            logging.error(f"'{video_path}' is not a valid file.")
            return extracted_text

        # Check the video format
        _, ext = os.path.splitext(video_path)
        if ext.lower() not in supported_formats:
            logging.error(f"Unsupported video format '{ext}'. Supported formats are: {supported_formats}")
            return extracted_text

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logging.error("Error opening the video file.")
                return extracted_text

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                logging.error("Could not determine the video's FPS.")
                return extracted_text

            frame_interval_frames = max(1, int(fps * frame_interval))

            logging.info(f"Video FPS: {fps}")
            logging.info(f"Processing every {frame_interval_frames}th frame (equivalent to {frame_interval} seconds).")

            frame_num = 0
            frames_to_process = []
            prev_gray = None  # Initialization for motion detection

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
                            logging.debug(f"Frame {frame_num} hinzugefügt für OCR-Verarbeitung.")
                        prev_gray = gray_frame

                frame_num += 1

            logging.info(f"Alle relevanten Frames ({len(frames_to_process)}) gesammelt für OCR-Verarbeitung.")

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
                            logging.debug(f"OCR abgeschlossen für Frame {frame_num}.")
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
    Extracts the content inside the <output> tags from the given text.

    :param text: The full text to be analyzed.
    :return: Concatenated content inside the <output> tags or the full text if no tags are found.
    """
    # Robust extraction of <output> tags
    matches = re.findall(r'<output>(.*?)</output>', text, re.DOTALL | re.IGNORECASE)
    if matches:
        # Mehrere Blöcke zusammenfassen
        return "\n".join(m.strip() for m in matches)
    else:
        # Versuch, den vollständigen Text zu verwenden
        logging.warning("Keine <output>-Tags in der LLM-Antwort gefunden. Versuch, den vollständigen Text zu verwenden.")
        return text.strip()

@retry_on_exception(max_retries=3, delay=2, exceptions=(openai.error.RateLimitError, openai.error.OpenAIError,))
def correct_text_with_llm(text: str, api_key: str) -> str:
    """
    Corrects the given text using an LLM (Language Model) and returns the cleaned text.

    :param text: The text to be corrected.
    :param api_key: API key for the DeepSeek API.
    :return: Corrected text as a string.
    """
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="deepseek-chat",  # **IMPORTANT: Use the correct model**
            messages=[
                {"role": "system", "content": (
                    "You are a helpful assistant. Clean up the following text by removing all unnecessary characters "
                    "such as special symbols, double spaces, random numbers, or strings. Correct any spelling errors, "
                    "check the capitalization, and format the text for readability. Remove anything that is not relevant "
                    "to the content (e.g., timestamps or irrelevant metadata), while preserving the meaning and structure "
                    "of the original text. Wrap your output in <output> tags."
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
        raise  # Re-raises the exception for the decorator to handle
    except openai.error.OpenAIError as e:
        logging.error(f"DeepSeek API Fehler: {e}")
        raise  # Re-raises the exception for the decorator to handle
    except Exception as e:
        logging.error(f"Unerwarteter Fehler während der LLM-Korrektur: {e}")
        raise  # Re-raises the exception

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
    Processes the video, extracts and corrects the text.

    :param video_path: Path to the video file.
    :param api_key: API key for the DeepSeek API.
    :param max_tokens: Maximum number of tokens per text chunk (including buffer).
    :param tesseract_path: Custom path to Tesseract-OCR executable.
    :param frame_interval: Time interval in seconds for selecting frames.
    :param scale_factor: Factor to reduce the size of the frames.
    :param motion_threshold: Threshold for motion detection.
    :param max_workers: Maximum number of workers for parallel processing.
    :param supported_formats: List of supported video formats.
    :param adaptive_threshold: Whether to use adaptive thresholding.
    :return: Final corrected text.
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
                    current_chunk = sentence  # Start a new chunk mit dem aktuellen Satz
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
                # Check if token count is within the limit
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
    Validates the API key by making a test call.

    :param api_key: API key to validate.
    :return: True if the key is valid, otherwise False.
    """
    if not api_key:
        logging.error("API-Schlüssel ist nicht gesetzt.")
        return False
    # Optional: Make a test call to verify the key's validity
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
    Configures the logging settings.

    :param log_level: The logging level as a string.
    :param log_file: The file path for the log file.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    ch = logging.StreamHandler()
    # Setze den StreamHandler ebenfalls auf numeric_level 
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (DEBUG und höher ins File)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def main():
    """
    Main function to process command line arguments and start the text extraction and correction process.
    """
    parser = argparse.ArgumentParser(description="Video-Text-Extraktion und -Korrektur")
    parser.add_argument('video_path', type=str, help='Pfad zur Videodatei')
    parser.add_argument('output_text_path', type=str, help='Pfad zur Ausgabedatei für den korrigierten Text')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Setze das Logging-Level')
    parser.add_argument('--frame-interval', type=float, default=1.0, help='Zeitintervall in Sekunden zur Auswahl der Frames')
    parser.add_argument('--scale-factor', type=int, default=2, help='Faktor zur Reduzierung der Größe der Frames')
    parser.add_argument('--motion-threshold', type=float, default=0.05, help='Schwellenwert für die Bewegungserkennung')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximale Anzahl von Arbeitern für die parallele Verarbeitung')  # Standard auf 4 gesetzt
    parser.add_argument('--supported-formats', type=str, nargs='*', default=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'], help='Liste der unterstützten Videoformate')
    parser.add_argument('--max-tokens', type=int, default=7000, help='Maximale Anzahl von Tokens pro Text-Chunk (inklusive Puffer)')
    parser.add_argument('--tesseract-path', type=str, default=None, help='Benutzerdefinierter Pfad zur Tesseract-OCR ausführbaren Datei')
    parser.add_argument('--adaptive-threshold', action='store_true', help='Verwende adaptive Schwellenwertsetzung statt Otsu')
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
