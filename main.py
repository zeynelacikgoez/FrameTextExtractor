import cv2
import numpy as np
import pytesseract
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_tesseract_path():
    if os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_with_pillow(frame: np.ndarray) -> str:
    image = Image.fromarray(frame)
    config = '--psm 6 --oem 1'
    return pytesseract.image_to_string(image, config=config)

def detect_movement(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: float = 0.05) -> bool:
    diff = cv2.absdiff(prev_frame, curr_frame)
    return np.mean(diff) > threshold * 255

def process_frame(frame: np.ndarray) -> Tuple[np.ndarray, str]:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame, extract_text_with_pillow(gray_frame)

def process_video_optimized(video_path: str, output_text: str):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logging.error("Fehler beim Öffnen der Videodatei.")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    
    prev_frame = None
    extracted_text: List[str] = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for _ in range(0, frame_count, fps):  # Process one frame per second
            cap.set(cv2.CAP_PROP_POS_FRAMES, _)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (width, height))
            futures.append(executor.submit(process_frame, frame))
        
        for future in futures:
            gray_frame, text = future.result()
            
            if prev_frame is not None and detect_movement(prev_frame, gray_frame):
                if text.strip():
                    extracted_text.append(text.strip())
            
            prev_frame = gray_frame
    
    with open(output_text, 'w', encoding='utf-8') as f:
        f.write("\n".join(extracted_text))
    
    cap.release()
    logging.info(f"Videoverarbeitung abgeschlossen. Extrahierter Text in {output_text} gespeichert.")

if __name__ == "__main__":
    set_tesseract_path()
    video_path = "video.mp4"
    output_text = "optimized_extracted_text.txt"
    process_video_optimized(video_path, output_text)