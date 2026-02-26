import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from typing import Tuple, List, Dict
import os

class NumberPlateDetector:
    """
    Detects and recognizes number plates using YOLO and EasyOCR.
    """
    
    def __init__(self, model_path: str = None, model_size: str = "n"):
        """
        Initialize the detector with YOLO model and EasyOCR reader.
        
        Args:
            model_path: Path to YOLO model weights (if None, uses local model)
            model_size: YOLOv8 model size - 'n' (nano), 's' (small), 'm' (medium)
        """
        # Load YOLO model from local path
        if model_path is None:
            # Use local YOLO model from workspace
            local_models = {
                'l':'license_plate_detector.pt',
                'b': 'best_bikee.pt',
                'n': 'yolov8n.pt',
                's': 'yolov8s.pt',
                'm': 'yolov8m.pt'
            }
            model_file = local_models.get(model_size, 'yolov8n.pt')
            model_path = os.path.join(os.path.dirname(__file__), model_file)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at: {model_path}")
        
        # Load YOLO model for general object detection
        # For number plate detection, we'll use a pre-trained YOLOv8 nano model
        # In production, use a model specifically trained on number plates
        self.yolo_model = YOLO(model_path)
        self.model_path = model_path
        
        # Initialize EasyOCR reader for text extraction
        # Using English recognizer (can add more languages if needed)
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Print model information
        print(f"Number Plate Detector initialized successfully")
        print(f"Model: {os.path.basename(model_path)}")
        print(f"Available Classes: {self.yolo_model.names}")
    
    def detect_and_extract_from_array(self, image: np.ndarray) -> List[Dict]:
        """
        Detect number plates and extract text from numpy array image.
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of dictionaries containing detected plates and extracted text
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image array")
        
        results = []
        
        # Run YOLO detection
        yolo_results = self.yolo_model(image, conf=0.5)
        
        # Extract plates and perform OCR
        for result in yolo_results:
            boxes = result.boxes
            
            for box in boxes:
                # Filter to only license_plate/numberplate detections
                # Skip other classes like Helmet, NoHelmet, etc
                class_id = int(box.cls[0]) if hasattr(box, 'cls') else None
                class_name = self.yolo_model.names.get(class_id, "") if class_id is not None else ""
                
                # Only process license plate detections (accept both 'numberplate' and 'license_plate')
                if class_name.lower() not in ["numberplate", "license_plate", "plate"]:
                    continue
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                
                # Extract region of interest (ROI)
                roi = image[y1:y2, x1:x2]
                
                # Perform OCR on the detected region
                try:
                    ocr_results = self.reader.readtext(roi, detail=1)
                    
                    # Extract and clean text
                    detected_text = self._extract_and_clean_text(ocr_results)
                    
                    if detected_text:  # Only include if text was detected
                        results.append({
                            'bounding_box': {
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2
                            },
                            'detected_text': detected_text,
                            'confidence': float(ocr_results[0][2]) if ocr_results else 0.0,
                            'roi_image': roi
                        })
                except Exception as e:
                    print(f"OCR Error on ROI: {e}")
        
        return results

    def detect_and_extract(self, image_path: str) -> List[Dict]:
        """
        Detect number plates and extract text from image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of dictionaries containing detected plates and extracted text
        """
        # Read image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        results = []
        
        # Run YOLO detection
        yolo_results = self.yolo_model(image, conf=0.5)
        
        # Extract plates and perform OCR
        for result in yolo_results:
            boxes = result.boxes
            
            for box in boxes:
                # Filter to only license_plate/numberplate detections
                class_id = int(box.cls[0]) if hasattr(box, 'cls') else None
                class_name = self.yolo_model.names.get(class_id, "") if class_id is not None else ""
                
                # Only process license plate detections (accept both 'numberplate' and 'license_plate')
                if class_name.lower() not in ["numberplate", "license_plate", "plate"]:
                    continue
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                
                # Extract region of interest (ROI)
                roi = image[y1:y2, x1:x2]
                
                # Perform OCR on the detected region
                try:
                    ocr_results = self.reader.readtext(roi, detail=1)
                    
                    # Extract and clean text
                    detected_text = self._extract_and_clean_text(ocr_results)
                    
                    if detected_text:  # Only include if text was detected
                        results.append({
                            'bounding_box': {
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2
                            },
                            'detected_text': detected_text,
                            'confidence': float(ocr_results[0][2]) if ocr_results else 0.0,
                            'roi_image': roi
                        })
                except Exception as e:
                    print(f"OCR Error on ROI: {e}")
        
        return results
    
    def detect_from_camera(self, camera_id: int = 0, timeout: int = 10) -> List[Dict]:
        """
        Detect number plates from camera feed.
        
        Args:
            camera_id: Camera device ID (default: 0)
            timeout: Timeout in seconds to capture frame
            
        Returns:
            List of detected plates and extracted text
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        results = []
        frame_count = 0
        max_frames = timeout * 30  # Assuming ~30 FPS
        
        print(f"Capturing from camera {camera_id} for {timeout} seconds...")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Run YOLO detection on frame
            yolo_results = self.yolo_model(frame, conf=0.5)
            
            for result in yolo_results:
                boxes = result.boxes
                
                for box in boxes:
                    # Filter to only license_plate/numberplate detections
                    class_id = int(box.cls[0]) if hasattr(box, 'cls') else None
                    class_name = self.yolo_model.names.get(class_id, "") if class_id is not None else ""
                    
                    # Only process license plate detections (accept both 'numberplate' and 'license_plate')
                    if class_name.lower() not in ["numberplate", "license_plate", "plate"]:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    roi = frame[y1:y2, x1:x2]
                    
                    try:
                        ocr_results = self.reader.readtext(roi, detail=1)
                        detected_text = self._extract_and_clean_text(ocr_results)
                        
                        if detected_text:
                            results.append({
                                'bounding_box': {
                                    'x1': x1,
                                    'y1': y1,
                                    'x2': x2,
                                    'y2': y2
                                },
                                'detected_text': detected_text,
                                'confidence': float(ocr_results[0][2]) if ocr_results else 0.0,
                                'timestamp': frame_count
                            })
                    except Exception as e:
                        print(f"OCR Error: {e}")
            
            frame_count += 1
            
            # Display frame with detections
            if yolo_results:
                annotated_frame = yolo_results[0].plot()
                cv2.imshow('Number Plate Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return results
    
    @staticmethod
    def _extract_and_clean_text(ocr_results) -> str:
        """
        Extract and clean text from EasyOCR results.
        
        Args:
            ocr_results: Results from EasyOCR readtext
            
        Returns:
            Cleaned text string
        """
        if not ocr_results:
            return ""
        
        # Combine all detected text
        text_parts = [result[1] for result in ocr_results]
        combined_text = "".join(text_parts)
        
        # Clean up: remove spaces and special characters, convert to uppercase
        # Indian number plates typically follow pattern: STATE_CODE + 2_DIGITS + LETTERS + 4_DIGITS
        cleaned = combined_text.replace(" ", "").replace("-", "").upper()
        
        # Keep only alphanumeric characters
        cleaned = "".join(c for c in cleaned if c.isalnum())
        
        return cleaned if len(cleaned) >= 6 else ""  # Minimum plate length
    
    def visualize_detections(self, image_path: str, detections: List[Dict]) -> None:
        """
        Visualize detected number plates on image.
        
        Args:
            image_path: Path to image
            detections: List of detections from detect_and_extract
        """
        image = cv2.imread(image_path)
        
        for detection in detections:
            bbox = detection['bounding_box']
            text = detection['detected_text']
            
            # Draw bounding box
            cv2.rectangle(
                image,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                (0, 255, 0),  # Green color
                2
            )
            
            # Put text label
            cv2.putText(
                image,
                text,
                (bbox['x1'], bbox['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Display image
        cv2.imshow("Number Plate Detection Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
