import cv2
import numpy as np
import easyocr
from typing import List, Dict
import os

class ContourDetector:
    """
    Detects number plates using contour-based approach.
    Alternative to YOLO for comparison and reliability.
    """
    
    def __init__(self):
        """Initialize the contour detector with EasyOCR reader."""
        # Initialize EasyOCR reader for text extraction
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("Contour Detector initialized successfully")
    
    def detect_and_extract_from_array(self, image: np.ndarray) -> List[Dict]:
        """
        Detect number plates using contour method from numpy array.
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of dictionaries containing detected plates and extracted text
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image array")
        
        return self._find_plates(image)
    
    def detect_and_extract(self, image_path: str) -> List[Dict]:
        """
        Detect number plates using contour method from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of dictionaries containing detected plates and extracted text
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        return self._find_plates(image)
    
    def _find_plates(self, image: np.ndarray) -> List[Dict]:
        """
        Find plates using contour-based approach.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of detected plates
        """
        results = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply edge detection
        edges = cv2.Canny(filtered, 30, 200)
        
        # Apply morphological operations to improve edge connectivity
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return results
        
        # Filter and process contours
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size and aspect ratio
            # License plates typically have aspect ratio between 2:1 and 5:1
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # Check minimum size and aspect ratio
            min_area = 500
            max_area = image.shape[0] * image.shape[1] * 0.3
            
            if area < min_area or area > max_area:
                continue
            
            if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                continue
            
            # Ensure coordinates are within bounds
            x = max(0, x)
            y = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)
            
            # Extract ROI
            roi = image[y:y2, x:x2]
            
            if roi.size == 0:
                continue
            
            # Perform OCR
            try:
                ocr_results = self.reader.readtext(roi, detail=1)
                
                # Extract and clean text
                detected_text = self._extract_and_clean_text(ocr_results)
                
                if detected_text:  # Only include if text was detected
                    results.append({
                        'bounding_box': {
                            'x1': x,
                            'y1': y,
                            'x2': x2,
                            'y2': y2
                        },
                        'detected_text': detected_text,
                        'confidence': float(ocr_results[0][2]) if ocr_results else 0.0,
                        'roi_image': roi,
                        'method': 'contour'
                    })
            except Exception as e:
                print(f"OCR Error on ROI: {e}")
        
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
        cleaned = combined_text.replace(" ", "").replace("-", "").upper()
        
        # Keep only alphanumeric characters
        cleaned = "".join(c for c in cleaned if c.isalnum())
        
        return cleaned if len(cleaned) >= 6 else ""
    
    def get_processing_steps(self, image: np.ndarray) -> Dict:
        """
        Get intermediate processing steps for visualization/debugging.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with processing steps
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 30, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return {
            'original': image,
            'grayscale': gray,
            'filtered': filtered,
            'edges': edges,
            'closed': closed
        }
    
    def visualize_detections(self, image_path: str, detections: List[Dict]) -> None:
        """
        Visualize detected number plates on image.
        
        Args:
            image_path: Path to image
            detections: List of detections
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
                (255, 0, 0),  # Blue color for contour
                2
            )
            
            # Put text label
            cv2.putText(
                image,
                text,
                (bbox['x1'], bbox['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
        
        # Display image
        cv2.imshow("Contour Detection Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
