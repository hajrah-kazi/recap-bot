import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ObjectDetector")

class ObjectDetector:
    def __init__(self, confidence_thresh: float = 1.0):  # Set default value for confidence_thresh
        model_path = os.path.join("Models", "yolov8n.pt")
        
        assert os.path.exists(model_path), f"Model file {model_path} not found!"
        
        self.model = YOLO(model_path)
        self.confidence_thresh = confidence_thresh
        
        logger.info(f"Loaded YOLO model from {model_path}")

    def detect_objects(self, frame_path: str, classes: Optional[List[str]] = None) -> List[Dict]:
        """
        Detect objects in a frame with optional class filtering.
        
        Args:
            frame_path (str): Path to input image
            classes (list): List of class names to filter (e.g., ["person", "car"])
                           If None, returns all detected classes.

        Returns:
            list: Detections as dictionaries with:
                - 'box': [x1, y1, x2, y2]
                - 'confidence': float (0-1)
                - 'class': str (class name)
        """
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Couldn't load image at {frame_path}")
            
            results = self.model(frame)[0]
            detections = []

            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls_idx = box.tolist()
                class_name = results.names[int(cls_idx)]

                if (conf >= self.confidence_thresh and 
                    (classes is None or class_name in classes)):
                    
                    detections.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(float(conf), 4),
                        "class": class_name
                    })

            logger.info(f"Detected {len(detections)} objects in {frame_path}")
            return detections

        except Exception as e:
            logger.error(f"Detection failed for {frame_path}: {str(e)}")
            return []

    def draw_detections(self, frame_path: str, output_path: str, 
                       classes: Optional[List[str]] = None) -> bool:
        """
        Draw bounding boxes on frame and save to output path.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Invalid image at {frame_path}")

            detections = self.detect_objects(frame_path, classes)
            
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{det['class']} {det['confidence']:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.imwrite(output_path, frame)
            return True

        except Exception as e:
            logger.error(f"Failed to draw detections: {str(e)}")
            return False

    def process_video(self, video_path: str, output_dir: str, frame_interval: int = 30,
                      classes: Optional[List[str]] = None) -> List[Dict]:
        """
        Process video to detect objects at specified intervals.
        
        Returns:
            list: Aggregated detections across all processed frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Couldn't open video {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        all_detections = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                temp_frame_path = os.path.join(output_dir, f"temp_{frame_count}.jpg")
                cv2.imwrite(temp_frame_path, frame)
                
                frame_detections = self.detect_objects(temp_frame_path, classes)
                all_detections.extend(frame_detections)
                
                # Save visualization
                output_path = os.path.join(output_dir, f"detected_{frame_count}.jpg")
                self.draw_detections(temp_frame_path, output_path, classes)
                
                os.remove(temp_frame_path)

            frame_count += 1

        cap.release()
        return all_detections
