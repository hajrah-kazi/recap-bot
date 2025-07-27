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
    def __init__(self, confidence_thresh: float = 0.25):
        model_path = os.path.join("Models", "yolov8n.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found! Please download and place it correctly.")
        self.model = YOLO(model_path)
        self.confidence_thresh = confidence_thresh
        # Store class names ONCE for reference
        self.class_names = self.model.names
        logger.info(f"Loaded YOLO model from {model_path} with confidence threshold {confidence_thresh}")

    def detect_objects(self, frame_path: str, classes: Optional[List[str]] = None) -> List[Dict]:
        """
        Detect objects in a frame with optional class filtering.
        Returns:
            list of dicts: { 'box': [x1, y1, x2, y2], 'confidence': float, 'class': str }
        """
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Couldn't load image at {frame_path}")
            results = self.model(frame)[0]
            detections = []

            # Debug: print raw boxes tensor
            logger.debug(f"Raw YOLO output for {frame_path}: {results.boxes.data}")

            # Loop over boxes (just like your working script)
            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls_idx = box.tolist()
                class_id = int(cls_idx)
                class_name = self.class_names.get(class_id, str(class_id))
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

    def draw_detections(self, frame_path: str, output_path: str, classes: Optional[List[str]] = None) -> bool:
        """
        Draw bounding boxes on detected objects and save the annotated image.
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
        Process video to detect objects at specified frame intervals.
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
