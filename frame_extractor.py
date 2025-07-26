import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class FrameExtractor:
    def __init__(self, video_path, output_dir="output", frame_interval=30, similarity_threshold=0.85):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.similarity_threshold = similarity_threshold
        self._setup_dirs()

    def _setup_dirs(self):
        os.makedirs(f"{self.output_dir}/frames_unique", exist_ok=True)

    def extract_unique_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        saved_count = 0
        prev_gray = None
        unique_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is None or ssim(prev_gray, gray) < self.similarity_threshold:
                    fname = f"frame_{saved_count:04d}.jpg"
                    fpath = os.path.join(self.output_dir, "frames_unique", fname)
                    cv2.imwrite(fpath, frame)
                    unique_frames.append(fpath)
                    prev_gray = gray
                    saved_count += 1

            frame_count += 1

        cap.release()
        return unique_frames