import json
import logging
import re
from typing import Dict, List
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class VideoSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", 
                 min_sentences: int = 3, max_sentences: int = 5):
        """
        Initialize summarization pipeline.
        
        Args:
            model_name: HuggingFace summarization model
            min_sentences: Minimum sentences in summary
            max_sentences: Maximum sentences in summary
        """
        self.summarizer = pipeline("summarization", model=model_name)
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.logger = logging.getLogger("VideoSummarizer")

    def generate_summary(self, frame_data: List[Dict], detection_data: List[Dict], 
                        transcript: str = None) -> str:
        """
        Generate coherent video summary from multiple data sources.
        
        Args:
            frame_data: List of frame descriptions [{"description": str, "frame": str}]
            detection_data: List of detection results [{"detections": list, "frame": str}]
            transcript: Optional audio transcription text
            
        Returns:
            str: Generated summary
        """
        try:
            # 1. Preprocess all text inputs
            textual_data = self._prepare_textual_data(frame_data, detection_data, transcript)
            
            # 2. Timeline analysis - cluster events by time segments
            timeline_summary = self._analyze_timeline(frame_data, detection_data)
            
            # 3. Key object analysis
            object_analysis = self._analyze_objects(detection_data)
            
            # 4. Generate AI summary
            full_context = f"""
            Video Content Timeline:
            {timeline_summary}
            
            Key Objects Detected:
            {object_analysis}
            
            Frame Descriptions:
            {textual_data['frame_text']}
            
            {"Transcript: " + textual_data['transcript_text'] if transcript else ""}
            """
            
            summary = self._summarize_with_ai(full_context)
            return self._postprocess_summary(summary)
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {str(e)}")
            return "Could not generate summary due to processing error."

    def _prepare_textual_data(self, frame_data, detection_data, transcript) -> Dict:
        """Combine and preprocess all text inputs"""
        frame_text = ". ".join([f"Frame {f['frame']}: {f['description']}" 
                              for f in frame_data])
        
        # Extract key detection info
        detection_text = []
        for det in detection_data:
            if det['detections']:
                objects = ", ".join(set([d['class'] for d in det['detections']]))
                detection_text.append(
                    f"Frame {det['frame']} contained: {objects}"
                )
                
        return {
            'frame_text': frame_text,
            'detection_text': ". ".join(detection_text),
            'transcript_text': transcript if transcript else ""
        }

    def _analyze_timeline(self, frame_data, detection_data) -> str:
        """Cluster frames into temporal segments"""
        try:
            # Use frame descriptions as features
            texts = [f['description'] for f in frame_data]
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            # Cluster into 3 temporal segments
            n_clusters = min(3, len(texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Build timeline summary
            segments = []
            for cluster_id in range(n_clusters):
                cluster_frames = [frame_data[i] for i in range(len(frame_data)) 
                                if clusters[i] == cluster_id]
                segment_text = self._summarize_with_ai(
                    " ".join([f['description'] for f in cluster_frames]),
                    max_length=100
                )
                segments.append(f"Segment {cluster_id+1}: {segment_text}")
                
            return "\n".join(segments)
            
        except Exception as e:
            self.logger.warning(f"Timeline analysis failed: {str(e)}")
            return ""

    def _analyze_objects(self, detection_data) -> str:
        """Analyze detected objects across frames"""
        try:
            all_objects = [d['class'] for det in detection_data 
                         for d in det['detections']]
            
            if not all_objects:
                return "No significant objects detected"
                
            unique_objects = list(set(all_objects))
            counts = {obj: all_objects.count(obj) for obj in unique_objects}
            sorted_objects = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            
            analysis = ["The video contains:"]
            for obj, count in sorted_objects[:5]:  # Top 5 objects
                analysis.append(f"- {obj} (appeared in {count} frames)")
                
            return "\n".join(analysis)
            
        except Exception as e:
            self.logger.warning(f"Object analysis failed: {str(e)}")
            return ""

    def _summarize_with_ai(self, text: str, max_length: int = None) -> str:
        """Generate summary using the configured model"""
        if not max_length:
            max_length = self.max_sentences * 25  # ~25 words per sentence
            
        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=max(10, self.min_sentences * 15),
            do_sample=False
        )
        return result[0]['summary_text']

    def _postprocess_summary(self, summary: str) -> str:
        """Clean and format the final summary"""
        # Remove incomplete sentences at end
        summary = re.sub(r'[^.]*$', '', summary).strip()
        
        # Capitalize properly
        sentences = [s.strip().capitalize() for s in summary.split('.') if s.strip()]
        
        # Join with proper punctuation
        return '. '.join(sentences) + '.' if sentences else summary
