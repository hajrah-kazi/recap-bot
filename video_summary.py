import logging
import re
from typing import Dict, List, Optional
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

class VideoSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn",
                 min_sentences: int = 3, max_sentences: int = 5):
        """
        Initialize summarization pipeline.

        Args:
            model_name: HuggingFace summarization model name
            min_sentences: Minimum sentences in summary
            max_sentences: Maximum sentences in summary
        """
        try:
            # Check if CUDA is available for faster processing
            device = 0 if torch.cuda.is_available() else -1
            self.summarizer = pipeline("summarization", model=model_name, device=device)
        except Exception as e:
            # Fallback to a smaller model if the main one fails
            logging.warning(f"Failed to load {model_name}, falling back to distilbart: {str(e)}")
            try:
                self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
            except Exception as e2:
                logging.error(f"Failed to load fallback model: {str(e2)}")
                raise RuntimeError("Could not initialize summarization model")
        
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.logger = logging.getLogger("VideoSummarizer")

    def generate_summary(self,
                         frame_data: List[Dict],
                         detection_data: List[Dict],
                         transcript: Optional[str] = None) -> str:
        """
        Generate coherent video summary from multiple data sources.

        Args:
            frame_data: List of frame descriptions [{"description": str, "frame": str}]
            detection_data: List of detection results [{"detections": list, "frame": str}]
            transcript: Optional audio transcription text

        Returns:
            str: Generated summary text
        """
        try:
            self.logger.info(f"Generating summary from {len(frame_data)} frames, {len(detection_data)} detections")
            
            # Validate inputs
            if not frame_data and not detection_data and not transcript:
                return "No content available to summarize."

            # Prepare combined textual data
            textual_data = self._prepare_textual_data(frame_data, detection_data, transcript)
            
            # Check if we have any meaningful content
            if not any(textual_data.values()):
                return "Insufficient content available for summary generation."

            # Analyze timeline by clustering frames into segments
            timeline_summary = self._analyze_timeline(frame_data)

            # Analyze key objects detected
            object_analysis = self._analyze_objects(detection_data)

            # Build context sections
            context_sections = []
            
            if timeline_summary:
                context_sections.append(f"Video Content Timeline:\n{timeline_summary}")
            
            if object_analysis:
                context_sections.append(f"Key Objects and Elements:\n{object_analysis}")
            
            if textual_data['frame_text']:
                context_sections.append(f"Visual Content:\n{textual_data['frame_text']}")
            
            if textual_data['transcript_text']:
                context_sections.append(f"Audio Content:\n{textual_data['transcript_text']}")

            # Combine all context
            full_context = "\n\n".join(context_sections)
            
            if len(full_context.strip()) < 50:  # Too little content
                return "Insufficient content available for meaningful summary generation."

            # Generate summary with adaptive length
            summary = self._summarize_with_ai(full_context)

            # Post-process summary for clean formatting
            final_summary = self._postprocess_summary(summary)
            
            if not final_summary or len(final_summary.strip()) < 10:
                return self._create_fallback_summary(textual_data, object_analysis)
            
            return final_summary

        except Exception as e:
            self.logger.error(f"Summary generation failed: {str(e)}", exc_info=True)
            return self._create_fallback_summary(
                self._prepare_textual_data(frame_data, detection_data, transcript),
                self._analyze_objects(detection_data)
            )

    def _prepare_textual_data(self,
                              frame_data: List[Dict],
                              detection_data: List[Dict],
                              transcript: Optional[str]) -> Dict[str, str]:
        """Combine and preprocess textual data from frames, detections and transcript"""
        try:
            # Process frame descriptions
            frame_descriptions = []
            for i, f in enumerate(frame_data):
                if f.get('description') and f['description'].strip():
                    # Clean and format description
                    desc = f['description'].strip()
                    if not desc.endswith('.'):
                        desc += '.'
                    frame_descriptions.append(f"Scene {i+1}: {desc}")
            
            frame_text = " ".join(frame_descriptions)

            # Process detection data
            detection_summaries = []
            object_counts = {}
            
            for det in detection_data:
                if det.get('detections'):
                    frame_objects = []
                    for d in det['detections']:
                        if d.get('class'):
                            obj_class = d['class'].replace('_', ' ')
                            frame_objects.append(obj_class)
                            object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
                    
                    if frame_objects:
                        unique_objects = list(set(frame_objects))
                        if len(unique_objects) <= 3:
                            detection_summaries.append(f"Contains {', '.join(unique_objects)}")
                        else:
                            detection_summaries.append(f"Contains {', '.join(unique_objects[:3])} and others")

            detection_text = ". ".join(detection_summaries)

            # Process transcript
            transcript_text = ""
            if transcript and transcript.strip() and transcript != "Transcription unavailable":
                # Clean transcript
                transcript_clean = re.sub(r'\s+', ' ', transcript.strip())
                # Limit transcript length to avoid overwhelming the summarizer
                if len(transcript_clean) > 1000:
                    transcript_text = transcript_clean[:1000] + "..."
                else:
                    transcript_text = transcript_clean

            return {
                'frame_text': frame_text,
                'detection_text': detection_text,
                'transcript_text': transcript_text,
                'object_counts': object_counts
            }
        except Exception as e:
            self.logger.warning(f"Failed to prepare textual data: {str(e)}")
            return {
                'frame_text': '', 
                'detection_text': '', 
                'transcript_text': transcript if transcript else "",
                'object_counts': {}
            }

    def _analyze_timeline(self, frame_data: List[Dict]) -> str:
        """Cluster frame descriptions into temporal segments and summarize"""
        try:
            descriptions = [f.get('description', '').strip() for f in frame_data if f.get('description', '').strip()]
            if len(descriptions) < 2:
                if descriptions:
                    return f"Single scene: {descriptions[0]}"
                return ""

            # Use TF-IDF to find similar frames
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            X = vectorizer.fit_transform(descriptions)

            # Determine optimal number of clusters
            n_clusters = min(max(2, len(descriptions) // 3), 4)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)

            # Group descriptions by cluster
            segments = []
            for cluster_id in range(n_clusters):
                cluster_descriptions = [descriptions[i] for i in range(len(descriptions)) if clusters[i] == cluster_id]
                if cluster_descriptions:
                    # Take the first description as representative, or combine if short
                    if len(cluster_descriptions) == 1:
                        segment_desc = cluster_descriptions[0]
                    else:
                        combined = " ".join(cluster_descriptions)
                        if len(combined) < 200:
                            segment_desc = combined
                        else:
                            segment_desc = cluster_descriptions[0]  # Use first as representative
                    
                    segments.append(f"Segment {cluster_id + 1}: {segment_desc}")
            
            return "\n".join(segments)
        except Exception as e:
            self.logger.warning(f"Timeline analysis failed: {str(e)}")
            return ""

    def _analyze_objects(self, detection_data: List[Dict]) -> str:
        """Analyze occurrence and frequency of detected objects"""
        try:
            object_counts = {}
            total_frames = len(detection_data)
            
            for det in detection_data:
                frame_objects = set()  # Use set to count unique objects per frame
                for d in det.get('detections', []):
                    if d.get('class'):
                        obj_class = d['class'].replace('_', ' ')
                        frame_objects.add(obj_class)
                
                # Count each unique object once per frame
                for obj in frame_objects:
                    object_counts[obj] = object_counts.get(obj, 0) + 1

            if not object_counts:
                return "No significant objects detected."

            # Sort by frequency
            sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
            
            analysis_parts = []
            
            # Main objects (appeared in >30% of frames)
            main_objects = [obj for obj, count in sorted_objects if count > total_frames * 0.3]
            if main_objects:
                analysis_parts.append(f"Primary elements: {', '.join(main_objects[:4])}")
            
            # Secondary objects (appeared in 10-30% of frames)
            secondary_objects = [obj for obj, count in sorted_objects if total_frames * 0.1 < count <= total_frames * 0.3]
            if secondary_objects:
                analysis_parts.append(f"Secondary elements: {', '.join(secondary_objects[:3])}")
            
            return ". ".join(analysis_parts) if analysis_parts else "Various objects detected throughout the video."
            
        except Exception as e:
            self.logger.warning(f"Object analysis failed: {str(e)}")
            return ""

    def _summarize_with_ai(self, text: str, max_length: int = None) -> str:
        """Use HuggingFace summarization pipeline to generate the summary"""
        try:
            if not text or len(text.strip()) < 50:
                return ""

            input_length = len(text.split())
            
            # Set appropriate lengths based on input
            if not max_length:
                max_length = min(150, max(50, input_length // 3))
            
            min_length = min(30, max_length // 2)
            
            # Truncate input if too long (BART has token limits)
            if input_length > 1000:
                words = text.split()
                text = " ".join(words[:1000])

            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            
            if result and len(result) > 0:
                return result[0].get('summary_text', '')
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"AI summarization failed: {str(e)}")
            return ""

    def _postprocess_summary(self, summary: str) -> str:
        """Cleanup and nicely format the summary text"""
        if not summary:
            return ""
        
        # Clean up the summary
        summary = summary.strip()
        
        # Remove incomplete sentences at the end
        sentences = summary.split('.')
        complete_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Ignore very short fragments
                if not sentence[0].isupper():
                    sentence = sentence.capitalize()
                complete_sentences.append(sentence)
        
        if complete_sentences:
            result = '. '.join(complete_sentences) + '.'
            # Remove duplicate periods
            result = re.sub(r'\.+', '.', result)
            return result
        
        return summary

    def _create_fallback_summary(self, textual_data: Dict, object_analysis: str) -> str:
        """Create a basic summary when AI summarization fails"""
        try:
            parts = []
            
            if textual_data.get('transcript_text'):
                parts.append("This video contains spoken content")
            
            if object_analysis:
                parts.append(f"showing {object_analysis.lower()}")
            
            if textual_data.get('frame_text'):
                frame_count = len([s for s in textual_data['frame_text'].split('Scene') if s.strip()])
                parts.append(f"across {frame_count} distinct scenes")
            
            if parts:
                return ". ".join(parts).capitalize() + "."
            else:
                return "Video content processed successfully."
                
        except Exception as e:
            self.logger.error(f"Fallback summary creation failed: {str(e)}")
            return "Video summary unavailable due to processing limitations."