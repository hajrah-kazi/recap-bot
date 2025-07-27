import os
import logging
import subprocess
import json
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory, abort
from dotenv import load_dotenv
import assemblyai as aai
from utils.frame_extractor import FrameExtractor
from utils.object_detection import ObjectDetector
from utils.minigpt_description import MiniGPTDescriptor
from utils.video_summary import VideoSummarizer
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
if not API_KEY:
    raise ValueError("No ASSEMBLYAI_API_KEY found in environment variables")

aai.settings.api_key = API_KEY

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/frames', exist_ok=True)

# Initialize components
# Make sure confidence_thresh is set low!
object_detector = ObjectDetector(confidence_thresh=0.25)
minigpt_descriptor = MiniGPTDescriptor()
video_summarizer = VideoSummarizer()

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path: str, audio_path: str, speed: float = 2.0) -> bool:
    try:
        logging.info(f"Extracting and speeding up audio from {video_path} at {speed}x")
        command = [
            'ffmpeg',
            '-i', video_path,
            '-filter:a', f'atempo={speed}',
            '-vn',
            audio_path
        ]
        subprocess.run(command, check=True)
        logging.info("Audio extraction and speeding up successful")
        return True
    except Exception as e:
        logging.error(f"Audio extraction failed: {str(e)}")
        return False

def transcribe_audio(audio_path: str) -> str:
    try:
        logging.info("Starting transcription with AssemblyAI")
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            language_detection=True,
            auto_highlights=True,
            speaker_labels=True
        )
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")
        logging.info("Transcription completed successfully")
        return transcript.text
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        return "Transcription unavailable"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(video_path)
        logging.info(f"Video uploaded: {video_path} with session_id: {session_id}")
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': filename,
            'message': 'Video uploaded successfully'
        })
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process/<session_id>', methods=['POST'])
def process_video(session_id):
    try:
        video_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(session_id)]
        if not video_files:
            return jsonify({'error': 'Video file not found'}), 404
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_files[0])
        audio_path = f"temp_audio_{session_id}.wav"
        
        # Extract and transcribe audio
        logging.info(f"Starting audio processing for session {session_id}")
        if not extract_audio(video_path, audio_path):
            logging.warning("Audio extraction failed, continuing without transcription")
            transcription = "Transcription unavailable"
        else:
            transcription = transcribe_audio(audio_path)
        
        logging.info(f"Transcription completed: {len(transcription) if transcription else 0} characters")
        
        frames_dir = os.path.join('static', 'frames', session_id)
        os.makedirs(frames_dir, exist_ok=True)

        #--- Frame Extraction ---
        logging.info(f"Starting frame extraction for session {session_id}")
        try:
            frame_extractor = FrameExtractor(video_path, output_dir=frames_dir)
            frames = frame_extractor.extract_unique_frames()
            logging.info(f"Extracted {len(frames)} frames")
        except Exception as e:
            logging.error(f"Frame extraction failed: {str(e)}")
            return jsonify({'error': f'Frame extraction failed: {str(e)}'}), 500

        # Handle frame paths
        if frames and ("frames_unique" in frames[0] or os.path.basename(os.path.dirname(frames[0])) == "frames_unique"):
            rel_frame_paths = [os.path.join('frames_unique', os.path.basename(f)) for f in frames]
        else:
            rel_frame_paths = [os.path.basename(f) for f in frames]

        #--- Object Detection ---
        logging.info(f"Starting object detection for {len(frames)} frames")
        detection_results = []
        successful_detections = 0
        
        for fidx, frame_path in enumerate(frames):
            try:
                detections = object_detector.detect_objects(frame_path)
                detection_results.append({
                    'frame': rel_frame_paths[fidx],
                    'detections': detections
                })
                if detections:
                    successful_detections += 1
            except Exception as e:
                logging.error(f"Object detection failed for frame {fidx}: {str(e)}")
                detection_results.append({
                    'frame': rel_frame_paths[fidx],
                    'detections': []
                })

        logging.info(f"Completed object detection: {successful_detections}/{len(frames)} frames had detections")

        #--- Frame Descriptions ---
        logging.info(f"Starting frame descriptions for {len(frames)} frames")
        descriptions = []
        successful_descriptions = 0
        
        for fidx, frame_path in enumerate(frames):
            try:
                description = minigpt_descriptor.describe_frame(frame_path)
                descriptions.append({
                    'frame': rel_frame_paths[fidx],
                    'description': description
                })
                if description and description.strip() and description != "Description not available":
                    successful_descriptions += 1
            except Exception as e:
                logging.error(f"Frame description failed for frame {fidx}: {str(e)}")
                descriptions.append({
                    'frame': rel_frame_paths[fidx],
                    'description': "Description not available"
                })

        logging.info(f"Completed frame descriptions: {successful_descriptions}/{len(frames)} frames had descriptions")

        #--- Generate Summary ---
        logging.info("Starting summary generation...")
        try:
            # Prepare data for summary generation
            valid_transcription = transcription if transcription != "Transcription unavailable" else None
            
            # Log what data we're passing to the summarizer
            logging.info(f"Summary input - Frames: {len(descriptions)}, Detections: {len(detection_results)}, "
                        f"Transcription: {'Yes' if valid_transcription else 'No'}")
            
            # Generate summary with all available data
            summary = video_summarizer.generate_summary(
                frame_data=descriptions,
                detection_data=detection_results,
                transcript=valid_transcription
            )
            
            if summary and len(summary.strip()) > 10:
                logging.info(f"Summary generated successfully: {len(summary)} characters")
            else:
                logging.warning("Summary generation produced minimal output")
                summary = "Summary generation completed but produced limited results."
                
        except Exception as e:
            logging.error(f"Summary generation failed: {str(e)}", exc_info=True)
            summary = f"Summary not available due to processing error: {str(e)}"

        # Prepare results
        results = {
            'session_id': session_id,
            'total_frames': len(frames),
            'frames': rel_frame_paths,
            'detections': detection_results,
            'descriptions': descriptions,
            'summary': summary,
            'transcription': transcription,
            'processing_stats': {
                'frames_extracted': len(frames),
                'successful_detections': successful_detections,
                'successful_descriptions': successful_descriptions,
                'transcription_available': transcription != "Transcription unavailable"
            }
        }
        
        # Save results
        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_results.json")
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {results_path}")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
        
        # Cleanup temporary audio file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logging.info("Temporary audio file cleaned up")
        except Exception as e:
            logging.warning(f"Failed to cleanup audio file: {str(e)}")
        
        return jsonify(results)

    except Exception as e:
        logging.error(f"Processing error for session {session_id}: {str(e)}", exc_info=True)
        # Try to cleanup audio file even on error
        try:
            audio_path = f"temp_audio_{session_id}.wav"
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<session_id>')
def api_get_results(session_id):
    try:
        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_results.json")
        if not os.path.exists(results_path):
            return jsonify({'error': 'Results not found'}), 404
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error retrieving results for session {session_id}: {str(e)}")
        return jsonify({'error': 'Failed to retrieve results'}), 500

@app.route('/results/<session_id>')
def results(session_id):
    return render_template('results.html', session_id=session_id)

@app.route('/static/frames/<session_id>/<path:filename>')
def serve_frame(session_id, filename):
    try:
        # Supports subfolders, e.g. frames_unique/frame_0001.jpg etc.
        frame_directory = os.path.join('static', 'frames', session_id)
        serve_path = os.path.join(frame_directory, filename)
        
        # Security check - ensure the path is within the expected directory
        if not os.path.abspath(serve_path).startswith(os.path.abspath(frame_directory)):
            logging.warning(f"Attempted path traversal attack: {filename}")
            abort(403)
            
        if not os.path.exists(serve_path):
            logging.warning(f"Frame not found: {serve_path}")
            abort(404)
            
        return send_from_directory(frame_directory, filename)
    except Exception as e:
        logging.error(f"Error serving frame {filename} for session {session_id}: {str(e)}")
        abort(500)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)