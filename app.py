import os
import logging
import subprocess
import json
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
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

aai.settings.api_key = API_KEY  # Use the API key from environment variables

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/frames', exist_ok=True)

# Initialize components
object_detector = ObjectDetector()
minigpt_descriptor = MiniGPTDescriptor()
video_summarizer = VideoSummarizer()

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path: str, audio_path: str, speed: float = 2.0) -> bool:
    """Extract audio from video file and speed it up with error handling."""
    try:
        logging.info(f"Extracting and speeding up audio from {video_path} at {speed}x")
        command = [
            'ffmpeg',
            '-i', video_path,
            '-filter:a', f'atempo={speed}',  # Speed up audio
            '-vn',  # No video
            audio_path
        ]
        subprocess.run(command, check=True)  # Run the command
        logging.info("Audio extraction and speeding up successful")
        return True
    except Exception as e:
        logging.error(f"Audio extraction failed: {str(e)}")
        return False

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using AssemblyAI with comprehensive error handling."""
    try:
        logging.info("Starting transcription with AssemblyAI")
        
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            language_detection=True,
            auto_highlights=True,
            speaker_labels=True
        )
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(
            audio_path,
            config=config
        )

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")
        
        logging.info("Transcription completed successfully")
        return transcript.text
    
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        raise

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
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
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
        # Find video file for the session
        video_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(session_id)]
        if not video_files:
            return jsonify({'error': 'Video file not found'}), 404
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_files[0])
        audio_path = f"temp_audio_{session_id}.wav"
        
        # Step 1: Extract and speed up audio
        if not extract_audio(video_path, audio_path):
            return jsonify({'error': 'Audio extraction failed'}), 500
        
        # Step 2: Transcribe audio
        transcription = transcribe_audio(audio_path)
        
        # Step 3: Extract unique frames
        frames_dir = f"static/frames/{session_id}"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Initialize FrameExtractor with output_dir
        frame_extractor = FrameExtractor(video_path, output_dir=frames_dir)
        frames = frame_extractor.extract_unique_frames()
        
        # Step 4: Perform object detection per frame
        detection_results = []
        for frame_path in frames:
            detections = object_detector.detect_objects(frame_path)
            detection_results.append({
                'frame': os.path.basename(frame_path),
                'detections': detections
            })
        
        # Step 5: Generate frame descriptions
        descriptions = []
        for frame_path in frames:
            description = minigpt_descriptor.describe_frame(frame_path)
            descriptions.append({
                'frame': os.path.basename(frame_path),
                'description': description
            })
        
        # Step 6: Generate video summary
        summary = video_summarizer.generate_summary(descriptions, detection_results)
        
        # Prepare results
        results = {
            'session_id': session_id,
            'total_frames': len(frames),
            'frames': [os.path.basename(f) for f in frames],
            'detections': detection_results,
            'descriptions': descriptions,
            'summary': summary,
            'transcription': transcription
        }
        
        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {results_path}")
        
        # Cleanup temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logging.info(f"Temporary audio file {audio_path} removed")
        
        return jsonify(results)
    
    except Exception as e:
        logging.error(f"Processing error for session {session_id}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<session_id>')
def api_get_results(session_id):
    results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_results.json")
    if not os.path.exists(results_path):
        return jsonify({'error': 'Results not found'}), 404
    with open(results_path, 'r') as f:
        results = json.load(f)
    return jsonify(results)

@app.route('/results/<session_id>')
def results(session_id):
    # Just render the results page where frontend will poll `/api/results/<session_id>`
    return render_template('results.html', session_id=session_id)

@app.route('/static/frames/<session_id>/<filename>')
def serve_frame(session_id, filename):
    return send_from_directory(f'static/frames/{session_id}', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
