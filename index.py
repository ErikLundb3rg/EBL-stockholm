from flask import Flask, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from gladia import transcribe
import text_data as td
from pitch import add_pich_to_transcript
from utils import replace_nan_with_none
import json

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins
        "methods": ["POST", "OPTIONS"],  # Allow POST and preflight requests
        "allow_headers": ["Content-Type", "Authorization"]  # Allow these headers
    }
})

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/video', methods=['POST', 'OPTIONS'])
def upload_video():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return {}, 200
        
    # Check if a file was included in the request
    if 'video' not in request.files:
        return {'error': 'No video file in request'}, 400
    
    file = request.files['video']
    
    # Check if a file was selected
    if file.filename == '':
        return {'error': 'No selected file'}, 400
    
    # Check if the file is allowed
    if not allowed_file(file.filename):
        return {'error': 'File type not allowed'}, 400
    try:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
    except Exception as e:
        return {'error': str(e)}, 500
    
    # check if we already transcribed the video
    if not os.path.exists(f"uploads/{filename}.json"):
        transcribed_text = transcribe(filepath)
        if "error" in transcribed_text:
            return transcribed_text, transcribed_text["error_code"]
    
        transcribed_text = add_pich_to_transcript(filepath, transcript=transcribed_text)
        # save transcript as json in uploads folder
        with open(f"uploads/{filename}.json", "w") as f:
            f.write(json.dumps(transcribed_text, indent=4))
    else: 
        with open(f"uploads/{filename}.json", "r") as f:
            transcribed_text = json.load(f)



    return replace_nan_with_none({
            'message': 'Video uploaded successfully',
            'filename': filename,
            'textData': td.text_data(transcribed_text),
            'pitch': transcribed_text.get("pitch")
        }), 200

if __name__ == '__main__':
    app.run(debug=True, port=1336)