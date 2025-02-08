from flask import Flask, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
        
        return {
            'message': 'Video uploaded successfully',
            'filename': filename
        }, 200
        
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True, port=1336)