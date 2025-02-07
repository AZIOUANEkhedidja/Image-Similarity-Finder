from flask import Flask, render_template, request, jsonify
import os
from script import query_image_data  # Import only necessary functions
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__, static_url_path='', static_folder='static')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and process the file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Query the image data and return results
        results = query_image_data(filepath)
        print(results)

        return jsonify({'message': 'File uploaded successfully', 'filepath': results}), 200

    return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
