import os
import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import yt_dlp # The new library for downloading videos

# Import your existing code
from detector_logic import load_models, process_video_file

# --- 1. Configuration ---
app = Flask(__name__)
CORS(app) # Allows the extension to call this server
TEMP_FOLDER = "temp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Load Models ONCE on Startup ---
print(" * Loading all AI models...")
VLM, PROCESSOR, CLASSIFIER = load_models(DEVICE)
print(" * Models loaded. Server is ready.")

# --- 3. Define the API Endpoint ---
@app.route('/check', methods=['POST'])
def check_video():
    data = request.json
    if 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data['url']
    video_path = None

    try:
        # --- 4. Download the Video ---
        print(f"Downloading video from: {url}")
        
        # Create a temp directory if it doesn't exist
        if os.path.exists(TEMP_FOLDER):
            shutil.rmtree(TEMP_FOLDER) # Clear old videos
        os.makedirs(TEMP_FOLDER)

        # Configure yt-dlp to download a small, fast-processing video
        ydl_opts = {
            # This is a much more flexible format.
            # It tries to get an mp4 up to 720p, or falls back to the 'best' of anything.
            'format': 'best[ext=mp4][height<=720]/best[height<=720]/best[ext=mp4]/best',
            'outtmpl': os.path.join(TEMP_FOLDER, 'video.%(ext)s'),
            'nocheckcertificate': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)

        print(f"Video downloaded to: {video_path}")

        # --- 5. Run Your Existing Logic ---
        # We pass the file path to your existing function
        result = process_video_file(video_path, VLM, PROCESSOR, CLASSIFIER, DEVICE)
        print(f"Analysis complete. Result: {result}")
        
        return jsonify(result)

    except Exception as e:
        print(f"Error processing {url}: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # --- 6. Clean Up ---
        if os.path.exists(TEMP_FOLDER):
            shutil.rmtree(TEMP_FOLDER)
            print(f"Cleaned up temp folder.")

# --- 7. Run the Server ---
if __name__ == '__main__':
    # We run on port 5000.
    app.run(debug=True, port=5000)