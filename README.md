AI-Generated Video Detector (Full-Stack)

A full-stack application that detects whether a video is AI-generated. This project uses a powerful Python/PyTorch backend to run a custom-trained AI model and a lightweight browser extension to display the results as an overlay on YouTube and Instagram.

(The overlay appears in the top-right corner, showing the "AI" or "REAL" verdict)

Features

Real-time Detection: Analyzes videos as you browse.

Custom AI Model: Uses a fine-tuned CLIP model (clip-vit-base-patch32) with a custom MLP classifier to determine if a video is real or AI-generated.

Seamless UI: Injects a simple, clean overlay with the detection result directly onto the video page.

Broad Support: Works on YouTube (full videos and Shorts) and Instagram (Reels).

Full-Stack Architecture: Demonstrates a robust client-server model with a Python/Flask API backend and a JavaScript extension frontend.

Tech Stack

Backend: Python, Flask, PyTorch, yt-dlp (for video downloading)

Frontend: JavaScript, HTML, CSS (as a Browser Extension)

AI Model: openai/clip-vit-base-patch32 + a custom MLP classifier (.pth)

Architecture: Why Server-Side?

This project uses a client-server architecture. The browser extension is a "dumb" client that simply sends the current video URL to a local Python server, which does all the heavy lifting.

content.js (Extension): Detects a new video URL (e.g., youtube.com/...).

fetch (Extension): Sends a POST request with the URL to http://127.0.0.1:5000/check.

app.py (Server):

Receives the URL.

Uses yt-dlp to download the video file to a temporary folder.

Uses your detection_logic.py to analyze the video and get the highest AI score.

Sends a JSON response back (e.g., {"label": "AI GENERATED", "confidence": 0.92}).

content.js (Extension): Receives the JSON and displays the result in the overlay.

A Critical Design Choice: Server-Side vs. Client-Side

I initially attempted a purely "Client-Side" (in-browser) solution, but it proved to be a poor fit for this project. The final server-side architecture was a deliberate choice for several key reasons:

The Problems with a Client-Side (In-Browser) Model

To run a PyTorch model in a browser, it must be converted to a web-friendly format like .onnx and run using a JavaScript library (like ONNX.js). This approach, while "pure," was plagued with issues:

Extreme Complexity: A Manifest V3 extension's security policies make this a nightmare. We had to fight:

Service Workers: Cannot run the ONNX library.

Offscreen Documents: A complex, hacky workaround is required to run the model in an invisible HTML page.

Security Policies (CSP): The extension's CSP must be modified with 'wasm-unsafe-eval' just to allow the WebAssembly model to compile, which is a significant security compromise.

File Errors: The library requires multiple .wasm and .mjs helper files, which constantly caused Failed to fetch errors inside the extension's sandboxed environment.

No Access to URLs: A client-side extension cannot accept a URL. It can only analyze pixels from a <video> tag that is already playing on the page. This makes it impossible to check a video just by its link.

Model Limitations: We would be forced to use the converted .onnx model, which is less flexible than the original Python .pth file.

The Advantages of the Server-Side Model (This Project)

By moving the AI logic to a local server, we solve every single one of these problems:

Simplicity: The extension becomes incredibly simple. It's just one script (content.js) that sends a fetch request. No complex architecture is needed.

Original Model: We can use our original .pth file and the full power of PyTorch. No model conversion is necessary.

Power & Flexibility: The server can use your computer's GPU, run complex Python logic, and is not limited by browser security.

Accepts URLs: This architecture allows us to use yt-dlp to download any video from just its URL, which was a core goal.

Easy to Maintain: To update the AI model, you just update the .pth file on the server. The extension doesn't even need to be reloaded.

For a personal or portfolio project, this architecture is a vastly superior, simpler, and more powerful solution.

How to Run This Project

You must run both the backend server and the frontend extension at the same time.

1. Run the Backend Server

Open your terminal.

Navigate to the backend folder:

cd /path/to/AI-Detector-Project/backend


Important: Place your trained best_nn_classifier_FINAL.pth file in this backend directory.

Install the required Python libraries:

pip install -r requirements.txt


Run the Flask server:

python app.py


You should see it load your models and say: * Running on http://127.0.0.1:5000.

Leave this terminal running.

2. Load the Frontend Extension

Open your Brave or Chrome browser.

Navigate to the extensions page: brave://extensions or chrome://extensions.

In the top-right corner, turn on "Developer mode".

Click "Load unpacked".

Select the entire frontend-extension folder.

3. (Brave Users Only) Disable Shields

The extension's content.js (running on youtube.com) needs to send a request to your server (127.0.0.1). Brave Shields will block this by default.

Go to a YouTube or Instagram page.

Click the Brave lion icon in the address bar.

Click the main toggle to turn off Shields for that site.

The page will reload, and the extension will now have permission to contact your server.

How to Use

Simply open a video on YouTube, a YouTube Short, or an Instagram Reel. The extension will automatically detect the video, show a gray "Analyzing..." box in the top-right corner, and then update it with the final "AI" or "REAL" verdict.


License

This project is licensed under the MIT License.

