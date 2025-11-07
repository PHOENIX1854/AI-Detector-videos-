// content.js
console.log("AI Detector: Content script loaded.");

let lastUrl = "";
const serverUrl = "http://127.0.0.1:5000/check";
let display = null; // We will re-use the display element

// Function to show the result on the page
function displayResult(text, label = "loading") {
  // Find or create the display box
  if (!display) {
    display = document.createElement('div');
    display.id = 'ai-detector-display';
    document.body.appendChild(display);
  }

  // Update text and style
  display.textContent = text;
  
  if (label === "AI GENERATED") {
    display.className = "ai-detector-ai";
  } else if (label === "REAL") {
    display.className = "ai-detector-real";
  } else if (label === "error") {
    display.className = "ai-detector-error";
  } else {
    display.className = "ai-detector-loading";
  }
}

// Function to hide the display
function hideResult() {
  if (display) {
    display.remove();
    display = null;
  }
}

// Main function to check the video
async function checkCurrentVideo() {
  const url = window.location.href;
  
  // 1. Check if it's a new URL
  if (url === lastUrl) return;
  lastUrl = url;

  // 2. Check if it's a valid video URL
  const isVideo = url.includes("youtube.com/watch") || 
                  url.includes("youtube.com/shorts") || 
                  url.includes("instagram.com/reel") ||
                  url.includes("instagram.com/reels");
                  
  if (!isVideo) {
    hideResult(); // Hide the box if we're not on a video
    return;
  }
  
  console.log(`AI Detector: New video detected. Sending to server: ${url}`);
  displayResult("Analyzing...", "loading");

  // 3. Send URL to our Python server
  try {
    const response = await fetch(serverUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: url })
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const result = await response.json();
    
    if (result.error) {
      throw new Error(result.error);
    }

    // 4. Show the final result
    const confidence = (result.confidence * 100).toFixed(1);
    const label = result.label;
    displayResult(`${label} (${confidence}%)`, label);

  } catch (e) {
    console.error("AI Detector Error:", e);
    displayResult("Error: Could not analyze.", "error");
  }
}

// Use a MutationObserver to detect when the user navigates
// (e.g., clicking a new Short or related video)
const observer = new MutationObserver(() => {
  if (window.location.href !== lastUrl) {
    checkCurrentVideo();
  }
});

observer.observe(document.body, { childList: true, subtree: true });

// Initial check when the page first loads
checkCurrentVideo();