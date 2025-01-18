# Real-Time AI Face Landmark and Hand Movement Detection

This project is a real-time face landmark and hand movement detection application built using Mediapipe, OpenCV, and Streamlit. The app leverages advanced machine learning models to detect and track face landmarks and hand movements through a webcam interface.

# Features

1. **Real-Time Face Landmark Detection:**

  Detects up to 468 facial landmarks in real-time.

  Tracks facial expressions and features such as eyes, nose, and mouth.

2. **Hand Movement Detection:**

  Detects up to 21 hand landmarks per hand.

  Tracks hand gestures and movements with precision.

  Displays hand skeletons and allows visualization of connected joints.

3. **Interactive Controls:**

  Start and stop the webcam feed with simple button controls in the Streamlit        interface.

4. **User-Friendly Interface:**

  Streamlit provides an easy-to-use and visually appealing web interface for     
  running the application.

-------------------------------------------------------------

# Technologies Used

1. **Python:** Core programming language for the project.

2. **Mediapipe:** For detecting and drawing face and hand landmarks.

3. **OpenCV:** For webcam access and video frame processing.

4. **Streamlit:** To create an interactive web application.

# How It Works

1. **Face Mesh Detection:**

  Mediapipe’s Face Mesh model detects facial landmarks on the input video stream.

  Landmarks are rendered with lines and dots to create a mesh-like visualization   
  on the face.

2. **Hand Landmark Detection:**

  Mediapipe’s Hands model detects hand landmarks and draws the connections between   them.
  
  Tracks multiple hands and renders skeleton-like structures for visualization.

3. **Webcam Feed Control:**

  The application opens the webcam feed when the Start Webcam button is pressed.
  
  Stops the webcam feed when the Stop Webcam button is clicked.

## Installation

### Prerequisites

1. Python 3.8 or higher installed on your system.
2. Basic knowledge of Python programming.

### Steps to Install

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/real-time-face-hand-detection.git
   cd real-time-face-hand-detection
   ```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

3. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the application:**

```bash
streamlit run app.py
```

# Usage

1. Start the application by running the command:
```bash
  streamlit run app.py
```

2. The application will open in your default web browser.

3. Click **Start Webcam** to begin the real-time detection.

4. To stop the webcam feed, click **Stop Webcam.**

# Project Structure

```bash
real-time-face-hand-detection/
├── app.py              # Main application code
├── requirements.txt    # Dependencies for the project
├── README.md           # Project documentation
└── assets/             # (Optional) Folder for additional resources or images
```

# Dependencies

1. **Mediapipe**: ```bash pip install mediapipe ```

2. **OpenCV**: ```bash pip install opencv-python ```

3. **Streamlit**: ```bash pip install streamlit ```

To install all dependencies at once, run:
```bash
  pip install -r requirements.txt
```

# Future Enhancements

1. Add gesture recognition for specific hand signs or gestures.

2. Integrate emotion detection based on facial landmarks.

3. Improve the UI for better user experience.

4. Allow saving of processed video frames with annotations.

# Contributing

Contributions are welcome! If you’d like to contribute, please follow these steps:

1. Fork the repository.

2. Create a new branch:
```bash
git checkout -b feature-name
```

3. Commit your changes:
```bash
git commit -m "Added a new feature"
```

4. Push to the branch:
```bash
git push origin feature-name
```

5. Submit a pull request.

# License

This project is licensed under the MIT License. See the LICENSE file for details.

# Author

Piyush Singhal

If you have any questions or suggestions, feel free to contact me via email or open an issue on this repository.

# Acknowledgments

1. Mediapipe for providing robust face and hand tracking models.

2. Streamlit for simplifying the creation of web applications.

3. OpenCV for efficient image processing.

