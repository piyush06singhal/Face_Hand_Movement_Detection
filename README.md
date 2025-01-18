# Real-Time AI Face Landmark and Hand Movement Detection

This project is a real-time face landmark and hand movement detection application built using Mediapipe, OpenCV, and Streamlit. The app leverages advanced machine learning models to detect and track face landmarks and hand movements through a webcam interface.

# Features

Real-Time Face Landmark Detection:

Detects up to 468 facial landmarks in real-time.

Tracks facial expressions and features such as eyes, nose, and mouth.

Hand Movement Detection:

Detects up to 21 hand landmarks per hand.

Tracks hand gestures and movements with precision.

Displays hand skeletons and allows visualization of connected joints.

Interactive Controls:

Start and stop the webcam feed with simple button controls in the Streamlit interface.

User-Friendly Interface:

Streamlit provides an easy-to-use and visually appealing web interface for running the application.

# Technologies Used

Python: Core programming language for the project.

Mediapipe: For detecting and drawing face and hand landmarks.

OpenCV: For webcam access and video frame processing.

Streamlit: To create an interactive web application.

# How It Works

Face Mesh Detection:

Mediapipe’s Face Mesh model detects facial landmarks on the input video stream.

Landmarks are rendered with lines and dots to create a mesh-like visualization on the face.

Hand Landmark Detection:

Mediapipe’s Hands model detects hand landmarks and draws the connections between them.

Tracks multiple hands and renders skeleton-like structures for visualization.

Webcam Feed Control:

The application opens the webcam feed when the Start Webcam button is pressed.

Stops the webcam feed when the Stop Webcam button is clicked.

# Installation

Prerequisites

Python 3.8 or higher installed on your system.

Basic knowledge of Python programming.
