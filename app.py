import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# Initialize mediapipe face mesh and hands modules
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define the face mesh and hand models
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=10, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=10, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Streamlit UI setup
st.title("Real-Time AI Face Landmark and Hand Movement Detection")
st.markdown("This is a detection app created using Mediapipe, OpenCV, and Streamlit.")

# List to store hand positions for drawing lines
hand_positions = []

# Function for real-time detection
def landmark_and_hand_detection():
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame. Exiting...")
            break

        # Mirror the frame to fix the left-right rotation issue
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe face mesh model
        results_face = face_mesh.process(rgb_frame)
        
        # Process the frame with Mediapipe hands model
        results_hands = hands.process(rgb_frame)

        # If face landmarks are found, draw them on the frame
        if results_face.multi_face_landmarks:
            for landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

        # If hand landmarks are found, draw them on the frame and track movement
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                hand_positions.clear()  # Clear the positions for each frame
                for i in range(0, 21):  # Iterate through the hand landmarks (21 points)
                    x = int(hand_landmarks.landmark[i].x * frame.shape[1])
                    y = int(hand_landmarks.landmark[i].y * frame.shape[0])
                    hand_positions.append((x, y))

                # Draw lines between specific hand points (for example, connecting the thumb and index finger)
                for i in range(0, len(hand_positions)-1):
                    cv2.line(frame, hand_positions[i], hand_positions[i+1], (255, 255, 0), 2)

        # Convert back to BGR for displaying
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Update the displayed frame
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()

# Streamlit UI controls for starting and stopping webcam
start_button = st.button("Start Webcam")

# Use session state to track webcam status
if 'webcam_started' not in st.session_state:
    st.session_state.webcam_started = False

if start_button and not st.session_state.webcam_started:
    st.session_state.webcam_started = True
    landmark_and_hand_detection()

# Show Stop button only after the webcam feed starts
if st.session_state.webcam_started:
    stop_button = st.button("Stop Webcam")
    if stop_button:
        st.session_state.webcam_started = False
        st.info("Webcam stopped. Click 'Start Webcam' to resume.")
        cv2.destroyAllWindows()
