import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark as landmark_module
import time
import numpy as np
import pyttsx3 # Added for Text-to-Speech

# Lip landmark indices
LIP_LANDMARK_INDICES = sorted(list(set([
    0, 13, 14, 17, 37, 39, 40, 61, 76, 78, 80, 81, 82, 84, 87, 88, 91, 95,
    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314,
    317, 318, 321, 324, 375, 402, 405, 409
])))

def speak_text(engine, text):
    """
    Uses the TTS engine to speak the given text.
    """
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error speaking text: {e}")

def main():
    # Initialize TTS Engine
    tts_engine = None # Initialize to None
    try:
        tts_engine = pyttsx3.init()
        # Removed problematic _driverName check and rate adjustment
        print("TTS Engine Initialized.")
        speak_text(tts_engine, "Text to speech initialized.")
    except Exception as e:
        print(f"Error initializing TTS engine: {e}")
        print("Attempting to install 'espeak' as it might be a missing dependency for pyttsx3 on Linux.")
        # This inner call to bash is not how the agent framework works.
        # This part would be handled by a subsequent agent turn if TTS fails.
        # For now, we just note the failure.
        # run_in_bash_session("sudo apt-get update && sudo apt-get install -y espeak")
        # print("If 'espeak' was installed, please try running the script again.")
        # return # Exit if TTS fails to initialize, as it's a key feature for this step.
        # For this subtask, we will let it continue even if TTS init fails, to check other parts.
        pass # Continue for now, to allow testing other parts if TTS fails


    model_path = 'face_landmarker.task'
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )

    landmarker = None # Initialize to None
    try:
        landmarker = vision.FaceLandmarker.create_from_options(options)
        print("FaceLandmarker created successfully.")
    except Exception as e:
        print(f"Error creating FaceLandmarker: {e}")
        if tts_engine: # Attempt to stop engine if it was started
            tts_engine.stop()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        if landmarker:
            landmarker.close()
        if tts_engine:
            tts_engine.stop()
        return

    frame_timestamp_ms = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)

        try:
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        except Exception as e:
            print(f"Error during detection: {e}")
            continue

        display_frame = np.copy(frame)
        height, width, _ = display_frame.shape

        if detection_result and detection_result.face_landmarks:
            for face_idx, face_landmarks_instance in enumerate(detection_result.face_landmarks):
                for landmark_idx, landmark in enumerate(face_landmarks_instance):
                    x_px = int(landmark.x * width)
                    y_px = int(landmark.y * height)
                    if 0 <= x_px < width and 0 <= y_px < height:
                         cv2.circle(display_frame, (x_px, y_px), 1, (255, 0, 0), -1)

                    if landmark_idx in LIP_LANDMARK_INDICES:
                        # print(f"Lip Landmark {landmark_idx}: (x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f})") # Keep this commented for now
                        if 0 <= x_px < width and 0 <= y_px < height:
                            cv2.circle(display_frame, (x_px, y_px), 2, (0, 255, 0), -1)

        cv2.imshow("Lip Reader Camera Feed", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if landmarker:
        landmarker.close()
    if tts_engine: # Ensure engine is stopped
        try:
            tts_engine.stop() # Important to allow script to exit if runAndWait had issues
        except Exception as e:
            print(f"Error stopping TTS engine: {e}")
    print("Resources released.")

if __name__ == "__main__":
    main()
