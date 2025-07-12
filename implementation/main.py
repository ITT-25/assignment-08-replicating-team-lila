import time
import click
import cv2
import numpy as np
import config as cfg

def capture_loop(dt: float, frame: np.ndarray) -> None:
    # TODO: Create seperate modules for each step in the pipeline
    # 1. ArUco marker detection and perspective transformation
    # Find aruco markers in the frame
    # Get perspective transformation matrix based on the detected markers
    # Apply matrix and trim frame to markers
    
    # 2. Fingertip position detection (Position + Pressing status)
    # Detect Hands with mediapipe and return fingertip coordinates relative to the cropped frame
    # - Also detect if a finger is pressed down
    # - Pressing status might require the hand detection to run before the perspective transformation,
    #   which would require the perspective transformation to be applied to the fingertip positions as well (I think mediapipe can do 3D coordinates to a certain extent)
    
    # 3. Map fingertips to piano keys
    # Match the fingertip positions with a virtual piano keyboard and play the corresponding notes (high pitch depending on vertical position on key)
    # Input: Fingertip Positions
    # Output: List of pressed notes (with pitch)
    
    # 4. Media playback + Visualization
    # Play the notes in a media player and visualize the pressed keys in a cv2 window
    pass

@click.command()
@click.option("--video-id", "-c", default=1, help="Video ID")
def main(video_id: int):
    cap = cv2.VideoCapture(video_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.VIDEO_HEIGHT)
    cfg.VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cfg.VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {video_id}")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            key = cv2.waitKey(1) & 0xFF
            if not ret or key == ord("q") or key == 27: # Q or Escape
                print("Error: Could not read frame from camera.")
                break
            capture_loop(1 / cfg.SAMPLING_RATE, frame)
            time.sleep(1 / cfg.SAMPLING_RATE)
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
    
    # Detect AruCo markers

if __name__ == "__main__":
    main()