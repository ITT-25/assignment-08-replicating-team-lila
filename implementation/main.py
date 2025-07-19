import time
import click
import cv2
import numpy as np
import config as cfg

# Pipeline modules
from marker_detection import MarkerDetection
from fingertip_detection import FingertipDetection
from piano import Piano
from media_playback import MediaPlayback

fingertips = FingertipDetection()
markers = MarkerDetection()
piano = Piano()
media = MediaPlayback(piano)


def capture_loop(dt: float, frame: np.ndarray) -> None:
    # 1. ArUco marker detection and perspective transformation
    mrks = markers.detect(frame)
    if mrks is None:
        print("Error: No markers detected.")
        return
    
    matrix = markers.get_transform_matrix(mrks, frame.shape[1], frame.shape[0])
    transformed_frame = markers.apply_transformation(frame, matrix)

    transformed_frame = cv2.resize(transformed_frame, (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    transformed_frame = cv2.flip(transformed_frame, -1)

    # 2. Fingertip position detection (Position + Pressing status)
    fts = fingertips.detect(transformed_frame, matrix)
    
    # 3. Map fingertips to piano keys
    piano.update(fts)
    media.update(dt, transformed_frame)

@click.command()
@click.option("--video-id", "-c", default=1, help="Video ID")
def main(video_id: int):
    cap = cv2.VideoCapture(video_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.VIDEO_HEIGHT)
    # cfg.VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # cfg.VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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