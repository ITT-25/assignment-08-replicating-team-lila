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
    """Process each frame of video to detect markers, hands, and render piano."""
    # 1. ArUco marker detection and perspective transformation
    mrks = markers.detect(frame)
    if len(mrks) < 4:
        print(f"{len(mrks)}/4 markers")
        return

    matrix = markers.get_transform_matrix(mrks, frame.shape[1], frame.shape[0])
    
    transformed_frame = markers.apply_transformation(frame, matrix)
    transformed_frame = cv2.resize(transformed_frame, (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))

    # 2. Fingertip position detection (Position + Pressing status)
    fts = fingertips.detect(frame, matrix)
    
    # Get the hand mask from fingertip detection
    hand_mask = fingertips.get_hand_mask()
    
    # 3. Map fingertips to piano keys
    piano.update(fts)

    # 4. Draw the piano keys on a separate frame
    piano_frame = np.zeros_like(frame)
    piano_frame = media.draw_keys(piano_frame, piano.keys)
    
    # Apply inverse transformation to get piano in original frame perspective
    untransformed_piano_frame = cv2.warpPerspective(piano_frame, np.linalg.inv(matrix), (frame.shape[1], frame.shape[0]))
    
    # 5. Create the final composite by starting with the original frame
    final_frame = frame.copy()
    
    # 6. Add the piano keys on top of the original frame, but only where there are no hands
    # Create a binary mask where piano keys should be visible (where hand mask is 0)
    piano_mask = cv2.bitwise_not(hand_mask)
    
    # Get the piano elements that should be visible (where there are no hands)
    visible_piano = cv2.bitwise_and(untransformed_piano_frame, untransformed_piano_frame, mask=piano_mask)
    
    # Add the visible piano elements to the final frame
    final_frame = cv2.add(final_frame, visible_piano)
    
    media.update(dt)
    
    cv2.imshow("Overlay", final_frame)
    cv2.waitKey(1)

@click.command()
@click.option("--video-id", "-c", default=1, help="Video ID")
@click.option("--debug", "-d", is_flag=True, help="Show debug visualization")
@click.option("--press-threshold", "-p", default=320, help="Minimum distance between base and tip for a finger press detection (pixels)")
@click.option("--velocity-threshold", "-v", default=20, help="Minimum distance velocity for finger press detection (pixels/frame)")
def main(video_id: int, debug: bool, press_threshold: int, velocity_threshold: int) -> None:
    """Main function to capture video and process frames."""
    global DEBUG_VISUALIZE
    DEBUG_VISUALIZE = debug
    
    # Set the thresholds from CLI parameters
    cfg.PRESS_DISTANCE_THRESHOLD = press_threshold
    cfg.DISTANCE_VELOCITY_THRESHOLD = velocity_threshold
    
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