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
    # ArUco marker detection and perspective transformation
    mrks = markers.detect(frame)
    if len(mrks) < 4:
        print(f"{len(mrks)}/4 markers")
        return

    matrix = markers.get_transform_matrix(mrks, frame.shape[1], frame.shape[0])
    
    transformed_frame = markers.apply_transformation(frame, matrix)
    transformed_frame = cv2.resize(transformed_frame, (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))

    # Fingertip position detection (Position + Pressing status)
    fts = fingertips.detect(frame, matrix)
    
    # Get the hand mask from fingertip detection
    hand_mask = fingertips.get_hand_mask()
    # blur the hand mask to reduce noise
    hand_mask = cv2.GaussianBlur(hand_mask, (3, 3), 0)
    
    # Map fingertips to piano keys
    piano.update(fts)

    # Draw the piano keys on a separate frame and get mask
    piano_frame = np.zeros_like(frame)
    piano_frame, key_mask = media.draw_keys(piano_frame, piano.keys)
    
    # Apply inverse transformation to get piano in original frame perspective
    untransformed_piano_frame = cv2.warpPerspective(piano_frame, np.linalg.inv(matrix), (frame.shape[1], frame.shape[0]))
    untransformed_key_mask = cv2.warpPerspective(key_mask, np.linalg.inv(matrix), (frame.shape[1], frame.shape[0]))
    
    # Create the final composite by starting with the original frame
    final_frame = frame.copy()
    
    # Add the piano keys on top of the original frame with proper masking
    normalized_hand_mask = hand_mask.astype(np.float32) / 255.0

    # Use the mask returned by draw_keys (after perspective transform)
    key_mask_float = untransformed_key_mask.astype(np.float32)
    key_mask_3ch = cv2.merge([key_mask_float, key_mask_float, key_mask_float])

    # Convert hand mask to 3-channel
    hand_mask_3ch = cv2.merge([normalized_hand_mask, normalized_hand_mask, normalized_hand_mask])

    # Composite: show original frame where hands are present, otherwise show piano keys where drawn
    final_frame = np.where(
        hand_mask_3ch > 0.1,  # Where hands are present
        frame.astype(np.float32),
        np.where(
            key_mask_3ch > 0.1,
            untransformed_piano_frame.astype(np.float32),
            frame.astype(np.float32)
        )
    ).astype(np.uint8)
    
    media.update(dt)
    
    if cfg.DEBUG:
        cv2.imshow("Overlay", final_frame)
        cv2.waitKey(1)

@click.command()
@click.option("--video-id", "-c", default=1, help="Video ID")
@click.option("--debug", "-d", is_flag=True, help="Show debug visualization")
@click.option("--press-threshold", "-p", default=320, help="Minimum distance between base and tip for a finger press detection (pixels)")
@click.option("--velocity-threshold", "-v", default=20, help="Minimum distance velocity for finger press detection (pixels/frame)")
def main(video_id: int, debug: bool, press_threshold: int, velocity_threshold: int) -> None:
    """Main function to capture video and process frames."""
    cfg.DEBUG = debug
    
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