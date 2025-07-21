import cv2
import numpy as np
import sys
import os

def play_videos_loop(video_paths):
    """
    Concatenate and play three videos in fullscreen, looping continuously.
    Press 'q' to quit.
    """
    if len(video_paths) != 3:
        print("Please provide exactly 3 video file paths")
        return
    
    # Verify all video files exist
    for path in video_paths:
        if not os.path.exists(path):
            print(f"Video file not found: {path}")
            return
    
    # Open video captures
    caps = [cv2.VideoCapture(path) for path in video_paths]
    
    # Verify all videos opened successfully
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error opening video: {video_paths[i]}")
            return
    
    # Create fullscreen window
    window_name = "Video Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("Playing videos in fullscreen. Press 'q' to quit.")
    
    try:
        while True:
            for i, cap in enumerate(caps):
                print(f"Playing video {i+1}/{len(caps)}")
                
                # Reset video to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Get original FPS
                fps = cap.get(cv2.CAP_PROP_FPS)
                delay = int(1000 / fps * 2)  # Double the delay for 0.5x speed
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    cv2.imshow(window_name, frame)
                    
                    # Use calculated delay for 0.5x speed
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
    
    except KeyboardInterrupt:
        print("\nStopping video playback...")
    
    finally:
        # Clean up
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python video_player.py <video1.mp4> <video2.mp4> <video3.mp4>")
        print("Example: python video_player.py vid1.mp4 vid2.mp4 vid3.mp4")
        sys.exit(1)
    
    video_paths = sys.argv[1:4]
    play_videos_loop(video_paths)