import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        # Features parameters for ShiTomasi corner detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7
        )

        # Lucas Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # We need a mask to only track features on the pitch (background), not the players/crowd
        # A simple approach is to track everything, but we can refine it later if needed.
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Mask out top and bottom to avoid tracking the crowd/scoreboards
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[20:-20, 20:-20] = 255 # Simple margin

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        import pickle
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = []
        old_gray = None
        old_features = None

        for frame_num, frame in enumerate(frames):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_num == 0:
                old_gray = frame_gray
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
                camera_movement.append([0, 0])
                continue

            new_features, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # Calculate movement from valid tracked features
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()
                
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_movement.append([camera_movement_x, camera_movement_y])
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            else:
                camera_movement.append([0, 0])
            
            old_gray = frame_gray.copy()

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame_copy = frame.copy()
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            cv2.putText(frame_copy, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame_copy, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            output_frames.append(frame_copy)

        return output_frames

    def adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    # Adjust position by subtracting camera movement
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    track_info['position_adjusted'] = position_adjusted

    def draw_camera_movement_single_frame(self, frame, frame_num, camera_movement_per_frame):
        frame_copy = frame.copy()
        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)

        x_movement, y_movement = camera_movement_per_frame[frame_num]
        cv2.putText(frame_copy, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame_copy, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return frame_copy
