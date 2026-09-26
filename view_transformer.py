import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        # Default pitch dimensions in meters
        pitch_width = 68
        pitch_length = 105
        
        # Define keypoints in the pixel space (example coordinates from a standard broadcast view)
        # Note: In a real advanced system, these would be detected dynamically using another model.
        # Here we provide a set of standard points that form a rectangle in the 2D view.
        self.pixel_vertices = np.array([
            [110, 1035], # Bottom left (near corner)
            [265, 275],  # Top left (far corner)
            [910, 260],  # Top right (far corner)
            [1640, 915]  # Bottom right (near corner)
        ])
        
        # Define the corresponding points in the real-world bird's-eye view (in meters)
        self.target_vertices = np.array([
            [0, pitch_width],
            [0, 0],
            [pitch_length, 0],
            [pitch_length, pitch_width]
        ])

        # Calculate the homography matrix
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)
        self.perspective_transform, _ = cv2.findHomography(self.pixel_vertices, self.target_vertices)
        
    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        
        # Check if the point is inside the polygon we defined
        # This prevents wildly inaccurate transformations for players off-pitch
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None
            
        reshaped_point = np.array(point).reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transform)
        return transformed_point.reshape(-1, 2)[0].tolist()

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position_adjusted', track_info['position'])
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        track_info['position_transformed'] = position_transformed
