import cv2
import numpy as np

class CameraTransformation:
    """
    Computes and applies planar homography transformations between broadcast camera
    perspective coordinates and top-down 2D pitch coordinates.
    """
    def __init__(self, src_points, dst_points):
        """
        Initializes the transformation matrix using 4 reference points from the video 
        and their corresponding 4 points on the 2D pitch.

        Args:
            src_points (list or np.ndarray): 4 (x, y) coordinates from the camera view.
            dst_points (list or np.ndarray): 4 (x, y) corresponding coordinates on the 2D map.
        """
        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)
        
        if len(self.src_points) < 4 or len(self.dst_points) < 4:
            raise ValueError("At least 4 points must be provided for both src_points and dst_points.")

        # Calculate forward perspective transformation matrix (camera -> 2D pitch)
        self.homography_matrix, _ = cv2.findHomography(self.src_points, self.dst_points)
        
        # Calculate inverse homography matrix (2D pitch -> camera)
        if self.homography_matrix is not None:
            self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)
        else:
            self.inv_homography_matrix = None

    def transform_point(self, point):
        """
        Transforms a single (x, y) point from the camera frame to the 2D flat map.

        Args:
            point (tuple or list): The (x, y) coordinates of the player/ball in the video.

        Returns:
            tuple or None: Transformed (x', y') integers on the 2D map, or None if invalid.
        """
        if self.homography_matrix is None or point is None:
            return None
            
        p = np.array([[point]], dtype=np.float32)
        transformed_p = cv2.perspectiveTransform(p, self.homography_matrix)
        
        res = transformed_p[0][0]
        if np.isnan(res).any() or np.isinf(res).any():
            return None
            
        return int(round(float(res[0]))), int(round(float(res[1])))

    def transform_points_batch(self, points):
        """
        Vectorized transformation of multiple (x, y) points.

        Args:
            points (list or np.ndarray): List of (x, y) tuples/lists.

        Returns:
            list: List of (x', y') tuples or None for invalid projections.
        """
        if self.homography_matrix is None or len(points) == 0:
            return []
            
        pts = np.array([points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)[0]
        
        results = []
        for pt in transformed:
            if np.isnan(pt).any() or np.isinf(pt).any():
                results.append(None)
            else:
                results.append((int(round(float(pt[0]))), int(round(float(pt[1])))))
        return results

    def transform_point_to_camera(self, point):
        """
        Inverse projection: Transforms a point from 2D tactical pitch back into video camera frame.
        Useful for projecting tactical zones, heatmaps, or lines onto the camera feed.
        """
        if self.inv_homography_matrix is None or point is None:
            return None
            
        p = np.array([[point]], dtype=np.float32)
        transformed_p = cv2.perspectiveTransform(p, self.inv_homography_matrix)
        
        res = transformed_p[0][0]
        if np.isnan(res).any() or np.isinf(res).any():
            return None
            
        return int(round(float(res[0]))), int(round(float(res[1])))