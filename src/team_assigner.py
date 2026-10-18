import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    """
    Uses K-Means clustering to:
    1. Extract dominant jersey colors from individual player crops.
    2. Cluster detected players into Team A, Team B, and Referee groups.
    """
    def __init__(self, team_colors=None, referee_color=(180, 180, 180)):
        # Default vivid team colors: Red and Yellow (BGR format)
        self.team_colors = team_colors or [(0, 0, 230), (0, 230, 230)]
        self.referee_color = referee_color
        
        self.kmeans = None
        self.referee_cluster = None
        self.team_clusters = []
        self.is_fitted = False
        
    def get_player_color(self, frame, bbox):
        """
        Extracts the dominant jersey color from the top half of a player's bounding box.
        
        Strategy: Run K-Means with 3 clusters on the jersey area. Penalize grass-green
        and dark shadow pixels to isolate the jersey fabric.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # 1. Crop player from full frame
        player_img = frame[y1:y2, x1:x2]
        if player_img.size == 0 or player_img.shape[0] < 4 or player_img.shape[1] < 4:
            return (0, 0, 0)
            
        # 2. Crop top half (jersey area, ignores shorts and grass)
        top_half = player_img[0:int(player_img.shape[0] / 2), :]
        if top_half.size == 0:
            return (0, 0, 0)
        
        # 3. Reshape pixels
        pixels_bgr = top_half.reshape(-1, 3).astype(np.float32)
        if len(pixels_bgr) < 10:
            return (0, 0, 0)
        
        # 4. Cluster BGR pixels into 3 groups
        k = min(3, len(pixels_bgr))
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(pixels_bgr)
        centers = kmeans.cluster_centers_
        
        # 5. Score each cluster
        best_idx = 0
        best_score = -1e9
        
        for i, center in enumerate(centers):
            hsv = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv
            
            # Penalize grass-green: hue 30-90
            green_penalty = 200 if (30 <= h <= 90 and s > 40 and v > 40) else 0
            
            # Penalize very dark colors (shadows)
            dark_penalty = 150 if v < 50 else 0
            
            score = float(s) + float(v) * 0.5 - green_penalty - dark_penalty
            if score > best_score:
                best_score = score
                best_idx = i
        
        dominant_color = centers[best_idx]
        return (int(round(dominant_color[0])), int(round(dominant_color[1])), int(round(dominant_color[2])))

    def fit_teams(self, raw_player_colors):
        """
        Fits K-Means (k=3) on collected player jersey colors to separate Team A, Team B, and Referee.
        
        Args:
            raw_player_colors (dict): {player_id: (B, G, R)}
            
        Returns:
            dict: {player_id: (B, G, R)} mapped to vivid team/referee colors.
        """
        if len(raw_player_colors) < 3:
            return {}
            
        ids = list(raw_player_colors.keys())
        colors_array = np.array([raw_player_colors[pid] for pid in ids], dtype=np.float32)
        
        self.kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        self.kmeans.fit(colors_array)
        
        # Identify referee cluster (lowest saturation * value -> most neutral/dark)
        centers = self.kmeans.cluster_centers_
        scores = []
        for center in centers:
            hsv = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_BGR2HSV)[0][0]
            scores.append(float(hsv[1]) * float(hsv[2]))
            
        self.referee_cluster = int(np.argmin(scores))
        self.team_clusters = [i for i in range(3) if i != self.referee_cluster]
        self.is_fitted = True
        
        player_colors = {}
        for pid, label in zip(ids, self.kmeans.labels_):
            if label == self.referee_cluster:
                player_colors[pid] = self.referee_color
            elif label == self.team_clusters[0]:
                player_colors[pid] = self.team_colors[0]
            else:
                player_colors[pid] = self.team_colors[1]
                
        return player_colors

    def assign_player(self, raw_color):
        """
        Predicts team color for an unseen player using fitted clustering model.
        """
        if not self.is_fitted or self.kmeans is None:
            return (255, 255, 255)
            
        arr = np.array([raw_color], dtype=np.float32)
        label = self.kmeans.predict(arr)[0]
        
        if label == self.referee_cluster:
            return self.referee_color
        elif label == self.team_clusters[0]:
            return self.team_colors[0]
        else:
            return self.team_colors[1]