from sklearn.cluster import KMeans
import cv2
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        # Reshape image into a 2D array of pixels
        image_2d = image.reshape(-1, 3)
        
        # Use K-Means with 2 clusters (background and player)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        
        # Crop player from the frame
        image = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if image.size == 0:
            return (0, 0, 0)
            
        # Get the top half of the player (usually where the jersey is)
        top_half_image = image[0:int(image.shape[0]/2), :]
        
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        
        # Determine background cluster vs player cluster
        # The background typically takes up more space around the edges
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        background_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - background_cluster
        
        player_color = kmeans.cluster_centers_[player_cluster]
        return tuple(map(int, player_color))

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        
        for player_id, player_dict in player_detections.items():
            bbox = player_dict["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
            
        if not player_colors:
            return
            
        # Cluster the collected player colors into 2 teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
            
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict([player_color])[0] + 1
        
        self.player_team_dict[player_id] = team_id
        return team_id
