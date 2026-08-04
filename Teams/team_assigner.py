import cv2
import numpy as np
from collections import Counter

try:
    from sklearn.cluster import KMeans
except ImportError:
    class KMeans:
        def __init__(self, n_clusters=2, init='k-means++', n_init=10, random_state=42):
            self.n_clusters = n_clusters
            self.cluster_centers_ = None

        def fit(self, data):
            data = np.array(data, dtype=np.float32)
            if len(data) < self.n_clusters:
                self.cluster_centers_ = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.float32)
                return self

            c1 = data[0].copy()
            distances = np.linalg.norm(data - c1, axis=1)
            c2 = data[np.argmax(distances)].copy()
            centroids = np.array([c1, c2], dtype=np.float32)

            for _ in range(30):
                d0 = np.linalg.norm(data - centroids[0], axis=1)
                d1 = np.linalg.norm(data - centroids[1], axis=1)
                labels = (d1 < d0).astype(int)

                new_c0 = data[labels == 0].mean(axis=0) if np.any(labels == 0) else centroids[0]
                new_c1 = data[labels == 1].mean(axis=0) if np.any(labels == 1) else centroids[1]

                if np.allclose(centroids[0], new_c0) and np.allclose(centroids[1], new_c1):
                    break
                centroids = np.array([new_c0, new_c1], dtype=np.float32)

            self.cluster_centers_ = centroids
            return self

        def predict(self, points):
            points = np.array(points, dtype=np.float32).reshape(-1, 3)
            d0 = np.linalg.norm(points - self.cluster_centers_[0], axis=1)
            d1 = np.linalg.norm(points - self.cluster_centers_[1], axis=1)
            return (d1 < d0).astype(int)


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}  # player_id -> team_id (1 or 2)
        self.kmeans = None
        # Custom team overrides (e.g. Goalkeepers with distinct jersey colors)
        self.team_overrides = {
            35: 2  # Hardcode goalkeeper player 35 to Team 2
        }

    def get_player_color(self, frame, bbox):
        """
        Extracts jersey color by taking a center-torso crop and filtering out green grass pixels.
        """
        h, w, _ = frame.shape
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if bbox_w <= 0 or bbox_h <= 0:
            return np.array([0, 0, 0])

        # Crop the upper-torso / chest area (10%-55% height, 20%-80% width)
        y1_torso = y1 + int(bbox_h * 0.10)
        y2_torso = y1 + int(bbox_h * 0.55)
        x1_torso = x1 + int(bbox_w * 0.20)
        x2_torso = x1 + int(bbox_w * 0.80)

        torso = frame[y1_torso:y2_torso, x1_torso:x2_torso]

        if torso.size == 0:
            # Fallback to top half
            torso = frame[y1 : y1 + int(bbox_h * 0.5), x1:x2]
            if torso.size == 0:
                return np.array([0, 0, 0])

        # Convert torso crop to HSV to isolate green pitch pixels
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_green_mask = cv2.bitwise_not(green_mask)

        non_green_pixels = torso[non_green_mask > 0]

        if len(non_green_pixels) > 10:
            # Mean BGR of non-grass jersey pixels
            player_color = non_green_pixels.mean(axis=0)
        else:
            # Fallback to mean color of torso
            player_color = torso.mean(axis=(0, 1))

        return player_color

    def assign_team_colors(self, video_frames, player_tracks):
        """
        Samples player jersey colors across multiple frames to fit a global 2-cluster KMeans.
        """
        player_colors = []
        num_frames = len(video_frames)
        step = max(1, num_frames // 20)  # Sample across ~20 frames evenly

        for frame_num in range(0, num_frames, step):
            if frame_num >= len(player_tracks):
                break
            frame = video_frames[frame_num]
            players_in_frame = player_tracks[frame_num]

            for _, player in players_in_frame.items():
                bbox = player['bbox']
                player_color = self.get_player_color(frame, bbox)
                player_colors.append(player_color)

        if len(player_colors) < 2:
            # Fallback if extremely few detections
            player_colors = [np.array([255, 0, 0]), np.array([0, 0, 255])]

        # Fit global 2-cluster KMeans for the two teams
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def assign_player_teams(self, video_frames, player_tracks):
        """
        Assigns team IDs and team colors to all players across all frames using multi-frame majority voting.
        """
        # Step 1: Fit global team colors if not already done
        if self.kmeans is None:
            self.assign_team_colors(video_frames, player_tracks)

        # Step 2: Collect player track samples per player_id
        player_frame_map = {}  # player_id -> list of (frame_num, bbox)
        for frame_num, frame_players in enumerate(player_tracks):
            for player_id, player in frame_players.items():
                if player_id not in player_frame_map:
                    player_frame_map[player_id] = []
                player_frame_map[player_id].append((frame_num, player['bbox']))

        # Step 3: Compute majority vote team per player_id
        for player_id, occurrences in player_frame_map.items():
            if player_id in self.team_overrides:
                self.player_team_dict[player_id] = self.team_overrides[player_id]
                continue

            # Sample up to 15 frames across player's trajectory
            sample_step = max(1, len(occurrences) // 15)
            sampled_occurrences = occurrences[::sample_step]

            team_votes = []
            for frame_num, bbox in sampled_occurrences:
                frame = video_frames[frame_num]
                player_color = self.get_player_color(frame, bbox)
                predicted_cluster = self.kmeans.predict(player_color.reshape(1, -1))[0]
                team_id = predicted_cluster + 1
                team_votes.append(team_id)

            if team_votes:
                most_common_team = Counter(team_votes).most_common(1)[0][0]
                self.player_team_dict[player_id] = most_common_team

        # Step 4: Apply assigned team and team_color to tracks
        for frame_num, frame_players in enumerate(player_tracks):
            for player_id, player in frame_players.items():
                team = self.player_team_dict.get(player_id, 1)
                player_tracks[frame_num][player_id]['team'] = team
                player_tracks[frame_num][player_id]['team_color'] = self.team_colors[team]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Backward-compatible single player team lookup / prediction.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if player_id in self.team_overrides:
            self.player_team_dict[player_id] = self.team_overrides[player_id]
            return self.team_overrides[player_id]

        if self.kmeans is None:
            return 1

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        self.player_team_dict[player_id] = team_id
        return team_id