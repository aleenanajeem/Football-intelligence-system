import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm

from utils.video_utils import read_video_generator
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
import argparse

def process_video(video_path):
    print("Initializing components...")
    model_path = 'yolov8n.pt'
    tracker = Tracker(model_path)
    
    # PASS 1: Tracking
    stub_track = 'stubs/track_stubs.pkl'
    if os.path.exists(stub_track):
        print("Loading tracks from stub...")
        with open(stub_track, 'rb') as f:
            tracks = pickle.load(f)
    else:
        print("Pass 1: Tracking objects...")
        tracks = {"players": [], "referees": [], "ball": []}
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total, desc="Tracking")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            results = tracker.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=640)[0]
            
            dict_players = {}
            dict_referees = {}
            dict_ball = {}
            
            for result in results.boxes.data.tolist():
                if len(result) == 6:
                    x1, y1, x2, y2, track_id, class_id = result
                elif len(result) == 5:
                    x1, y1, x2, y2, class_id = result
                    track_id = 1
                elif len(result) == 7:
                    x1, y1, x2, y2, track_id, conf, class_id = result
                else: continue
                
                track_id = int(track_id)
                class_id = int(class_id)
                
                if class_id == 0:
                    dict_players[track_id] = {"bbox": [x1, y1, x2, y2], "class_id": class_id}
                elif class_id == 32:
                    dict_ball[track_id] = {"bbox": [x1, y1, x2, y2], "class_id": class_id}

            tracks["players"].append(dict_players)
            tracks["referees"].append(dict_referees)
            tracks["ball"].append(dict_ball)
            pbar.update(1)
            
        pbar.close()
        cap.release()
        os.makedirs('stubs', exist_ok=True)
        with open(stub_track, 'wb') as f:
            pickle.dump(tracks, f)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    tracker.add_position_to_tracks(tracks)

    # PASS 2: Camera Movement
    stub_cam = 'stubs/camera_movement_stub.pkl'
    if os.path.exists(stub_cam):
        print("Loading camera movement from stub...")
        with open(stub_cam, 'rb') as f:
            camera_movement_per_frame = pickle.load(f)
    else:
        print("Pass 2: Estimating camera movement...")
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total, desc="Camera")
        
        # We need the first frame
        ret, frame0 = cap.read()
        camera_estimator = CameraMovementEstimator(frame0)
        pbar.update(1)
        
        gen = read_video_generator(video_path)
        camera_movement_per_frame = camera_estimator.get_camera_movement(gen)
        cap.release()
        pbar.close()
        
        # We bypassed the stub saving in get_camera_movement due to generator change, so save it here:
        with open(stub_cam, 'wb') as f:
            pickle.dump(camera_movement_per_frame, f)

    # Initialize estimator just for adjust
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    cap.release()
    camera_estimator = CameraMovementEstimator(frame0)
    camera_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    print("Transforming View...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # PASS 3: Team Assignment
    print("Pass 3: Assigning Teams...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frame0, tracks['players'][0])
    
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc="Teams")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_num < len(tracks['players']):
            for player_id, track in tracks['players'][frame_num].items():
                team = team_assigner.get_player_team(frame, track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        
        frame_num += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    print("Assigning Ball Possession...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        if frame_num >= len(tracks['ball']):
            ball_bbox = []
        else:
            ball_bbox_dict = tracks['ball'][frame_num].get(1, {})
            ball_bbox = ball_bbox_dict.get('bbox', []) if isinstance(ball_bbox_dict, dict) else []
            
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) > 0 else 0)

    print("Estimating Speed and Distance...")
    speed_estimator = SpeedAndDistanceEstimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    print("Saving fully processed tracks to stubs/final_tracks.pkl...")
    final_data = {
        "tracks": tracks,
        "team_ball_control": team_ball_control,
        "camera_movement": camera_movement_per_frame
    }
    with open('stubs/final_tracks.pkl', 'wb') as f:
        pickle.dump(final_data, f)
    print("Done! Engine pass is complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    args = parser.parse_args()
    process_video(args.video_path)
