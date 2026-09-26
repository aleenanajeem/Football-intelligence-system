import cv2
import numpy as np
import os
import pickle
import math
import json
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from collections import defaultdict, deque
from utils import measure_distance

# DESIGN COLORS
C_TEAM_A        = (220, 220, 220)
C_TEAM_A_LABEL  = (255, 255, 255)
C_TEAM_A_BG     = (30,  30,  50)

C_TEAM_B        = (50,  130, 255)
C_TEAM_B_LABEL  = (140, 195, 255)
C_TEAM_B_BG     = (10,  15,  40)

C_BALL     = (0,   220, 255)
C_WHITE    = (255, 255, 255)
C_BLACK    = (0,   0,   0)
C_PANEL_BG = (15,  20,  30)
C_SELECT   = (0,   255, 100)

LABEL_FONT = cv2.FONT_HERSHEY_DUPLEX

def get_direction_string(dx, dy):
    # Using 5.0 pixel displacement over 15 frames as threshold for Stationary
    if math.sqrt(dx**2 + dy**2) < 5.0: return "Stationary"
    angle = math.degrees(math.atan2(-dy, dx))
    if angle < 0: angle += 360
    directions = ["East", "North-East", "North", "North-West", "West", "South-West", "South", "South-East"]
    idx = int((angle + 22.5) / 45.0) % 8
    return directions[idx]

def get_action_string(speed):
    if speed < 1.5: return "Standing"
    elif speed < 6.0: return "Walking"
    elif speed < 12.0: return "Jogging"
    elif speed < 20.0: return "Running"
    else: return "Sprinting"

def get_zone_and_calibration(pos_transformed):
    if pos_transformed is None: return "Uncalibrated", "Uncalibrated"
    x, y = pos_transformed
    # Strict boundary check (standard pitch 105m x 68m)
    if x < 0 or x > 105 or y < 0 or y > 68: return "Uncalibrated", "Uncalibrated"
    
    if x < 105 / 3: zone = "Defence"
    elif x > 2 * 105 / 3: zone = "Attack"
    else: zone = "Midfield"
    return zone, "Valid"

class UIDrawer:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.scale = max(0.5, H / 1080.0)
        self.font = LABEL_FONT
        self.overlay = np.zeros((H, W, 3), dtype=np.uint8)

    def draw_text(self, img, text, x, y, font_scale, color, thick):
        cv2.putText(img, text, (x, y), self.font, font_scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), self.font, font_scale, color, thick, cv2.LINE_AA)

    def draw_unselected_player(self, canvas, cx, cy, team_id):
        col = C_TEAM_A if team_id == 1 else C_TEAM_B if team_id == 2 else (150, 150, 150)
        r = max(4, int(6 * self.scale))
        cv2.circle(canvas, (int(cx), int(cy)), r, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(cx), int(cy)), r, (0,0,0), 1, cv2.LINE_AA)

    def draw_selected_player_tracking(self, canvas, cx, cy, x1, y1, x2, y2, team_id):
        rx = max((x2 - x1) // 2 + 10, 15)
        ry = max(int(rx * 0.3), 6)
        
        # True glowing effect
        glow_layer = np.zeros_like(canvas)
        # Draw multiple concentric ellipses with decreasing opacity
        for i in range(3, 0, -1):
            thickness = i * 3
            cv2.ellipse(glow_layer, (cx, y2), (rx+thickness, ry+thickness), 0, 0, 360, C_SELECT, -1, cv2.LINE_AA)
        
        # Blur the glow layer
        glow_layer = cv2.GaussianBlur(glow_layer, (15, 15), 0)
        
        # Add the glow to the canvas using a mask where glow is present
        mask = glow_layer.astype(bool)
        canvas[mask] = cv2.addWeighted(canvas, 0.3, glow_layer, 0.7, 0)[mask]
        
        cv2.ellipse(canvas, (cx, y2), (rx, ry), 0, 0, 360, C_SELECT, max(2, int(3 * self.scale)), cv2.LINE_AA)
        
        scale = 0.5 * self.scale
        thick = max(1, int(1.5 * self.scale))
        text = "SELECTED"
        (tw, th), _ = cv2.getTextSize(text, self.font, scale, thick)
        self.draw_text(canvas, text, cx - tw//2, max(th+5, y1 - 10), scale, C_SELECT, thick)

    def draw_selected_panel(self, canvas, tid, team_id, spd, dist, dx, dy, has_ball, zone, dist_to_ball, ball_status, calibration_status):
        scale = 0.45 * self.scale
        thick = max(1, int(1.2 * self.scale))
        pad = int(15 * self.scale)
        
        action = get_action_string(spd)
        direction = get_direction_string(dx, dy)
        activity_lvl = min(100, int((spd / 25.0) * 100))
        
        hb_str = "Yes" if has_ball else "No"
        t_name = "Team A" if team_id == 1 else "Team B" if team_id == 2 else "Unknown"
        
        spd_str = f"{spd:.1f} km/h" if calibration_status == "Valid" else f"Estimated {spd:.1f} km/h"
        
        if calibration_status == "Uncalibrated":
            dist_b_str = "Not Available"
        elif dist_to_ball is not None:
            dist_b_str = f"{dist_to_ball:.1f} m"
        else:
            dist_b_str = "Not Available"

        lines = [
            f"PLAYER ID: {tid}",
            f"TEAM: {t_name}",
            f"SPEED: {spd_str}",
            f"DISTANCE: {dist:.1f} m",
            f"ACTION: {action}",
            f"DIRECTION: {direction}",
            f"BALL DIST: {dist_b_str}",
            f"BALL STATUS: {ball_status}",
            f"POSSESSION: {hb_str}",
            f"ZONE: {zone}",
            f"CALIBRATION: {calibration_status}"
        ]
        
        sizes = [cv2.getTextSize(l, self.font, scale, thick)[0] for l in lines]
        max_w = max(s[0] for s in sizes)
        total_h = sum(s[1] for s in sizes) + int(10*self.scale) * (len(lines)-1)
        
        bx = pad
        by = self.H // 2 - total_h // 2
        
        np.copyto(self.overlay, canvas)
        cv2.rectangle(self.overlay, (bx, by), (bx + max_w + pad*2, by + total_h + pad*2), C_PANEL_BG, -1)
        cv2.rectangle(self.overlay, (bx, by), (bx + max_w + pad*2, by + total_h + pad*2), C_SELECT, max(1, int(2*self.scale)))
        cv2.addWeighted(self.overlay, 0.9, canvas, 0.1, 0, canvas)
        
        tcol = C_TEAM_A_LABEL if team_id == 1 else C_TEAM_B_LABEL if team_id == 2 else C_WHITE
        y_cur = by + pad
        for i, (l, (w, h)) in enumerate(zip(lines, sizes)):
            col = C_SELECT if i == 0 else tcol if i == 1 else C_WHITE
            self.draw_text(canvas, l, bx + pad, y_cur+h, scale, col, thick)
            y_cur += h + int(10*self.scale)



    def draw_bottom_possession(self, canvas, pa, pb):
        bar_w = int(400 * self.scale)
        bar_h = int(12 * self.scale)
        bx = self.W // 2 - bar_w // 2
        by = self.H - int(60 * self.scale) - bar_h

        pad = int(10 * self.scale)
        np.copyto(self.overlay, canvas)
        cv2.rectangle(self.overlay, (bx - pad, by - int(30*self.scale)), (bx + bar_w + pad, by + bar_h + pad), C_PANEL_BG, -1)
        cv2.addWeighted(self.overlay, 0.85, canvas, 0.15, 0, canvas)

        af = int(bar_w * pa / 100)
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 40), -1)
        if af > 0: cv2.rectangle(canvas, (bx, by), (bx + af, by + bar_h), C_TEAM_A, -1)
        if af < bar_w: cv2.rectangle(canvas, (bx + af, by), (bx + bar_w, by + bar_h), C_TEAM_B, -1)
        
        scale = 0.45 * self.scale
        thick = max(1, int(1.2 * self.scale))
        l1 = f"Team A: {pa}%"
        l2 = f"Team B: {pb}%"
        
        s1 = cv2.getTextSize(l1, self.font, scale, thick)[0]
        s2 = cv2.getTextSize(l2, self.font, scale, thick)[0]
        
        self.draw_text(canvas, l1, bx, by - int(8*self.scale), scale, C_TEAM_A, thick)
        self.draw_text(canvas, l2, bx + bar_w - s2[0], by - int(8*self.scale), scale, C_TEAM_B, thick)

    def draw_minimap(self, canvas, players_dict, selected_id):
        mw = int(240 * self.scale)
        mh = int(150 * self.scale)
        pad = int(15 * self.scale)
        mx = self.W - mw - pad
        my = self.H - mh - int(40*self.scale) - pad

        np.copyto(self.overlay, canvas)
        cv2.rectangle(self.overlay, (mx, my), (mx + mw, my + mh), (15, 25, 20), -1)
        cv2.addWeighted(self.overlay, 0.85, canvas, 0.15, 0, canvas)
        
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (40, 100, 50), max(1, int(1.5*self.scale)))
        cv2.line(canvas, (mx + mw//2, my), (mx + mw//2, my + mh), (40, 100, 50), 1)
        cv2.ellipse(canvas, (mx + mw//2, my + mh//2), (int(25*self.scale), int(15*self.scale)), 0, 0, 360, (40, 100, 50), 1)

        # Draw players on minimap based on transformed position
        for tid, info in players_dict.items():
            pos_t = info.get('position_transformed')
            if not pos_t: continue
            
            # pitch is 105x68
            px = int(mx + (pos_t[0] / 105.0) * mw)
            py = int(my + (pos_t[1] / 68.0) * mh)
            px = np.clip(px, mx + 3, mx + mw - 3)
            py = np.clip(py, my + 3, my + mh - 3)
            
            team_id = info.get("team", 0)
            col = C_TEAM_A if team_id == 1 else C_TEAM_B if team_id == 2 else (150, 150, 150)
            
            if tid == selected_id:
                cv2.circle(canvas, (px, py), int(6 * self.scale), C_SELECT, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (px, py), int(3 * self.scale) + 1, col, -1, cv2.LINE_AA)

    def draw_footer(self, canvas):
        scale = 0.45 * self.scale
        thick = max(1, int(1.2 * self.scale))
        text = "Football Match Intelligence System | Developed by Aleena"
        h_bar = int(26 * self.scale)
        
        np.copyto(self.overlay, canvas)
        cv2.rectangle(self.overlay, (0, self.H - h_bar), (self.W, self.H), (5, 5, 10), -1)
        cv2.addWeighted(self.overlay, 0.9, canvas, 0.1, 0, canvas)
        
        (w, h), _ = cv2.getTextSize(text, self.font, scale, thick)
        self.draw_text(canvas, text, self.W // 2 - w // 2, self.H - h_bar // 2 + h // 2, scale, (180, 180, 200), thick)


class InteractivePlayer:
    def __init__(self, video_path, data_path, interactive=False, selection_script="", out_path="football_output.mp4"):
        self.video_path = video_path
        self.out_path = out_path
        self.interactive = interactive
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing data file: {data_path}. Please run engine.py first.")
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.tracks = self.data["tracks"]
        self.team_ball_control = self.data["team_ball_control"]
        self.camera_movement = self.data["camera_movement"]
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.writer = self._make_writer()
        
        self.selected_player_id = None
        self.click_point = None
        self.player_history = defaultdict(lambda: {"speed": deque(maxlen=15), "pos": deque(maxlen=15)})
        
        self.last_valid_ball_pos_t = None
        self.ball_missing_frames = 0
        
        if self.interactive:
            self.win_name = "Football Match Intelligence System (Zero Lag HUD)"
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.win_name, self._mouse_cb)

        self.script_timeline = {}
        if selection_script and os.path.exists(selection_script):
            with open(selection_script, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                parts = k.split(":")
                secs = int(parts[0]) * 60 + int(parts[1])
                self.script_timeline[secs] = v
        self.last_script_sec = -1

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)

    def _make_writer(self):
        for fc in ["mp4v", "avc1"]:
            w = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*fc), self.fps, (self.W, self.H))
            if w.isOpened(): return w
            w.release()
        raise RuntimeError("No codec found")

    def play(self):
        drawer = UIDrawer(self.W, self.H)
        frame_idx = 0
        
        hist_spd, hist_act = [], []
        t1_count = 0
        t2_count = 0
        
        with tqdm(total=self.total, unit="fr", ncols=80, colour="green") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                if frame_idx >= len(self.tracks["players"]): break
                
                canvas = frame
                players_dict = self.tracks["players"][frame_num := frame_idx]
                ball_dict = self.tracks["ball"][frame_num] if frame_num < len(self.tracks["ball"]) else {}
                
                current_sec = int(frame_idx / self.fps)
                if current_sec in self.script_timeline and current_sec != self.last_script_sec:
                    self.selected_player_id = self.script_timeline[current_sec]
                    self.last_script_sec = current_sec

                if self.interactive and self.click_point:
                    cx, cy = self.click_point
                    found = False
                    for tid, info in players_dict.items():
                        tx1,ty1,tx2,ty2 = info["bbox"]
                        if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                            self.selected_player_id = tid
                            found = True
                            break
                    if not found: self.selected_player_id = None
                    self.click_point = None

                if self.interactive:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27: # ESC
                        self.selected_player_id = None
                    elif key in [ord('n'), ord('N')]:
                        if players_dict:
                            keys = list(players_dict.keys())
                            if self.selected_player_id in keys:
                                idx = (keys.index(self.selected_player_id) + 1) % len(keys)
                                self.selected_player_id = keys[idx]
                            else:
                                self.selected_player_id = keys[0]
                    elif key in [ord('p'), ord('P')]:
                        if players_dict:
                            keys = list(players_dict.keys())
                            if self.selected_player_id in keys:
                                idx = (keys.index(self.selected_player_id) - 1) % len(keys)
                                self.selected_player_id = keys[idx]
                            else:
                                self.selected_player_id = keys[-1]

                # Draw Ball
                for bid, b_info in ball_dict.items():
                    bx1, by1, bx2, by2 = b_info["bbox"]
                    if math.isnan(bx1) or math.isnan(by1) or math.isnan(bx2) or math.isnan(by2):
                        continue
                    bx, by = int((bx1 + bx2)/2), int((by1 + by2)/2)
                    cv2.circle(canvas, (bx, by), max(4, int(6*drawer.scale)), C_WHITE, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (bx, by), max(4, int(6*drawer.scale)), C_BALL, 2, cv2.LINE_AA)

                # Ball Tracking Logic with Grace Period
                current_ball_pos_t = None
                if ball_dict:
                    ball_info = list(ball_dict.values())[0]
                    current_ball_pos_t = ball_info.get("position_transformed")
                
                if current_ball_pos_t is not None:
                    self.last_valid_ball_pos_t = current_ball_pos_t
                    self.ball_missing_frames = 0
                    ball_status = "Detected"
                else:
                    self.ball_missing_frames += 1
                    if self.ball_missing_frames <= 10 and self.last_valid_ball_pos_t is not None:
                        ball_status = "Temporarily estimated"
                        current_ball_pos_t = self.last_valid_ball_pos_t
                    else:
                        ball_status = "Not detected"
                        current_ball_pos_t = None

                spds = []
                # Draw Players
                for tid, info in players_dict.items():
                    x1,y1,x2,y2 = [int(v) for v in info["bbox"]]
                    cx, cy = int((x1+x2)/2), int(y2)
                    team_id = info.get("team", 0)
                    
                    spd_raw = info.get("speed", 0.0)
                    if spd_raw is not None:
                        self.player_history[tid]["speed"].append(spd_raw)
                    
                    spd = float(np.mean(self.player_history[tid]["speed"])) if len(self.player_history[tid]["speed"]) > 0 else 0.0
                    spds.append(spd)
                    
                    pos_adj = info.get("position_adjusted")
                    if pos_adj is not None:
                        self.player_history[tid]["pos"].append(pos_adj)
                    
                    if tid == self.selected_player_id:
                        drawer.draw_selected_player_tracking(canvas, cx, cy, x1, y1, x2, y2, team_id)
                        
                        dist = info.get("distance", 0.0) or 0.0
                        pos_t = info.get("position_transformed")
                        zone, calibration_status = get_zone_and_calibration(pos_t)
                        has_ball = info.get("has_ball", False)
                        
                        dist_to_ball = None
                        if current_ball_pos_t and pos_t and calibration_status == "Valid":
                            dist_to_ball = measure_distance(pos_t, current_ball_pos_t)
                        
                        # Derive direction from camera adjusted position over up to 15 frames
                        dx, dy = 0.0, 0.0
                        pos_q = self.player_history[tid]["pos"]
                        if len(pos_q) > 1:
                            dx = pos_q[-1][0] - pos_q[0][0]
                            dy = pos_q[-1][1] - pos_q[0][1]
                        
                        drawer.draw_selected_panel(canvas, tid, team_id, spd, dist, dx, dy, has_ball, zone, dist_to_ball, ball_status, calibration_status)
                    else:
                        drawer.draw_unselected_player(canvas, cx, y2, team_id)


                
                # Possession logic
                if frame_idx < len(self.team_ball_control):
                    control = self.team_ball_control[frame_idx]
                    if control == 1:
                        t1_count += 1
                    elif control == 2:
                        t2_count += 1
                
                tot = t1_count + t2_count
                pa = round(t1_count/tot*100, 1) if tot > 0 else 50.0
                pb = round(t2_count/tot*100, 1) if tot > 0 else 50.0
                
                drawer.draw_bottom_possession(canvas, pa, pb)
                drawer.draw_minimap(canvas, players_dict, self.selected_player_id)
                drawer.draw_footer(canvas)

                if self.interactive:
                    win_scale = 1080.0 / float(max(self.H, 1))
                    if win_scale < 1.0:
                        disp = cv2.resize(canvas, (0,0), fx=win_scale, fy=win_scale)
                        cv2.imshow(self.win_name, disp)
                    else:
                        cv2.imshow(self.win_name, canvas)
                    
                self.writer.write(canvas)
                frame_idx += 1
                pbar.update(1)

        self.cap.release()
        self.writer.release()
        if self.interactive:
            cv2.destroyAllWindows()
        print("[DONE] football_output.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--interactive", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--selection-script", type=str, default="")
    parser.add_argument("--data", type=str, default="stubs/final_tracks.pkl", help="Pre-computed physics tracks")
    args = parser.parse_args()
    
    player = InteractivePlayer(args.video_path, args.data, args.interactive, args.selection_script)
    player.play()
