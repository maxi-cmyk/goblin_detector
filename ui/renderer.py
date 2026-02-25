import cv2
import numpy as np
import time
import os
import threading
import random
import math

class GameRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)
        self.white = (255, 255, 255)
        self.goblin_green = (0, 180, 0)

        # Stink particles: list of dicts with x, y, vx, vy, life, max_life, wave_offset
        self.particles = []
        self.particle_spawn_timer = 0

        # Game-over flash state
        self.game_over_triggered = False
        self.game_over_start = 0
        self.flash_duration = 5

        # Goblin shame state
        self.goblin_shame_active = False
        self.goblin_shame_start = 0
        self.goblin_shame_duration = 5

        # Load the shrimp image once
        asset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
        img_path = os.path.join(asset_dir, 'shrimp_posture.png')
        self.shrimp_img = cv2.imread(img_path)
        if self.shrimp_img is None:
            print(f"WARNING: Could not load {img_path}")

        self.audio_path = os.path.join(asset_dir, 'cat-laugh-meme-1.mp3')

    def _play_audio(self):
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(self.audio_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Audio playback error: {e}")

    def draw_skeleton(self, frame, p1, p2):
        if p1 is None or p2 is None:
            return
        point1 = (int(p1[0]), int(p1[1]))
        point2 = (int(p2[0]), int(p2[1]))
        cv2.line(frame, point1, point2, self.green, 3)
        cv2.circle(frame, point1, 5, self.red, -1)
        cv2.circle(frame, point2, 5, self.red, -1)

    def spawn_stink_particles(self, keypoints):
        """Spawn new stink particles near nose and shoulders."""
        now = time.time()
        if now - self.particle_spawn_timer < 0.03:  # spawn every 30ms
            return
        self.particle_spawn_timer = now

        spawn_points = [keypoints['nose'], keypoints['left_shoulder'], keypoints['right_shoulder']]
        for pt in spawn_points:
            for _ in range(2):  # Spawn 2 particles per point to make it thicker
                self.particles.append({
                    'x': pt[0] + random.randint(-25, 25),
                    'y': pt[1] + random.randint(-10, 10),
                    'vy': -random.uniform(2.0, 5.0),  # float upward faster
                    'life': 0,
                    'max_life': random.randint(25, 45),  # frames to live
                    'wave': random.uniform(0, math.pi * 2),  # phase offset for sine wave
                    'wave_speed': random.uniform(0.3, 0.6),  # faster waving
                    'wave_amp': random.uniform(4, 12),       # wider amplitude
                })

    def update_and_draw_particles(self, frame):
        """Update particle positions and draw them as wavy stink lines."""
        alive = []
        for p in self.particles:
            p['life'] += 1
            if p['life'] >= p['max_life']:
                continue

            # Move upward
            p['y'] += p['vy']
            # Wavy horizontal drift
            p['wave'] += p['wave_speed']
            wave_x = math.sin(p['wave']) * p['wave_amp']
            
            # Predict next position to draw a line segment connecting the path
            next_wave_x = math.sin(p['wave'] + p['wave_speed']) * p['wave_amp']

            # Fade: starts at 1.0, fades to 0
            fade = 1.0 - (p['life'] / p['max_life'])
            green_val = int(220 * fade)  # brighter green
            color = (0, green_val, 0)

            # Draw a small wavy line segment along the trajectory
            x1 = int(p['x'] + wave_x)
            y1 = int(p['y'])
            x2 = int(p['x'] + next_wave_x)
            y2 = int(p['y'] + p['vy'] * 1.5) # Stretch the line a bit
            
            cv2.line(frame, (x1, y1), (x2, y2), color, 3)

            alive.append(p)

        self.particles = alive

    def draw_goblin_face(self, frame, keypoints):
        nose = keypoints['nose']
        left_sh = keypoints['left_shoulder']
        right_sh = keypoints['right_shoulder']

        cx, cy = int(nose[0]), int(nose[1])
        sh_width = abs(left_sh[0] - right_sh[0])
        face_r = max(int(sh_width * 0.35), 30)

        color = self.goblin_green
        thick = 3

        # Pointy Ears
        for side in [-1, 1]:
            base = (cx + side * face_r, cy - int(face_r * 0.1))
            tip = (cx + side * (face_r + int(face_r * 0.7)), cy - int(face_r * 0.8))
            bot = (cx + side * face_r, cy + int(face_r * 0.3))
            cv2.polylines(frame, [np.array([base, tip, bot], np.int32)], True, color, thick)

        # Horns
        for side in [-1, 1]:
            b1 = (cx + side * int(face_r * 0.4), cy - face_r)
            tip = (cx + side * int(face_r * 0.6), cy - face_r - int(face_r * 0.9))
            b2 = (cx + side * int(face_r * 0.2), cy - face_r)
            cv2.polylines(frame, [np.array([b1, tip, b2], np.int32)], True, color, thick)

        # Fangs
        fang_top = cy + int(face_r * 0.55)
        for side in [-1, 1]:
            cv2.line(frame, (cx + side * int(face_r * 0.25), fang_top),
                     (cx + side * int(face_r * 0.2), fang_top + int(face_r * 0.35)), color, thick)

        # Label
        label = "GOBLIN"
        text_size = cv2.getTextSize(label, self.font, 1.2, 3)[0]
        text_x = cx - text_size[0] // 2
        text_y = cy + face_r + int(face_r * 0.9) + 10
        cv2.putText(frame, label, (text_x, text_y), self.font, 1.2, color, 3)

    def _hp_color(self, hp_pct):
        """Return BGR color that smoothly transitions green → yellow → red."""
        if hp_pct > 0.5:
            # Green (0,255,0) → Yellow (0,255,255)
            t = (1.0 - hp_pct) / 0.5  # 0 at 100%, 1 at 50%
            return (0, 255, int(255 * t))
        else:
            # Yellow (0,255,255) → Red (0,0,255)
            t = (0.5 - hp_pct) / 0.5  # 0 at 50%, 1 at 0%
            return (0, int(255 * (1 - t)), 255)

    def draw_ui(self, frame, state, keypoints=None):
        h, w, _ = frame.shape

        # 1. Calibration Instructions
        if not state.is_calibrated:
            cv2.putText(frame, "SIT STRAIGHT & PRESS 'C'", (10, h - 40), 
                        self.font, 0.8, self.white, 2)
            return

        # 2. Goblin shame overlay — takes priority over game-over check
        if self.goblin_shame_active:
            elapsed = time.time() - self.goblin_shame_start
            if elapsed < self.goblin_shame_duration:
                state.current_hp = max(state.current_hp, 30)
                if keypoints is not None:
                    self.draw_goblin_face(frame, keypoints)
            else:
                self.goblin_shame_active = False

        # 3. Game Over — flash shrimp image for 5 seconds
        elif self.game_over_triggered or state.current_hp <= 0:
            if not self.game_over_triggered:
                self.game_over_triggered = True
                self.game_over_start = time.time()
                state.total_game_overs += 1
                threading.Thread(target=self._play_audio, daemon=True).start()

            state.current_hp = 0
            elapsed = time.time() - self.game_over_start
            if elapsed < self.flash_duration:
                if self.shrimp_img is not None:
                    resized = cv2.resize(self.shrimp_img, (w, h))
                    frame[:] = resized
                return
            else:
                self.game_over_triggered = False
                self.goblin_shame_active = True
                self.goblin_shame_start = time.time()
                state.current_hp = 30
                state.status = "GOOD"
                try:
                    import pygame
                    pygame.mixer.music.stop()
                except Exception:
                    pass
                return

        # 4. HP Bar with gradient color
        bar_x, bar_y = 50, 50
        bar_w, bar_h = 300, 30
        hp_pct = state.current_hp / state.max_hp
        fill_width = int(hp_pct * bar_w)
        
        bar_color = self._hp_color(hp_pct)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.white, 2)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_h), bar_color, -1)
        
        cv2.putText(frame, f"HP: {int(state.current_hp)}", (bar_x, bar_y - 10), 
                    self.font, 0.7, self.white, 2)

        # 5. Status Text
        status_color = self.green if "GOOD" in state.status else self.red
        cv2.putText(frame, f"STATUS: {state.status}", (50, 130), 
                    self.font, 0.7, status_color, 2)

        # 6. Stink Particles (spawn when bad posture, always draw)
        is_bad_posture = "GOOD" not in state.status
        if (is_bad_posture or self.goblin_shame_active) and keypoints is not None:
            self.spawn_stink_particles(keypoints)
        self.update_and_draw_particles(frame)

        # 6. Streak Counter (bottom-right)
        streak_s = int(state.current_streak)
        streak_m = streak_s // 60
        streak_s = streak_s % 60
        streak_text = f"STREAK: {streak_m:02d}:{streak_s:02d}"
        text_size = cv2.getTextSize(streak_text, self.font, 0.7, 2)[0]
        cv2.putText(frame, streak_text, (w - text_size[0] - 15, h - 20),
                    self.font, 0.7, self.green, 2)

        # 7. Best Streak (bottom-right, above current)
        best_s = int(state.best_streak)
        best_m = best_s // 60
        best_s = best_s % 60
        best_text = f"BEST: {best_m:02d}:{best_s:02d}"
        text_size2 = cv2.getTextSize(best_text, self.font, 0.5, 1)[0]
        cv2.putText(frame, best_text, (w - text_size2[0] - 15, h - 45),
                    self.font, 0.5, (255, 150, 0), 1)

        # 8. Break Reminder
        if state.show_break_reminder:
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//2 - 200, h//2 - 40), (w//2 + 200, h//2 + 40), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.putText(frame, "TAKE A BREAK!", (w//2 - 150, h//2 + 12),
                        self.font, 1.2, (0, 200, 255), 3)