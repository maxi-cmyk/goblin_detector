import time

class WardenState:
    def __init__(self, max_hp=100):
        self.max_hp = max_hp
        self.current_hp = max_hp
        
        self.is_calibrated = False
        self.baseline_ratio = 0
        self.baseline_nose_y = 0
        self.baseline_shoulder_y = 0
        self.baseline_forward_lean = None
        
        self.nose_drop_tolerance = 15

        # Posture thresholds
        self.forward_lean_threshold = 0.08   # how much forward_lean can drop from baseline
        self.shoulder_tilt_threshold = 0.15  # max allowed shoulder asymmetry
        
        self.status = "IDLE"  # IDLE, GOOD, GOBLIN

        # Streak tracking
        self.streak_start = None
        self.current_streak = 0
        self.best_streak = 0

        # Session stats
        self.session_start = time.time()
        self.total_good_frames = 0
        self.total_bad_frames = 0
        self.total_game_overs = 0

        # Break reminder
        self.break_interval = 30 * 60
        self.last_break_time = time.time()
        self.show_break_reminder = False
        self.break_reminder_start = 0
        self.break_reminder_duration = 8

    def update(self, posture):
        if not self.is_calibrated:
            return

        current_ratio = posture['ratio']
        current_nose_y = posture['nose_y']
        shoulder_tilt = posture.get('shoulder_tilt', 0)

        # Check 1: Vertical slouch
        ratio_dropped = current_ratio < (self.baseline_ratio - 0.05)
        nose_actually_dropped = current_nose_y > (self.baseline_nose_y + self.nose_drop_tolerance)
        is_slouching = ratio_dropped and nose_actually_dropped

        # Check 2: Shoulder asymmetry / tilting
        is_tilting = shoulder_tilt > self.shoulder_tilt_threshold

        # Determine status based on checks
        if is_slouching:
            self.status = "BECOMING A GOBLIN?"
            self.take_damage(1.0)
            self._on_bad_posture()
        elif is_tilting:
            self.status = "LEANING SIDEWAYS!"
            self.take_damage(0.5)
            self._on_bad_posture()
        else:
            self.status = "GOOD POSTURE FINALLY"
            self.heal(0.1)
            self.total_good_frames += 1
            if self.streak_start is None:
                self.streak_start = time.time()
            self.current_streak = time.time() - self.streak_start
            if self.current_streak > self.best_streak:
                self.best_streak = self.current_streak

        # Break reminder check
        if time.time() - self.last_break_time >= self.break_interval:
            if not self.show_break_reminder:
                self.show_break_reminder = True
                self.break_reminder_start = time.time()

        if self.show_break_reminder:
            if time.time() - self.break_reminder_start >= self.break_reminder_duration:
                self.show_break_reminder = False
                self.last_break_time = time.time()

    def _on_bad_posture(self):
        self.total_bad_frames += 1
        if self.streak_start is not None:
            self.streak_start = None
            self.current_streak = 0
            
    def calibrate(self, posture):
        self.baseline_ratio = posture['ratio']
        self.baseline_nose_y = posture['nose_y']
        self.baseline_shoulder_y = posture['shoulder_y']
        self.baseline_forward_lean = posture.get('forward_lean')
        self.is_calibrated = True
        lean_str = f"{self.baseline_forward_lean:.3f}" if self.baseline_forward_lean else "N/A"
        print(f"Calibrated! Ratio: {self.baseline_ratio:.3f}, Nose Y: {self.baseline_nose_y:.1f}, Forward Lean: {lean_str}")

    def take_damage(self, amount):
        self.current_hp = max(0, self.current_hp - amount)

    def heal(self, amount):
        self.current_hp = min(self.max_hp, self.current_hp + amount)

    def get_session_summary(self):
        duration = time.time() - self.session_start
        total_frames = self.total_good_frames + self.total_bad_frames
        good_pct = (self.total_good_frames / total_frames * 100) if total_frames > 0 else 0

        mins = int(duration // 60)
        secs = int(duration % 60)
        best_m = int(self.best_streak // 60)
        best_s = int(self.best_streak % 60)

        w = 34  # inner width
        lines = [
            f"{'SESSION SUMMARY':^{w}}",
            f"  {'Duration:':<16}{mins:02d}m {secs:02d}s{'':>{w - 25}}",
            f"  {'Good Posture:':<16}{good_pct:.1f}%{'':>{w - 24 - len(f'{good_pct:.1f}')}}",
            f"  {'Best Streak:':<16}{best_m:02d}m {best_s:02d}s{'':>{w - 25}}",
            f"  {'Game Overs:':<16}{self.total_game_overs}{'':>{w - 19 - len(str(self.total_game_overs))}}",
        ]

        border = "═" * w
        sep = "═" * w
        result = f"\n╔{border}╗\n"
        result += f"║{lines[0]}║\n"
        result += f"╠{sep}╣\n"
        for line in lines[1:]:
            result += f"║{line:<{w}}║\n"
        result += f"╚{border}╝"
        return result