from ultralytics import YOLO
import numpy as np

class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.pt'):
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
    
    def get_keypoints(self, frame):
        results = self.model(frame, verbose=False)
        
        if not results or len(results[0].keypoints) == 0:
            return None

        kp = results[0].keypoints.data[0].cpu().numpy()
        
        # YOLO Indices: Nose=0, Left Eye=1, Right Eye=2,
        # Left Ear=3, Right Ear=4, Left Shoulder=5, Right Shoulder=6
        conf_threshold = 0.5

        # Required keypoints
        if kp[0][2] < conf_threshold or kp[5][2] < conf_threshold or kp[6][2] < conf_threshold:
            return None

        result = {
            'nose': kp[0][:2],
            'left_shoulder': kp[5][:2],
            'right_shoulder': kp[6][:2],
        }

        # Optional: ears (for forward head detection)
        if kp[3][2] >= conf_threshold:
            result['left_ear'] = kp[3][:2]
        if kp[4][2] >= conf_threshold:
            result['right_ear'] = kp[4][:2]

        return result