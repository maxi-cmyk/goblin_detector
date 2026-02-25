import cv2
import time
from core.detector import PoseDetector
from core.geometry import calculate_posture_metric
from core.state import WardenState
from ui.renderer import GameRenderer

def main():
    # 1. Initialize Components
    detector = PoseDetector(model_path='yolov8n-pose.pt')
    
    state = WardenState()
    
    renderer = GameRenderer()
    
    # 2. Open and set up Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    print("--- SPINE WARDEN INITIALIZED ---")
    print("Rules: Sit straight, press 'C' to calibrate.")
    print("Press 'Q' to quit.")

    # Variables for FPS calculation
    prev_time = 0

    while True:
        # 3. Read Frame
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally so it acts like a mirror (easier for the user)
        frame = cv2.flip(frame, 1)

        # 4. AI Inference
        # Returns a dictionary: {'nose': (x,y), 'left_shoulder': (x,y), 'right_shoulder': (x,y)}
        keypoints = detector.get_keypoints(frame)

        # 5. Logic Pipeline
        if keypoints is not None:
            # A. Calculate the Metric (returns dict with ratio, nose_y, shoulder_y)
            posture = calculate_posture_metric(keypoints)
            
            # B. Update Game State
            # This handles the HP drain logic if the ratio drops too low
            state.update(posture)
            
            # C. Visualization (Debug Lines)
            nose_x, nose_y = int(keypoints['nose'][0]), int(keypoints['nose'][1])
            
            # Midpoint between shoulders
            mid_x = int((keypoints['left_shoulder'][0] + keypoints['right_shoulder'][0]) / 2)
            mid_y = int((keypoints['left_shoulder'][1] + keypoints['right_shoulder'][1]) / 2)

            # Draw the critical measurement line (Yellow)
            cv2.line(frame, (nose_x, nose_y), (mid_x, mid_y), (0, 255, 255), 2)
            
            # Draw shoulder width line (Blue) - to show the "Ruler"
            l_sh = (int(keypoints['left_shoulder'][0]), int(keypoints['left_shoulder'][1]))
            r_sh = (int(keypoints['right_shoulder'][0]), int(keypoints['right_shoulder'][1]))
            cv2.line(frame, l_sh, r_sh, (255, 0, 0), 2)

            # Draw dots on keypoints
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)

            # D. Handle Input
            keys = cv2.waitKey(1) & 0xFF
            if keys == ord('c'):
                state.calibrate(posture)
                print(f"Calibration Set! Ratio: {posture['ratio']:.4f}")
            elif keys == ord('q'):
                break
        else:
            # If no person is detected, just check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 6. Render UI
        # Pass the frame and state to the renderer to draw HP bars and text
        renderer.draw_ui(frame, state, keypoints)

        # Calculate and display FPS (Tech Flex)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        fh = frame.shape[0]
        cv2.putText(frame, f"FPS: {int(fps)}", (10, fh - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        # 7. Show Output
        cv2.imshow("Spine Warden: Goblin Mode Detector", frame)

    # Session Summary
    print(state.get_session_summary())

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()