import math

def calculate_posture_metric(keypoints):
    """
    Calculates posture metrics from nose, shoulder, and ear keypoints.
    
    Returns dict with:
        'ratio':          nose-to-shoulder vertical distance / shoulder width
        'nose_y':         raw Y of nose
        'shoulder_y':     raw Y of shoulder midpoint
        'forward_lean':   how far the head is forward (ear-shoulder gap) / shoulder width
                          Lower = more forward lean. None if ears not visible.
        'shoulder_tilt':  abs vertical difference between shoulders / shoulder width
                          0 = level, higher = more asymmetric tilt.
    """
    nose = keypoints['nose']
    left_sh = keypoints['left_shoulder']
    right_sh = keypoints['right_shoulder']
    
    mid_shoulder_x = (left_sh[0] + right_sh[0]) / 2
    mid_shoulder_y = (left_sh[1] + right_sh[1]) / 2
    
    vertical_dist = mid_shoulder_y - nose[1]
    
    shoulder_width = math.sqrt((left_sh[0] - right_sh[0])**2 + (left_sh[1] - right_sh[1])**2)
    
    if shoulder_width == 0:
        return {
            'ratio': 0, 'nose_y': nose[1], 'shoulder_y': mid_shoulder_y,
            'forward_lean': None, 'shoulder_tilt': 0
        }
        
    ratio = vertical_dist / shoulder_width

    # --- Forward Head Detection ---
    # When the head juts forward toward a screen, the ears drop closer
    # to shoulder height (in a front-facing camera, the ear-to-shoulder
    # vertical gap shrinks). measure the average ear Y relative to
    # shoulder midpoint Y, normalised by shoulder width.
    forward_lean = None
    left_ear = keypoints.get('left_ear')
    right_ear = keypoints.get('right_ear')
    
    ears = [e for e in [left_ear, right_ear] if e is not None]
    if ears:
        avg_ear_y = sum(e[1] for e in ears) / len(ears)
        # How high the ear is above the shoulders (bigger = better posture)
        ear_shoulder_gap = mid_shoulder_y - avg_ear_y
        forward_lean = ear_shoulder_gap / shoulder_width

    # --- Shoulder Tilt / Asymmetry ---
    # Absolute vertical difference between left and right shoulder
    # normalised by shoulder width. 0 = perfectly level.
    shoulder_tilt = abs(left_sh[1] - right_sh[1]) / shoulder_width
    
    return {
        'ratio': ratio,
        'nose_y': nose[1],
        'shoulder_y': mid_shoulder_y,
        'forward_lean': forward_lean,
        'shoulder_tilt': shoulder_tilt,
    }