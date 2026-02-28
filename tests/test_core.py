import math
import time
import types
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from core.geometry import calculate_posture_metric
from core.state import WardenState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_keypoints(
    nose=(320, 100),
    left_shoulder=(270, 200),
    right_shoulder=(370, 200),
    left_ear=None,
    right_ear=None,
):
    kp = {
        'nose': nose,
        'left_shoulder': left_shoulder,
        'right_shoulder': right_shoulder,
    }
    if left_ear is not None:
        kp['left_ear'] = left_ear
    if right_ear is not None:
        kp['right_ear'] = right_ear
    return kp


# ---------------------------------------------------------------------------
# geometry.calculate_posture_metric
# ---------------------------------------------------------------------------

class TestCalculatePostureMetric:
    def test_basic_ratio(self):
        kp = make_keypoints(nose=(320, 100), left_shoulder=(270, 200), right_shoulder=(370, 200))
        m = calculate_posture_metric(kp)
        shoulder_width = math.dist((270, 200), (370, 200))  # 100
        expected_ratio = (200 - 100) / shoulder_width       # 1.0
        assert pytest.approx(m['ratio'], rel=1e-5) == expected_ratio

    def test_nose_y_and_shoulder_y(self):
        kp = make_keypoints(nose=(320, 80), left_shoulder=(270, 200), right_shoulder=(370, 200))
        m = calculate_posture_metric(kp)
        assert m['nose_y'] == 80
        assert m['shoulder_y'] == 200

    def test_zero_shoulder_width_returns_safe_defaults(self):
        kp = make_keypoints(left_shoulder=(300, 200), right_shoulder=(300, 200))
        m = calculate_posture_metric(kp)
        assert m['ratio'] == 0
        assert m['forward_lean'] is None
        assert m['shoulder_tilt'] == 0

    def test_forward_lean_with_ears(self):
        # Ears high above shoulders → large positive gap → good posture
        kp = make_keypoints(
            nose=(320, 100),
            left_shoulder=(270, 200),
            right_shoulder=(370, 200),
            left_ear=(290, 130),
            right_ear=(350, 130),
        )
        m = calculate_posture_metric(kp)
        assert m['forward_lean'] is not None
        # avg_ear_y = 130, mid_shoulder_y = 200, gap = 70, width = 100
        assert pytest.approx(m['forward_lean'], rel=1e-5) == 70 / 100

    def test_forward_lean_none_without_ears(self):
        kp = make_keypoints()
        m = calculate_posture_metric(kp)
        assert m['forward_lean'] is None

    def test_forward_lean_single_ear(self):
        kp = make_keypoints(
            left_shoulder=(270, 200),
            right_shoulder=(370, 200),
            right_ear=(350, 150),
        )
        m = calculate_posture_metric(kp)
        # Only right ear: avg_ear_y = 150, gap = 50, width = 100
        assert pytest.approx(m['forward_lean'], rel=1e-5) == 50 / 100

    def test_shoulder_tilt_level(self):
        kp = make_keypoints(left_shoulder=(270, 200), right_shoulder=(370, 200))
        m = calculate_posture_metric(kp)
        assert m['shoulder_tilt'] == 0

    def test_shoulder_tilt_asymmetric(self):
        # left shoulder 20 px lower than right
        kp = make_keypoints(left_shoulder=(270, 220), right_shoulder=(370, 200))
        m = calculate_posture_metric(kp)
        width = math.dist((270, 220), (370, 200))
        expected_tilt = 20 / width
        assert pytest.approx(m['shoulder_tilt'], rel=1e-4) == expected_tilt

    def test_slouch_gives_lower_ratio(self):
        # head lower than shoulders → negative ratio
        good_kp = make_keypoints(nose=(320, 80), left_shoulder=(270, 200), right_shoulder=(370, 200))
        bad_kp  = make_keypoints(nose=(320, 180), left_shoulder=(270, 200), right_shoulder=(370, 200))
        good = calculate_posture_metric(good_kp)
        bad  = calculate_posture_metric(bad_kp)
        assert good['ratio'] > bad['ratio']


# ---------------------------------------------------------------------------
# state.WardenState
# ---------------------------------------------------------------------------

class TestWardenState:
    def test_initial_hp(self):
        ws = WardenState(max_hp=100)
        assert ws.current_hp == 100

    def test_not_calibrated_update_is_noop(self):
        ws = WardenState()
        ws.update({'ratio': 0.5, 'nose_y': 100, 'shoulder_y': 200, 'shoulder_tilt': 0})
        assert ws.status == "IDLE"

    def test_calibrate_sets_baseline(self):
        ws = WardenState()
        posture = {'ratio': 1.0, 'nose_y': 80, 'shoulder_y': 200, 'forward_lean': 0.7}
        ws.calibrate(posture)
        assert ws.is_calibrated
        assert ws.baseline_ratio == 1.0
        assert ws.baseline_nose_y == 80
        assert ws.baseline_forward_lean == 0.7

    def test_take_damage(self):
        ws = WardenState(max_hp=100)
        ws.take_damage(10)
        assert ws.current_hp == 90

    def test_take_damage_floor_at_zero(self):
        ws = WardenState(max_hp=100)
        ws.take_damage(200)
        assert ws.current_hp == 0

    def test_heal(self):
        ws = WardenState(max_hp=100)
        ws.current_hp = 50
        ws.heal(20)
        assert ws.current_hp == 70

    def test_heal_capped_at_max(self):
        ws = WardenState(max_hp=100)
        ws.current_hp = 95
        ws.heal(20)
        assert ws.current_hp == 100

    def _calibrated_state(self, ratio=1.0, nose_y=80, shoulder_y=200):
        ws = WardenState(max_hp=100)
        ws.calibrate({'ratio': ratio, 'nose_y': nose_y, 'shoulder_y': shoulder_y, 'forward_lean': 0.7})
        return ws

    def test_good_posture_status(self):
        ws = self._calibrated_state()
        ws.update({'ratio': 1.0, 'nose_y': 80, 'shoulder_y': 200, 'shoulder_tilt': 0.0})
        assert ws.status == "GOOD POSTURE FINALLY"

    def test_good_posture_heals(self):
        ws = self._calibrated_state()
        ws.current_hp = 50
        ws.update({'ratio': 1.0, 'nose_y': 80, 'shoulder_y': 200, 'shoulder_tilt': 0.0})
        assert ws.current_hp > 50

    def test_slouch_status(self):
        ws = self._calibrated_state(ratio=1.0, nose_y=80)
        # ratio drops well below baseline and nose drops > tolerance
        ws.update({'ratio': 0.3, 'nose_y': 130, 'shoulder_y': 200, 'shoulder_tilt': 0.0})
        assert ws.status == "BECOMING A GOBLIN?"

    def test_slouch_takes_damage(self):
        ws = self._calibrated_state(ratio=1.0, nose_y=80)
        initial_hp = ws.current_hp
        ws.update({'ratio': 0.3, 'nose_y': 130, 'shoulder_y': 200, 'shoulder_tilt': 0.0})
        assert ws.current_hp < initial_hp

    def test_shoulder_tilt_status(self):
        ws = self._calibrated_state()
        ws.update({'ratio': 1.0, 'nose_y': 80, 'shoulder_y': 200, 'shoulder_tilt': 0.5})
        assert ws.status == "LEANING SIDEWAYS!"

    def test_shoulder_tilt_takes_half_damage(self):
        ws = self._calibrated_state()
        ws.current_hp = 100
        ws.update({'ratio': 1.0, 'nose_y': 80, 'shoulder_y': 200, 'shoulder_tilt': 0.5})
        assert ws.current_hp == 99.5

    def test_good_posture_increments_good_frames(self):
        ws = self._calibrated_state()
        ws.update({'ratio': 1.0, 'nose_y': 80, 'shoulder_y': 200, 'shoulder_tilt': 0.0})
        assert ws.total_good_frames == 1

    def test_bad_posture_increments_bad_frames(self):
        ws = self._calibrated_state(ratio=1.0, nose_y=80)
        ws.update({'ratio': 0.3, 'nose_y': 130, 'shoulder_y': 200, 'shoulder_tilt': 0.0})
        assert ws.total_bad_frames == 1

    def test_streak_resets_on_bad_posture(self):
        ws = self._calibrated_state()
        ws.update({'ratio': 1.0, 'nose_y': 80, 'shoulder_y': 200, 'shoulder_tilt': 0.0})
        assert ws.streak_start is not None
        ws.update({'ratio': 0.3, 'nose_y': 130, 'shoulder_y': 200, 'shoulder_tilt': 0.0})
        assert ws.streak_start is None
        assert ws.current_streak == 0

    def test_get_session_summary_format(self):
        ws = WardenState()
        summary = ws.get_session_summary()
        assert "SESSION SUMMARY" in summary
        assert "Duration:" in summary
        assert "Good Posture:" in summary
        assert "Best Streak:" in summary
        assert "Game Overs:" in summary

    def test_session_summary_good_pct_zero_frames(self):
        ws = WardenState()
        summary = ws.get_session_summary()
        assert "0.0%" in summary


# ---------------------------------------------------------------------------
# detector.PoseDetector (mocked YOLO)
# ---------------------------------------------------------------------------

class TestPoseDetector:
    def _make_mock_yolo(self, kp_data, conf_values):
        """
        kp_data:    list of (x, y) per keypoint index
        conf_values: list of confidence floats per keypoint index
        Returns a mock that mimics results[0].keypoints.data[0].cpu().numpy()
        """
        raw = np.array([[x, y, c] for (x, y), c in zip(kp_data, conf_values)], dtype=np.float32)

        mock_keypoints = MagicMock()
        mock_keypoints.data = [MagicMock()]
        mock_keypoints.data[0].cpu.return_value.numpy.return_value = raw
        # len() check: results[0].keypoints needs to have length > 0
        mock_keypoints.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.keypoints = mock_keypoints

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        return mock_model

    def _default_kp_data(self):
        """17 keypoints; we only care about indices 0,3,4,5,6."""
        kp = [(0, 0)] * 17
        kp[0]  = (320, 100)   # nose
        kp[3]  = (290, 130)   # left ear
        kp[4]  = (350, 130)   # right ear
        kp[5]  = (270, 200)   # left shoulder
        kp[6]  = (370, 200)   # right shoulder
        return kp

    def _default_conf(self, nose=0.9, left_ear=0.9, right_ear=0.9, left_sh=0.9, right_sh=0.9):
        conf = [0.0] * 17
        conf[0] = nose
        conf[3] = left_ear
        conf[4] = right_ear
        conf[5] = left_sh
        conf[6] = right_sh
        return conf

    def _make_detector(self, mock_model):
        with patch('core.detector.YOLO', return_value=mock_model):
            from core.detector import PoseDetector
            return PoseDetector('fake.pt')

    def test_returns_keypoints_when_confident(self):
        mock_model = self._make_mock_yolo(self._default_kp_data(), self._default_conf())
        det = self._make_detector(mock_model)
        result = det.get_keypoints(MagicMock())
        assert result is not None
        assert 'nose' in result
        assert 'left_shoulder' in result
        assert 'right_shoulder' in result

    def test_includes_ears_when_confident(self):
        mock_model = self._make_mock_yolo(self._default_kp_data(), self._default_conf())
        det = self._make_detector(mock_model)
        result = det.get_keypoints(MagicMock())
        assert 'left_ear' in result
        assert 'right_ear' in result

    def test_excludes_ears_when_low_confidence(self):
        conf = self._default_conf(left_ear=0.1, right_ear=0.1)
        mock_model = self._make_mock_yolo(self._default_kp_data(), conf)
        det = self._make_detector(mock_model)
        result = det.get_keypoints(MagicMock())
        assert result is not None
        assert 'left_ear' not in result
        assert 'right_ear' not in result

    def test_returns_none_when_nose_low_confidence(self):
        conf = self._default_conf(nose=0.1)
        mock_model = self._make_mock_yolo(self._default_kp_data(), conf)
        det = self._make_detector(mock_model)
        result = det.get_keypoints(MagicMock())
        assert result is None

    def test_returns_none_when_shoulder_low_confidence(self):
        conf = self._default_conf(left_sh=0.1)
        mock_model = self._make_mock_yolo(self._default_kp_data(), conf)
        det = self._make_detector(mock_model)
        result = det.get_keypoints(MagicMock())
        assert result is None

    def test_returns_none_when_no_results(self):
        mock_model = MagicMock()
        mock_model.return_value = []
        det = self._make_detector(mock_model)
        result = det.get_keypoints(MagicMock())
        assert result is None

    def test_nose_coordinates_correct(self):
        mock_model = self._make_mock_yolo(self._default_kp_data(), self._default_conf())
        det = self._make_detector(mock_model)
        result = det.get_keypoints(MagicMock())
        assert result['nose'][0] == pytest.approx(320)
        assert result['nose'][1] == pytest.approx(100)
