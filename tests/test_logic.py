"""Tests for the parts that do not need a webcam. Run with:  pytest"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_model import build_classifier, landmarks_to_features, load_dataset, mirror
from mouse_controller import (CursorMapper, HandMouseController, HoldTrigger,
                              PinchDetector, Settings)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def classifier():
    return build_classifier()


def sample(dataset, digit):
    landmarks, labels = dataset
    return landmarks[labels == digit][0]


class FakeBackend:
    """Records what the controller asks for instead of moving the real mouse."""

    def __init__(self):
        self.events = []
        self.cursor = (500, 500)

    def position(self):
        return self.cursor

    def move_to(self, x, y):
        self.cursor = (x, y)

    def mouse_down(self):
        self.events.append("down")

    def mouse_up(self):
        self.events.append("up")

    def show_desktop(self):
        self.events.append("show_desktop")

    def press_escape(self):
        self.events.append("escape")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_controller(classifier, **settings):
    backend, clock = FakeBackend(), FakeClock()
    controller = HandMouseController(classifier, backend, (1920, 1080),
                                     Settings(**settings), clock)
    return controller, backend, clock


def hold(controller, clock, hand, seconds, fps=30):
    for _ in range(int(seconds * fps)):
        clock.now += 1 / fps
        controller.update(hand, 640, 480)


# --- features and classifier -------------------------------------------------

def test_features_ignore_hand_position_and_size(dataset):
    hand = sample(dataset, 2)
    moved = hand * 0.4 + np.array([0.3, -0.2, 0.0])
    assert np.allclose(landmarks_to_features(hand), landmarks_to_features(moved))


def test_features_ignore_hand_tilt(dataset):
    hand = sample(dataset, 2)
    angle = np.radians(35)
    tilted = hand.copy()
    tilted[:, 0] = hand[:, 0] * np.cos(angle) - hand[:, 1] * np.sin(angle)
    tilted[:, 1] = hand[:, 0] * np.sin(angle) + hand[:, 1] * np.cos(angle)
    assert np.allclose(landmarks_to_features(hand), landmarks_to_features(tilted))


def test_classifier_accepts_a_slightly_different_hand(dataset, classifier):
    landmarks, labels = dataset
    rng = np.random.default_rng(0)
    for digit in (1, 2):
        hands = landmarks[labels == digit][:50].copy()
        palm = np.linalg.norm(hands[:, 9, :2] - hands[:, 0, :2], axis=1)[:, None, None]
        hands[:, :, :2] += rng.normal(0, 0.05, size=(50, 21, 2)) * palm
        assert np.mean(classifier.predict(hands) == digit) > 0.95


def test_classifier_recognises_each_digit(dataset, classifier):
    for digit in range(10):
        assert classifier.predict_one(sample(dataset, digit)) == digit


def test_classifier_works_for_the_other_hand(dataset, classifier):
    assert classifier.predict_one(mirror(sample(dataset, 2))) == 2


def test_classifier_rejects_a_pose_it_never_saw(classifier):
    scrambled = np.random.default_rng(0).uniform(0, 1, size=(21, 3))
    assert classifier.predict_one(scrambled) is None


# --- cursor ------------------------------------------------------------------

def test_cursor_reaches_every_edge_but_never_a_corner_pixel():
    mapper = CursorMapper(1920, 1080)
    assert mapper.update(0.0, 0.0, smoothing=0) == (2, 2)
    mapper.reset()
    assert mapper.update(1.0, 1.0, smoothing=0) == (1917, 1077)


def test_cursor_smoothing_is_symmetric_and_settles_on_the_target():
    def step_from_centre(x):
        mapper = CursorMapper(1001, 1001, edge_padding=0)
        centre = mapper.update(0.5, 0.5, smoothing=4)
        return mapper.update(x, 0.5, smoothing=4)[0] - centre[0], mapper

    right, _ = step_from_centre(0.6)
    left, mapper = step_from_centre(0.4)
    assert right == -left > 0

    for _ in range(200):
        position = mapper.update(0.4, 0.5, smoothing=4)
    assert position == tuple(round(v) for v in mapper._to_screen(0.4, 0.5))


# --- pinch -------------------------------------------------------------------

def test_pinch_matches_the_dataset(dataset):
    def pixels(hand):
        return [(x * 640, y * 480) for x, y, _ in hand]

    assert PinchDetector().update(pixels(sample(dataset, 9)))      # thumb touches index
    assert not PinchDetector().update(pixels(sample(dataset, 5)))  # open hand


def test_pinch_does_not_flicker_between_thresholds():
    def hand_with_ratio(ratio):
        points = [(0.0, 0.0)] * 21
        points[9] = (0.0, 100.0)            # palm length 100
        points[4] = (0.0, 0.0)
        points[8] = (ratio * 100.0, 0.0)
        return points

    detector = PinchDetector()
    assert not detector.update(hand_with_ratio(0.40))  # not close enough to start
    assert detector.update(hand_with_ratio(0.20))      # pinched
    assert detector.update(hand_with_ratio(0.40))      # still held
    assert not detector.update(hand_with_ratio(0.50))  # released


# --- hold trigger ------------------------------------------------------------

def test_hold_trigger_fires_once_per_hold():
    trigger = HoldTrigger(0.5)
    assert not trigger.update(True, 0.0)
    assert not trigger.update(True, 0.4)
    assert trigger.update(True, 0.5)
    assert not trigger.update(True, 5.0)   # still held: no repeat
    assert not trigger.update(False, 5.1)  # released...
    assert not trigger.update(False, 5.5)  # ...for longer than the grace period
    assert not trigger.update(True, 5.6)
    assert trigger.update(True, 6.1)


def test_hold_trigger_survives_a_short_flicker():
    trigger = HoldTrigger(0.5)
    trigger.update(True, 0.0)
    trigger.update(False, 0.2)          # one or two unrecognised frames
    assert trigger.update(True, 0.5)


def test_hold_trigger_starts_again_after_a_real_break():
    trigger = HoldTrigger(0.5)
    trigger.update(True, 0.0)
    trigger.update(False, 0.1)
    trigger.update(False, 0.5)          # gone for longer than the grace period
    assert not trigger.update(True, 0.6)
    assert trigger.update(True, 1.1)


def test_flicker_does_not_fire_a_held_gesture_twice():
    trigger = HoldTrigger(0.5)
    trigger.update(True, 0.0)
    assert trigger.update(True, 0.5)
    trigger.update(False, 0.6)
    assert not trigger.update(True, 0.7)
    assert not trigger.update(True, 2.0)


# --- controller --------------------------------------------------------------

def test_holding_peace_shows_the_desktop_exactly_once(dataset, classifier):
    controller, backend, clock = make_controller(classifier)
    hold(controller, clock, sample(dataset, 2), seconds=3)
    assert backend.events == ["show_desktop"]


def test_point_does_nothing_unless_enabled(dataset, classifier):
    controller, backend, clock = make_controller(classifier)
    hold(controller, clock, sample(dataset, 1), seconds=2)
    assert backend.events == []

    controller, backend, clock = make_controller(classifier, point_enabled=True)
    hold(controller, clock, sample(dataset, 1), seconds=2)
    assert backend.events == ["escape"]


def test_pinch_presses_and_releases_the_button(dataset, classifier):
    controller, backend, clock = make_controller(classifier)
    hold(controller, clock, sample(dataset, 9), seconds=0.2)
    hold(controller, clock, sample(dataset, 5), seconds=0.2)
    assert backend.events == ["down", "up"]


def test_cursor_stays_still_during_a_quick_pinch(dataset, classifier):
    controller, backend, clock = make_controller(classifier)
    hold(controller, clock, sample(dataset, 5), seconds=1)
    hold(controller, clock, sample(dataset, 9), seconds=1 / 30)
    at_press = backend.cursor
    hold(controller, clock, sample(dataset, 9) + np.array([0.2, 0.1, 0.0]), seconds=0.15)
    assert backend.cursor == at_press


def test_button_is_released_when_the_hand_disappears(dataset, classifier):
    controller, backend, clock = make_controller(classifier)
    hold(controller, clock, sample(dataset, 9), seconds=0.2)
    controller.update(None, 640, 480)
    assert backend.events == ["down", "up"]


def test_disabled_pinch_never_clicks(dataset, classifier):
    controller, backend, clock = make_controller(classifier, pinch_enabled=False)
    hold(controller, clock, sample(dataset, 9), seconds=1)
    assert backend.events == []


def test_cursor_glides_from_where_the_mouse_already_is(dataset, classifier):
    controller, backend, clock = make_controller(classifier, smoothing=10)
    backend.cursor = (1500, 900)
    hold(controller, clock, sample(dataset, 5), seconds=1 / 30)
    assert abs(backend.cursor[0] - 1500) < 200 and abs(backend.cursor[1] - 900) < 200


def test_point_fires_even_if_some_frames_are_not_recognised(dataset, classifier):
    controller, backend, clock = make_controller(classifier, point_enabled=True)
    point, other = sample(dataset, 1), sample(dataset, 5)
    for frame in range(60):
        clock.now += 1 / 30
        controller.update(other if frame % 5 == 4 else point, 640, 480)
    assert backend.events == ["escape"]
