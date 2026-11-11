"""Turns hand landmarks into mouse and keyboard actions.

Nothing here touches the camera, the screen or the GUI: the controller gets
landmarks in and calls a small "backend" object out. That keeps the logic
testable without a webcam.
"""

import math
import time
from dataclasses import dataclass

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
MIDDLE_MCP = 9

PEACE = 2  # ASL digit 2: index and middle finger up
POINT = 1  # ASL digit 1: index finger up


class CursorMapper:
    """Maps a hand position in the camera frame to a smoothed screen position.

    Only the central part of the frame is used, so the screen edges can be
    reached without the hand leaving the camera's view.
    """

    def __init__(self, screen_width, screen_height, margin=0.2, edge_padding=2):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.margin = margin
        # Staying a couple of pixels off the edges keeps the cursor out of the
        # screen corners, which PyAutoGUI treats as an emergency stop.
        self.edge_padding = edge_padding
        self._position = None

    def reset(self, start=None):
        """Forget the smoothing history; optionally continue from `start`."""
        self._position = start

    def _to_screen(self, x, y):
        span = 1 - 2 * self.margin
        u = min(max((x - self.margin) / span, 0.0), 1.0)
        v = min(max((y - self.margin) / span, 0.0), 1.0)
        pad = self.edge_padding
        return (pad + u * (self.screen_width - 1 - 2 * pad),
                pad + v * (self.screen_height - 1 - 2 * pad))

    def update(self, x, y, smoothing):
        """x, y: hand position in the frame, 0..1. smoothing: 0 (none) .. 10 (max).

        Returns the new cursor position in pixels as integers.
        """
        target = self._to_screen(x, y)
        if self._position is None:
            self._position = target
        else:
            # Exponential smoothing: move a fraction of the way to the target.
            alpha = 1.0 / (1.0 + max(smoothing, 0))
            px, py = self._position
            self._position = (px + alpha * (target[0] - px), py + alpha * (target[1] - py))
        return round(self._position[0]), round(self._position[1])


class PinchDetector:
    """Detects thumb and index fingertip touching, independent of camera distance.

    The fingertip distance is divided by the palm length. Two thresholds are
    used so the pinch does not flicker on and off near the limit. In the
    dataset, touching fingertips give a ratio below 0.28 and an open hand a
    ratio above 0.58.
    """

    def __init__(self, press_below=0.30, release_above=0.45):
        self.press_below = press_below
        self.release_above = release_above
        self.pinched = False

    @staticmethod
    def ratio(points):
        """points: 21 (x, y) pairs in pixels."""
        palm = math.dist(points[WRIST], points[MIDDLE_MCP])
        if palm < 1e-9:
            return math.inf
        return math.dist(points[THUMB_TIP], points[INDEX_TIP]) / palm

    def update(self, points):
        ratio = self.ratio(points)
        if self.pinched:
            self.pinched = ratio < self.release_above
        else:
            self.pinched = ratio < self.press_below
        return self.pinched


class HoldTrigger:
    """Fires once when a gesture has been held for `hold_seconds`.

    Recognition flickers on real camera frames, so gaps shorter than
    `grace_seconds` do not interrupt the hold. After firing, the gesture has
    to be released before it can fire again.
    """

    def __init__(self, hold_seconds, grace_seconds=0.25):
        self.hold_seconds = hold_seconds
        self.grace_seconds = grace_seconds
        self.reset()

    def reset(self):
        self._since = None
        self._last_seen = None
        self._fired = False

    def update(self, active, now):
        if not active:
            if self._last_seen is None or now - self._last_seen > self.grace_seconds:
                self.reset()
            return False
        self._last_seen = now
        if self._since is None:
            self._since = now
        if not self._fired and now - self._since >= self.hold_seconds:
            self._fired = True
            return True
        return False


@dataclass
class Settings:
    """What the user can change from the control panel."""
    smoothing: int = 4
    pinch_enabled: bool = True
    peace_enabled: bool = True
    point_enabled: bool = False


class HandMouseController:
    """Decides what to do for each camera frame.

    backend must provide: position(), move_to(x, y), mouse_down(), mouse_up(),
    show_desktop() and press_escape().
    """

    def __init__(self, classifier, backend, screen_size, settings=None, clock=time.monotonic):
        self.classifier = classifier
        self.backend = backend
        self.settings = settings or Settings()
        self.clock = clock
        self.cursor = CursorMapper(*screen_size)
        self.pinch = PinchDetector()
        self.peace = HoldTrigger(0.5)
        self.point = HoldTrigger(0.6)
        self.click_freeze_seconds = 0.25
        self._pinch_started = 0.0
        self.button_down = False
        self.gesture = None
        self._hand_visible = False

    def update(self, landmarks, frame_width, frame_height):
        """landmarks: 21 (x, y, ...) points with x, y in 0..1, or None if no hand."""
        if landmarks is None:
            self.hand_lost()
            return

        if not self._hand_visible:
            # Hand just appeared: glide from where the cursor is now instead of jumping.
            self.cursor.reset(self.backend.position())
            self._hand_visible = True

        now = self.clock()

        # Pinch = hold the left button, so a short pinch clicks and a long one drags.
        pixels = [(p[0] * frame_width, p[1] * frame_height) for p in landmarks]
        pinched = self.pinch.update(pixels) and self.settings.pinch_enabled
        if pinched and not self.button_down:
            self.backend.mouse_down()
            self.button_down = True
            self._pinch_started = now
        elif not pinched and self.button_down:
            self.backend.mouse_up()
            self.button_down = False

        # The cursor follows the middle knuckle rather than a fingertip, so it
        # stays put while the fingers pinch or make a gesture. It is also held
        # still for the first moment of a pinch, so a quick pinch is a clean
        # click and not a tiny drag.
        frozen = self.button_down and now - self._pinch_started < self.click_freeze_seconds
        if not frozen:
            anchor = landmarks[MIDDLE_MCP]
            self.backend.move_to(*self.cursor.update(anchor[0], anchor[1], self.settings.smoothing))

        # The classifier is only needed while a classifier gesture is switched on.
        if self.settings.peace_enabled or self.settings.point_enabled:
            self.gesture = self.classifier.predict_one(landmarks)
        else:
            self.gesture = None

        if self.peace.update(self.gesture == PEACE and self.settings.peace_enabled, now):
            self.backend.show_desktop()
        if self.point.update(self.gesture == POINT and self.settings.point_enabled, now):
            self.backend.press_escape()

    def hand_lost(self):
        """Call when no hand is visible: let go of everything."""
        if self.button_down:
            self.backend.mouse_up()
            self.button_down = False
        self.pinch.pinched = False
        self.peace.reset()
        self.point.reset()
        self.gesture = None
        self._hand_visible = False
