"""Control the mouse with your hand in front of a webcam.

Run:  python hand_mouse.py
Quit: press ESC in the camera window, or close either window.
"""

import argparse
import time
import tkinter as tk
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks.python import vision

from gesture_model import build_classifier
from mouse_controller import HandMouseController, Settings

HAND_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

WINDOW_NAME = "Hand Mouse"
PANEL_COLOUR = "#76B3D3"

# Pairs of landmark indices that are joined to draw the hand skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # little finger
    (0, 17),                                 # palm edge
]


class HandTracker:
    """Finds one hand in a camera frame using MediaPipe's hand landmarker."""

    def __init__(self):
        if not HAND_MODEL_PATH.exists():
            print("Downloading the MediaPipe hand model (about 8 MB, first run only)...")
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            # Passing the file contents avoids problems with unusual characters in Windows paths.
            base_options=mp.tasks.BaseOptions(model_asset_buffer=HAND_MODEL_PATH.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,  # a second hand would fight over the cursor
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp = 0

    def find_hand(self, frame_bgr):
        """Return the hand as an array of 21 (x, y, z) points, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # MediaPipe needs a timestamp that always increases.
        timestamp = max(int(time.monotonic() * 1000), self._last_timestamp + 1)
        self._last_timestamp = timestamp
        result = self._landmarker.detect_for_video(image, timestamp)
        if not result.hand_landmarks:
            return None
        return np.array([(point.x, point.y, point.z) for point in result.hand_landmarks[0]])

    def close(self):
        self._landmarker.close()


class PyAutoGuiBackend:
    """Carries out the controller's decisions on the real mouse and keyboard."""

    def __init__(self):
        # PyAutoGUI waits 0.1 s after every call by default, which would limit
        # the whole program to about 10 frames per second.
        pyautogui.PAUSE = 0

    def position(self):
        return tuple(pyautogui.position())

    def move_to(self, x, y):
        pyautogui.moveTo(x, y)

    def mouse_down(self):
        pyautogui.mouseDown()

    def mouse_up(self):
        pyautogui.mouseUp()

    def show_desktop(self):
        pyautogui.hotkey("win", "d")

    def press_escape(self):
        pyautogui.press("esc")


def draw_hand(frame, hand):
    height, width = frame.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in hand]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


class App:
    """The control panel plus the camera loop, both on the main thread."""

    def __init__(self, camera_index):
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise SystemExit(f"Could not open camera {camera_index}. "
                             "Is it connected and not in use by another program?")

        self.tracker = HandTracker()
        self.settings = Settings()
        self.backend = PyAutoGuiBackend()
        self.controller = HandMouseController(
            build_classifier(), self.backend, pyautogui.size(), self.settings)

        self.last_frame_time = time.monotonic()
        self.fps = 0.0
        self.running = True
        self._build_panel()
        cv2.namedWindow(WINDOW_NAME)

    def _build_panel(self):
        self.root = tk.Tk()
        self.root.title("Hand Mouse Controls")
        self.root.geometry("300x260")
        self.root.configure(bg=PANEL_COLOUR)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        self.smoothing = tk.IntVar(value=self.settings.smoothing)
        self.pinch = tk.BooleanVar(value=self.settings.pinch_enabled)
        self.peace = tk.BooleanVar(value=self.settings.peace_enabled)
        self.point = tk.BooleanVar(value=self.settings.point_enabled)

        tk.Label(self.root, text="Hand Gesture Mouse", font=("Helvetica", 14, "bold"),
                 bg=PANEL_COLOUR).pack(pady=10)
        tk.Label(self.root, text="Cursor smoothing", font=("Helvetica", 10),
                 bg=PANEL_COLOUR).pack()
        tk.Scale(self.root, from_=0, to=10, orient=tk.HORIZONTAL, variable=self.smoothing,
                 length=200, tickinterval=5).pack(pady=5)
        for text, variable in (("\U0001F90F Pinch = Left click / drag", self.pinch),
                               ("✌ Peace = Show desktop", self.peace),
                               ("☝ Point = ESC", self.point)):
            tk.Checkbutton(self.root, text=text, font=("Helvetica", 10), variable=variable,
                           bg=PANEL_COLOUR, activebackground=PANEL_COLOUR,
                           anchor="w", padx=10).pack(fill="x", pady=2)

    def _read_panel(self):
        self.settings.smoothing = self.smoothing.get()
        self.settings.pinch_enabled = self.pinch.get()
        self.settings.peace_enabled = self.peace.get()
        self.settings.point_enabled = self.point.get()

    def step(self):
        """Handle one camera frame, then ask Tk to call this again."""
        if not self.running:
            return
        ok, frame = self.camera.read()
        if not ok:
            print("The camera stopped sending frames.")
            self.stop()
            return

        frame = cv2.flip(frame, 1)  # mirror, so moving right moves the cursor right
        height, width = frame.shape[:2]
        hand = self.tracker.find_hand(frame)
        self._read_panel()

        try:
            self.controller.update(hand, width, height)
        except pyautogui.FailSafeException:
            print("Mouse moved into a screen corner: stopping (PyAutoGUI fail-safe).")
            self.stop()
            return

        self._show(frame, hand)
        key = cv2.waitKey(1) & 0xFF
        window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        if key == 27 or window_closed:
            self.stop()
            return
        self.root.after(1, self.step)

    def _show(self, frame, hand):
        now = time.monotonic()
        instant_fps = 1.0 / max(now - self.last_frame_time, 1e-6)
        self.fps = instant_fps if self.fps == 0 else 0.9 * self.fps + 0.1 * instant_fps
        self.last_frame_time = now

        if hand is not None:
            draw_hand(frame, hand)
        gesture = self.controller.gesture
        lines = [f"Gesture: {'-' if gesture is None else gesture}",
                 f"Pinch: {'yes' if self.controller.button_down else 'no'}",
                 f"FPS: {self.fps:.0f}"]
        for row, text in enumerate(lines):
            cv2.putText(frame, text, (10, 30 + 30 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow(WINDOW_NAME, frame)

    def stop(self):
        if not self.running:
            return
        self.running = False
        try:
            self.controller.hand_lost()  # never leave the mouse button held down
        except pyautogui.FailSafeException:
            pass
        self.tracker.close()
        self.camera.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self):
        self.root.after(0, self.step)
        try:
            self.root.mainloop()
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="Control the mouse with hand gestures.")
    parser.add_argument("--camera", type=int, default=0,
                        help="index of the webcam to use (default: 0)")
    App(parser.parse_args().camera).run()


if __name__ == "__main__":
    main()
