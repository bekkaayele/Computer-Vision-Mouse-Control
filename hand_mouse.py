"""Control the mouse with your hand in front of a webcam.

Run:  python hand_mouse.py
Quit: press Esc, click Quit, or close the window.
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

# Control panel colours
BACKGROUND = "#14171c"
CARD = "#1f242c"
CARD_HOVER = "#2a313b"
TEXT = "#e6e9ee"
MUTED = "#8b94a3"
ACCENT = "#3b82f6"
GREEN = "#22c55e"
ON_COLOUR = "#1d4d33"
ON_HOVER = "#256341"
FONT = "Segoe UI"

# Drawing the camera picture is the slowest part of the window, so it is shown
# a little smaller and refreshed on every second frame. The mouse is still
# updated on every frame.
VIDEO_WIDTH = 480
VIDEO_EVERY = 2

# Smoothing presets: label -> how strongly the cursor is smoothed (0 = not at all).
SMOOTHING_LEVELS = [("Off", 0), ("Low", 2), ("Medium", 4), ("High", 8)]

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
            print("Downloading the MediaPipe hand model. It's about 8 MB, and for the first run only :)")
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


class Button(tk.Label):
    """A large flat button. Everything in the panel is a single click on a big
    target, because dragging and small targets are hard with a hand-driven cursor."""

    def __init__(self, parent, text, command, colour=CARD, hover=CARD_HOVER, **options):
        super().__init__(parent, text=text, bg=colour, fg=TEXT, font=(FONT, 11, "bold"),
                         pady=12, cursor="hand2", **options)
        self.colour, self.hover = colour, hover
        self.bind("<Button-1>", lambda event: command())
        self.bind("<Enter>", lambda event: self.configure(bg=self.hover))
        self.bind("<Leave>", lambda event: self.configure(bg=self.colour))

    def set_colour(self, colour, hover):
        self.colour, self.hover = colour, hover
        self.configure(bg=colour)


class Toggle(tk.Frame):
    """A wide on/off row: gesture name, what it does, and its state."""

    def __init__(self, parent, title, action, command):
        super().__init__(parent, bg=CARD, cursor="hand2")
        self.title = tk.Label(self, text=title, font=(FONT, 11, "bold"), anchor="w")
        self.action = tk.Label(self, text=action, font=(FONT, 9), anchor="w")
        self.state = tk.Label(self, font=(FONT, 11, "bold"), width=4)
        self.state.pack(side="right", padx=(0, 12))
        self.title.pack(fill="x", padx=14, pady=(8, 0))
        self.action.pack(fill="x", padx=14, pady=(0, 8))
        self.colour, self.hover = CARD, CARD_HOVER
        for widget in (self, self.title, self.action, self.state):
            widget.bind("<Button-1>", lambda event: command())
        self.bind("<Enter>", lambda event: self._paint(self.hover))
        self.bind("<Leave>", lambda event: self._paint(self.colour))

    def _paint(self, colour):
        for widget in (self, self.title, self.action, self.state):
            widget.configure(bg=colour)

    def set(self, enabled):
        self.colour, self.hover = (ON_COLOUR, ON_HOVER) if enabled else (CARD, CARD_HOVER)
        self.state.configure(text="ON" if enabled else "OFF", fg=GREEN if enabled else MUTED)
        self.title.configure(fg=TEXT if enabled else MUTED)
        self.action.configure(fg=MUTED)
        self._paint(self.colour)


class App:
    """One window with the camera view and the controls, all on the main thread."""

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
        self.frame_count = 0
        self.paused = False
        self.running = True
        self._build_window()

    # --- window layout ---------------------------------------------------------

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Hand Mouse")
        self.root.configure(bg=BACKGROUND)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.bind("<Escape>", lambda event: self.stop())

        self.video = tk.Label(self.root, bg="black", bd=0)
        self.video.pack(side="left", anchor="n", padx=(16, 0), pady=16)
        self.video_image = None

        panel = tk.Frame(self.root, bg=BACKGROUND)
        panel.pack(side="left", fill="y", padx=16, pady=16)

        tk.Label(panel, text="Hand Mouse", bg=BACKGROUND, fg=TEXT,
                 font=(FONT, 18, "bold"), anchor="w").pack(fill="x")
        self.status = tk.Label(panel, bg=BACKGROUND, fg=MUTED, font=(FONT, 10),
                               anchor="w", justify="left")
        self.status.pack(fill="x", pady=(2, 0))

        self._heading(panel, "Cursor smoothing")
        row = tk.Frame(panel, bg=BACKGROUND)
        row.pack(fill="x")
        self.smoothing_buttons = {}
        for column, (label, value) in enumerate(SMOOTHING_LEVELS):
            button = Button(row, label, lambda value=value: self._set_smoothing(value), width=7)
            button.grid(row=0, column=column, padx=(0 if column == 0 else 6, 0), sticky="ew")
            row.columnconfigure(column, weight=1)
            self.smoothing_buttons[value] = button
        self._set_smoothing(self.settings.smoothing)

        self._heading(panel, "Gestures")
        self.gesture_toggles = {}
        for name, title, action in (("pinch_enabled", "Pinch", "Click, or hold to drag"),
                                    ("peace_enabled", "Peace sign", "Show desktop"),
                                    ("point_enabled", "Point", "Press Esc")):
            toggle = Toggle(panel, title, action, lambda name=name: self._toggle_gesture(name))
            toggle.pack(fill="x", pady=(0, 6))
            toggle.set(getattr(self.settings, name))
            self.gesture_toggles[name] = toggle

        tk.Frame(panel, bg=BACKGROUND).pack(fill="both", expand=True)  # pushes the rest down
        self.pause_button = Button(panel, "", self._toggle_pause)
        self.pause_button.pack(fill="x", pady=(12, 6))
        self._refresh_pause()
        Button(panel, "Quit", self.stop).pack(fill="x")
        tk.Label(panel, text="Esc also quits", bg=BACKGROUND, fg=MUTED,
                 font=(FONT, 9)).pack(pady=(6, 0))

    @staticmethod
    def _heading(parent, text):
        tk.Label(parent, text=text.upper(), bg=BACKGROUND, fg=MUTED, font=(FONT, 9, "bold"),
                 anchor="w").pack(fill="x", pady=(18, 6))

    # --- button actions --------------------------------------------------------

    def _set_smoothing(self, value):
        self.settings.smoothing = value
        for level, button in self.smoothing_buttons.items():
            if level == value:
                button.set_colour(ACCENT, ACCENT)
            else:
                button.set_colour(CARD, CARD_HOVER)

    def _toggle_gesture(self, name):
        enabled = not getattr(self.settings, name)
        setattr(self.settings, name, enabled)
        self.gesture_toggles[name].set(enabled)

    def _toggle_pause(self):
        self.paused = not self.paused
        self._refresh_pause()

    def _refresh_pause(self):
        if self.paused:
            self.pause_button.configure(text="Resume hand control")
            self.pause_button.set_colour(GREEN, "#16a34a")
        else:
            self.pause_button.configure(text="Pause hand control")
            self.pause_button.set_colour(CARD, CARD_HOVER)

    # --- camera loop -----------------------------------------------------------

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

        try:
            # While paused the controller is told there is no hand, so it lets go of everything.
            self.controller.update(None if self.paused else hand, width, height)
        except pyautogui.FailSafeException:
            print("Mouse moved into a screen corner: stopping (PyAutoGUI fail-safe).")
            self.stop()
            return

        self._show(frame, hand)
        self.root.after(1, self.step)

    def _show(self, frame, hand):
        now = time.monotonic()
        instant_fps = 1.0 / max(now - self.last_frame_time, 1e-6)
        self.fps = instant_fps if self.fps == 0 else 0.9 * self.fps + 0.1 * instant_fps
        self.last_frame_time = now

        self.frame_count += 1
        if self.frame_count % VIDEO_EVERY == 1:
            if hand is not None:
                draw_hand(frame, hand)
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (VIDEO_WIDTH, round(height * VIDEO_WIDTH / width)))
            # Tk can read the PPM image format directly, so no extra image library is needed.
            encoded = cv2.imencode(".ppm", frame)[1].tobytes()
            self.video_image = tk.PhotoImage(data=encoded, format="PPM")
            self.video.configure(image=self.video_image)

        if self.paused:
            state = "Paused"
        elif hand is None:
            state = "No hand in view"
        elif self.controller.button_down:
            state = "Pinching"
        elif self.controller.gesture is not None:
            state = f"Gesture {self.controller.gesture}"
        else:
            state = "Tracking"
        self.status.configure(text=f"{state}   |   {self.fps:.0f} FPS",
                              fg=MUTED if self.paused or hand is None else GREEN)

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
