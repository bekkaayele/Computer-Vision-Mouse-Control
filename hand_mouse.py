"""Control the mouse with your hand in front of a webcam.

Run:  python hand_mouse.py
Quit: press Esc, click Quit, or close the window.
"""

import argparse
import threading
import time
import traceback
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

VIDEO_WIDTH = 480
# How often the window asks the camera thread for a new picture, in milliseconds.
REFRESH_MS = 40

# Shown in the status line; other digits are shown as "Digit N".
GESTURE_NAMES = {1: "Point (digit 1)", 2: "Peace sign (digit 2)"}

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
    """The window with the camera view and the controls.

    Two threads share the work. A background thread reads the camera, finds
    the hand and drives the mouse, so hand control keeps running whatever
    happens to the window (minimised, covered, being redrawn). The main
    thread only runs the Tk window: it handles the buttons and shows the most
    recent camera picture.
    """

    def __init__(self, camera_index):
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise SystemExit(f"Could not open camera {camera_index}. "
                             "Is it connected and not in use by another program?")

        # Height of the camera view in the window, from the camera's own picture shape.
        width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480
        self.video_height = round(VIDEO_WIDTH * height / width)

        self.tracker = HandTracker()
        self.settings = Settings()
        self.backend = PyAutoGuiBackend()
        self.controller = HandMouseController(
            build_classifier(), self.backend, pyautogui.size(), self.settings)

        self.paused = False
        self.running = True
        self.stop_requested = threading.Event()
        self.worker = threading.Thread(target=self._camera_loop, daemon=True)
        # What the camera thread last produced, for the window to display.
        self._lock = threading.Lock()
        self._picture = None
        self._state = "Starting"
        self._fps = 0.0
        self._build_window()

    # --- window layout ---------------------------------------------------------

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Hand Mouse")
        self.root.configure(bg=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.bind("<Escape>", lambda event: self.stop())

        # The camera view gets a fixed-size box, so the layout does not depend
        # on whether a picture has arrived yet.
        video_box = tk.Frame(self.root, bg="black", width=VIDEO_WIDTH, height=self.video_height)
        video_box.pack(side="left", anchor="n", padx=(16, 0), pady=16)
        video_box.pack_propagate(False)
        self.video = tk.Label(video_box, bg="black", bd=0)
        self.video.pack(fill="both", expand=True)
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

        # Lock the window to the size its contents need. Setting the minimum
        # and maximum size (rather than resizable(False, False)) stops Windows
        # from bringing the window back as a bare title bar after it was
        # minimised by the show-desktop shortcut.
        self.root.update_idletasks()
        self.window_size = (self.root.winfo_reqwidth(), self.root.winfo_reqheight())
        self.root.geometry("%dx%d" % self.window_size)
        self.root.minsize(*self.window_size)
        self.root.maxsize(*self.window_size)

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

    # --- camera thread ---------------------------------------------------------

    def _camera_loop(self):
        """Runs in the background thread until the program stops. No Tk calls here."""
        last_time = time.monotonic()
        fps = 0.0
        try:
            while not self.stop_requested.is_set():
                ok, frame = self.camera.read()
                if not ok:
                    print("The camera stopped sending frames.")
                    break
                frame = cv2.flip(frame, 1)  # mirror, so moving right moves the cursor right
                height, width = frame.shape[:2]

                try:
                    hand = self.tracker.find_hand(frame)
                    # While paused the controller is told there is no hand, so it lets go of everything.
                    self.controller.update(None if self.paused else hand, width, height)
                except pyautogui.FailSafeException:
                    print("Mouse moved into a screen corner: stopping (PyAutoGUI fail-safe).")
                    break
                except Exception:
                    # One bad frame must not end hand control: report it and carry on.
                    traceback.print_exc()
                    continue

                now = time.monotonic()
                instant = 1.0 / max(now - last_time, 1e-6)
                fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
                last_time = now

                if hand is not None:
                    draw_hand(frame, hand)
                picture = cv2.resize(frame, (VIDEO_WIDTH, self.video_height))
                with self._lock:
                    self._picture = picture
                    self._state = self._describe(hand)
                    self._fps = fps
        finally:
            try:
                self.controller.hand_lost()  # never leave the mouse button held down
            except pyautogui.FailSafeException:
                pass
            self.tracker.close()
            self.camera.release()
            self.stop_requested.set()

    def _describe(self, hand):
        if self.paused:
            return "Paused"
        if hand is None:
            return "No hand in view"
        if self.controller.button_down:
            return "Pinching"
        gesture = self.controller.gesture
        if gesture is not None:
            return GESTURE_NAMES.get(gesture, f"Digit {gesture}")
        return "Tracking"

    # --- window refresh --------------------------------------------------------

    def _refresh(self):
        """Runs in the main thread: show the newest picture and status."""
        if not self.running:
            return
        if self.stop_requested.is_set():  # the camera thread has ended
            self.stop()
            return

        with self._lock:
            picture, self._picture = self._picture, None
            state, fps = self._state, self._fps

        # Nothing to draw while the window is minimised.
        if self.root.state() != "iconic":
            # Safety net: if the window ever comes back smaller than it should be, restore it.
            if (self.root.winfo_width(), self.root.winfo_height()) != self.window_size:
                self.root.geometry("%dx%d" % self.window_size)
            if picture is not None:
                # Tk can read the PPM image format directly, so no extra image library is needed.
                encoded = cv2.imencode(".ppm", picture)[1].tobytes()
                self.video_image = tk.PhotoImage(data=encoded, format="PPM")
                self.video.configure(image=self.video_image)
            active = state not in ("Paused", "No hand in view", "Starting")
            self.status.configure(text=f"{state}   |   {fps:.0f} FPS",
                                  fg=GREEN if active else MUTED)
        self.root.after(REFRESH_MS, self._refresh)

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.stop_requested.set()
        if self.worker.is_alive():
            self.worker.join(timeout=3)  # the camera thread cleans up after itself
        self.root.destroy()

    def run(self):
        self.worker.start()
        self.root.after(0, self._refresh)
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
