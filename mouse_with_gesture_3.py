import cv2
import mediapipe as mp
import pyautogui
import math
import joblib
import numpy as np
import threading
from tkinter import *

# shared boolean flags for both main thread and the gui
motion_scale = 6
peace_enabled = True
pinch_enabled = True
point_enabled = False

# GUI
def start_gui():
    global motion_scale
    global peace_enabled, pinch_enabled, point_enabled

    gui = Tk()
    gui.title("🖐 Hand Mouse Controls")
    gui.geometry("300x250")
    gui.configure(bg="#76B3D3")

    # tkinter variables created after root
    motion_var = IntVar(value=motion_scale, master=gui)
    peace_var = IntVar(value=1, master=gui)
    pinch_var = IntVar(value=1, master=gui)
    point_var = IntVar(value=0, master=gui)

    def on_gui_update():
        nonlocal motion_var, peace_var, pinch_var, point_var # nonlocal variables cause they're not defined globally on the start_gui funciton
        global motion_scale, peace_enabled, pinch_enabled, point_enabled
        
        # changing the global variables from inside the function
        motion_scale = motion_var.get() # returns the vale of motion_var
        peace_enabled = peace_var.get() == 1 # peace_var.get() returns 1 if peace_var is checked
        pinch_enabled = pinch_var.get() == 1
        point_enabled = point_var.get() == 1
        
        gui.after(100, on_gui_update) # runs this function again after 100 milliseconds to check for updates 
                                      # doesnt block the gui thread, basically like a timer to run the function again later
        
    # header
    Label(gui, text="Hand Gesture Mouse", font=("Helvetica", 14, "bold"), bg="#76B3D3").pack(pady=10)

    # motion scale
    Label(gui, text="Cursor Speed", font=("Helvetica", 10), bg="#76B3D3").pack()
    Scale(gui, from_=0, to=15, orient=HORIZONTAL, variable=motion_var, length=200, tickinterval=5).pack(pady=5)
    
    # toggle on and off teh gestures
    Checkbutton(gui, text="✌ Peace = Show Desktop", font=("Helvetica", 10),
                variable=peace_var, bg="#76B3D3", anchor="w", padx=10).pack(fill="x", pady=2)
    Checkbutton(gui, text="🤏 Pinch = Left Click", font=("Helvetica", 10),
                variable=pinch_var, bg="#76B3D3", anchor="w", padx=10).pack(fill="x", pady=2)
    Checkbutton(gui, text="☝ Point = ESC", font=("Helvetica", 10),
                variable=point_var, bg="#76B3D3", anchor="w", padx=10).pack(fill="x", pady=2)

    on_gui_update() # calls gui update which later keeps calling itself to check for updates
    gui.mainloop()

# start GUI in a background thread
gui_thread = threading.Thread(target=start_gui)
gui_thread.daemon = True
gui_thread.start()

# webcam and ML setup
cap = cv2.VideoCapture(0)
model = joblib.load("gesture_model.pkl")

screen_width, screen_height = pyautogui.size()
prev_x, prev_y = 0, 0 # starting mouse position
smoothening = 15 # to reduce "jitteryness" of the mouse

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

peace_frame_counter = 0
point_frame_counter = 0
clicking = False

def distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

while True:
    ret, frame = cap.read()
                # captures a single frame from the webcam
                # ret is true if it succcesfully captured the frame and frame is the frame that was captured 
    if not ret:
        break
    
    frame = cv2.flip(frame, 1) # flips the frame horizontally to mirror the mouse effect
    h, w, _ = frame.shape  # _ for number of channels because we don't really need it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # to convert the bgr cv2 frame to rgb for mediapipe
    result = hands.process(rgb_frame) # this result object contains all teh hand detection info for the frame

    if result.multi_hand_landmarks:# list of detected hands with each hand having 21 specific landmarks
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # position of fingertip landmarks
            thumb_tip = int(landmarks[4].x * w), int(landmarks[4].y * h)
            index_tip = int(landmarks[8].x * w), int(landmarks[8].y * h)
            middle_tip = int(landmarks[12].x * w), int(landmarks[12].y * h)
            
            # mapping where the middle finger is to a location on the screen
            # using the frame width and height to get better mapping
            screen_x = int(landmarks[12].x * 2000)
            screen_y = int(landmarks[12].y * 1500)

            delta_x = (screen_x - prev_x) * motion_scale
            delta_y = (screen_y - prev_y) * motion_scale
            smoothed_x = int(prev_x + delta_x // smoothening)
            smoothed_y = int(prev_y + delta_y // smoothening)
            
            # moving the mouse and saving it's position
            pyautogui.moveTo(smoothed_x, smoothed_y)
            prev_x, prev_y = smoothed_x, smoothed_y
            
            ## model gesture recognition

            features = []
            for lm in landmarks:
                features.extend([lm.x, lm.y, lm.z])
            features = np.array(features).reshape(1, -1)
            gesture = model.predict(features)[0]

            # checking if any action should be taken
            if distance(thumb_tip, index_tip) < 30 and pinch_enabled:
                if not clicking:
                    clicking = True
                    pyautogui.click()
            else:
                clicking = False
        
            if int(gesture) == 2 and peace_enabled:
                peace_frame_counter += 1
                if peace_frame_counter == 5:
                    pyautogui.hotkey('win', 'd')
                    peace_frame_counter = 0
    
            if int(gesture) == 1 and point_enabled:
                point_frame_counter += 1
                if point_frame_counter == 10: # higher threshold because model guesses 1 a lot 
                    pyautogui.press('esc')
                    point_frame_counter = 0
                            
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Mouse", frame)
    if cv2.waitKey(1) == 27: # esc to exit the program
        break

cap.release()
cv2.destroyAllWindows()