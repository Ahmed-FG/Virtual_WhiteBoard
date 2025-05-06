# Requirements
# Python: 3.10.11
# Mediapipe version: 0.10.9
# OpenCV version: 4.10.0
# NumPy version: 1.26.4
# Pillow version: 11.0.0
# python -m pip install mediapipe opencv-python numpy pillow

import time
import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image, ImageSequence

# MediaPipe Gesture Recognizer Setup
BaseOptions = mp.tasks.BaseOptions
model_path = 'gesture_recognizer.task'

base_options = BaseOptions(model_asset_path=model_path)
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Gesture-Related and Drawing Variables
is_drawingEnabled = False
thickness = 4 # Drawing thickness
color = (0, 0, 139) # Drawing Color
prevxy = None # Initialize previous x,y for the detected gesture
# lastxy = None # Keeps Last Draw Location

def draw_info_text(whiteboard, gesture):    
    if gesture != "":
        cv.putText(whiteboard, "Finger Gesture: " + gesture.category_name + ", Confidence: " + str(gesture.score), (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(whiteboard, "Finger Gesture: " + gesture.category_name + ", Confidence: " + str(gesture.score), (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    return whiteboard

# Function to set hand_landmark location on window
def get_handlandmark_location(hand_landmarks, hand_landmark_point, window_width, window_height):
    x = int(hand_landmarks[hand_landmark_point].x * window_width)
    y = int(hand_landmarks[hand_landmark_point].y * window_height)
    if 0 <= x < window_width and 0 <= y < window_height:  # Valid area check
        return x, y

# Function to create a dotted whiteboard background
def create_dotted_whiteboard(height, width, spacing=30):
    whiteboard = np.ones((height, width, 3), dtype='uint8') * 255
    for y in range(0, height, spacing):
        for x in range(0, width, spacing):
            cv.circle(whiteboard, (x, y), 1, (200, 200, 200), -1)
    return whiteboard

# Function to draw eraser as a rectangle and set eraser
def draw_eraser_rectangle(whiteboard, mask, center, width, height, color=(0, 0, 0)):
    cx, cy = center
    half_width, half_height = width // 2, height // 2
    x1, y1 = cx - half_width, cy - half_height
    x2, y2 = cx + half_width, cy + half_height

    # Draw Rectangle
    cv.rectangle(whiteboard, (x1, y1), (x2, y2), color, thickness)
    # Erase
    mask[y1:y2, x1:x2] = 0

# Function to load GIF frames
def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    frames = []
    for frame in ImageSequence.Iterator(gif):
        frame_rgba = frame.convert("RGBA")
        np_frame = np.array(frame_rgba)
        np_frame = cv.cvtColor(np_frame, cv.COLOR_RGBA2BGRA)
        frames.append(np_frame)
    return frames

# Function to overlay GIF frames
def overlay_gif_on_frame(background, overlay, center_x, center_y):
    overlay_rgb = overlay[:, :, :3]
    overlay_alpha = overlay[:, :, 3] / 255.0
    h, w = overlay_rgb.shape[:2]

    x1 = int(center_x - w / 2)
    y1 = int(center_y - h / 2)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x1 + w, background.shape[1]), min(y1 + h, background.shape[0])

    overlay_cropped = overlay[0:y2-y1, 0:x2-x1]
    overlay_rgb = overlay_cropped[:, :, :3]
    overlay_alpha = overlay_cropped[:, :, 3] / 255.0

    roi = background[y1:y2, x1:x2]

    for c in range(3):
        roi[:, :, c] = (overlay_rgb[:, :, c] * overlay_alpha +
                        roi[:, :, c] * (1 - overlay_alpha))

    background[y1:y2, x1:x2] = roi
    return background

# Load GIFs
waving_hand_path = "waving-hand.gif"
waving_hand_frames = load_gif_frames(waving_hand_path)
victory_hand_path = "victory-hand.gif"
victory_hand_frames = load_gif_frames(victory_hand_path)
gif_index1, gif_index2 = 0, 0

# Set GestureRecognizerOptions
options = GestureRecognizerOptions(base_options, running_mode=VisionRunningMode.VIDEO)

# Create a resizable window
window_name = "Whiteboard Window"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, 800, 600)

# Set Initial Frame And Camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize whiteboard(background) and mask(drawing,erasing related area)
window_width, window_height = 800, 600
whiteboard = create_dotted_whiteboard(window_height, window_width)
mask = np.zeros_like(whiteboard)

with GestureRecognizer.create_from_options(options) as recognizer:
    start_time = time.time()

    while cap.isOpened():
        ret, camera_frame = cap.read()
        if not ret:
            break

        # Flip Camera Frame Get Mirror Effect
        camera_frame = cv.flip(camera_frame, 1)        
        # Get Window size
        window_rect = cv.getWindowImageRect(window_name)
        window_width, window_height = window_rect[2], window_rect[3]

        # Resize frames if window size changes
        if whiteboard.shape[:2] != (window_height, window_width):
            whiteboard = create_dotted_whiteboard(window_height, window_width)
            mask = cv.resize(mask, (window_width, window_height), interpolation=cv.INTER_NEAREST)

        # Convert camera frame BGR to RGB for MediaPipe
        rgb_camera_frame = cv.cvtColor(camera_frame, cv.COLOR_BGR2RGB)        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_camera_frame)        

        try:
            gesture_result = recognizer.recognize_for_video(mp_image, int((time.time() - start_time) * 1000))
        except Exception:
            continue

        # Handle Gestures
        if gesture_result.gestures:
            for gesture in gesture_result.gestures[0]:
                for hand_landmarks in gesture_result.hand_landmarks:
                    whiteboard = draw_info_text(whiteboard, gesture)
                    # print(f"Gesture: {gesture.category_name}, Confidence: {gesture.score}, is_drawingEnabled: {is_drawingEnabled}")
                    if gesture.category_name == 'Closed_Fist':
                        is_drawingEnabled = True

                    elif gesture.category_name == 'Pointing_Up' and is_drawingEnabled:                    
                        x, y = get_handlandmark_location(hand_landmarks, 8, window_width, window_height)
                        if prevxy is not None:
                            cv.line(mask, prevxy, (x, y), color, thickness)
                            # Last Draw Point
                            # lastxy = get_handlandmark_location(hand_landmarks, 8, 1280, 720)
                        prevxy = (x, y)    

                    elif gesture.category_name == 'Thumb_Up' and is_drawingEnabled:       
                        x, y = get_handlandmark_location(hand_landmarks, 4, window_width, window_height)
                        # draw_dashed_rectangle_center(camera_frame, (x, y), 50, 50, color=(0, 0, 0))
                        draw_eraser_rectangle(whiteboard, mask, (x, y), 50, 50, color=(0, 0, 0))

                    elif gesture.category_name == 'Open_Palm':                        
                        gif_frame = waving_hand_frames[gif_index1] # Get GIF frame by Index                       
                        # Display GIF frame onto the whiteboard frame
                        x, y = get_handlandmark_location(hand_landmarks, 9, window_width, window_height)                        
                        whiteboard = overlay_gif_on_frame(whiteboard, gif_frame, x, y)    
                        # camera_frame = overlay_gif_on_frame(camera_frame, gif_frame, x, y) # Display GIF frame onto the camera frame                        
                        gif_index1 = (gif_index1 + 1) % len(waving_hand_frames) # Update GIF frame index

                    elif gesture.category_name == 'Victory' and is_drawingEnabled:                        
                        gif_frame = victory_hand_frames[gif_index2] # Get GIF frame by Index                       
                        # Display GIF frame onto the whiteboard frame
                        x, y = get_handlandmark_location(hand_landmarks, 9, window_width, window_height)                        
                        whiteboard = overlay_gif_on_frame(whiteboard, gif_frame, x, y)    
                        # camera_frame = overlay_gif_on_frame(camera_frame, gif_frame, x, y) # Display GIF frame onto the camera frame                        
                        gif_index2 = (gif_index2 + 1) % len(victory_hand_frames) # Update GIF frame index

                    elif gesture.category_name == 'Thumb_Down':
                        # Clean the mask (drawing area)
                        mask = np.zeros_like(whiteboard)                    

                    else:
                        prevxy = None  # Reset if hand moves out of frame

        # Visualize Last Draw Point
        # if lastxy is not None:
        #     cv.circle(camera_frame, lastxy, 20, (152, 251, 152), -1)                

        # Combine whiteboard and mask
        output = np.where(mask > 0, mask, whiteboard)

        # Overlay resized camera feed in bottom-right corner
        resized_camera_frame = cv.resize(camera_frame, (200, 150))
        output[-150:, -200:] = resized_camera_frame
        # Display the output
        cv.imshow(window_name, output)
        # Reset whiteboard frame. Because we are not properly keep items in whiteboard
        whiteboard = create_dotted_whiteboard(window_height, window_width)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
