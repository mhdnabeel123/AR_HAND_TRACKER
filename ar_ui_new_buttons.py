import cv2
import mediapipe as mp
import math
import numpy as np
import os
from datetime import datetime

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# NEW: Added Save button and reorganized layout
buttons = [
    {"name": "Play", "x": 50, "y": 80, "w": 120, "h": 60, "color": (0, 255, 0), "action": "play"},
    {"name": "Pause", "x": 200, "y": 80, "w": 120, "h": 60, "color": (255, 255, 0), "action": "pause"},
    {"name": "Stop", "x": 350, "y": 80, "w": 120, "h": 60, "color": (0, 0, 255), "action": "stop"},
    {"name": "Reset", "x": 500, "y": 80, "w": 120, "h": 60, "color": (128, 128, 128), "action": "reset"},
    {"name": "Save", "x": 650, "y": 80, "w": 120, "h": 60, "color": (255, 0, 255), "action": "save"}  # NEW BUTTON
]

colors = [
    {"name": "Red", "x": 50, "y": 500, "w": 50, "h": 50, "color": (0, 0, 255)},
    {"name": "Green", "x": 120, "y": 500, "w": 50, "h": 50, "color": (0, 255, 0)},
    {"name": "Blue", "x": 190, "y": 500, "w": 50, "h": 50, "color": (255, 0, 0)},
    {"name": "Yellow", "x": 260, "y": 500, "w": 50, "h": 50, "color": (0, 255, 255)},
    {"name": "White", "x": 330, "y": 500, "w": 50, "h": 50, "color": (255, 255, 255)}
]

canvas = None
current_color = (255, 255, 255)
drawing_mode = False
current_action = "None"
volume_level = 50
save_count = 0  # NEW: Track saved images

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def reset_canvas():
    global canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

def save_drawing():  # NEW: Save canvas function
    global save_count
    if not os.path.exists('saved_drawings'):
        os.makedirs('saved_drawings')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_drawings/drawing_{timestamp}_{save_count}.png"
    cv2.imwrite(filename, canvas)
    save_count += 1
    return filename

reset_canvas()

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Draw UI with new Save button
        for button in buttons:
            cv2.rectangle(image, 
                         (button["x"], button["y"]), 
                         (button["x"] + button["w"], button["y"] + button["h"]), 
                         button["color"], -1)
            cv2.putText(image, button["name"], 
                       (button["x"] + 15, button["y"] + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for color_box in colors:
            cv2.rectangle(image, 
                         (color_box["x"], color_box["y"]), 
                         (color_box["x"] + color_box["w"], color_box["y"] + color_box["h"]), 
                         color_box["color"], -1)
        
        # Volume slider
        cv2.rectangle(image, (1000, 500), (1100, 600), (100, 100, 100), -1)
        cv2.rectangle(image, (1000, 600 - volume_level), (1100, 600), (200, 200, 0), -1)
        cv2.putText(image, f"Vol: {volume_level}%", (1000, 480), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        current_action = "None"
        drawing_mode = False
        save_message = ""  # NEW: Save confirmation message
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                middle_tip = hand_landmarks.landmark[12]
                
                h, w, _ = image.shape
                index_x = int(index_tip.x * w)
                index_y = int(index_tip.y * h)
                thumb_x = int(thumb_tip.x * w)
                thumb_y = int(thumb_tip.y * h)
                middle_x = int(middle_tip.x * w)
                middle_y = int(middle_tip.y * h)
                
                cv2.circle(image, (index_x, index_y), 10, (0, 255, 0), -1)
                cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
                cv2.circle(image, (middle_x, middle_y), 10, (0, 255, 255), -1)
                
                # Pinch gesture
                pinch_distance = calculate_distance(index_tip, thumb_tip)
                if pinch_distance < 0.05:
                    drawing_mode = True
                    cv2.circle(canvas, (index_x, index_y), 5, current_color, -1)
                
                # Button interactions including NEW Save button
                for button in buttons:
                    if (button["x"] < index_x < button["x"] + button["w"] and 
                        button["y"] < index_y < button["y"] + button["h"]):
                        current_action = button["action"]
                        cv2.rectangle(image, 
                                     (button["x"] - 5, button["y"] - 5), 
                                     (button["x"] + button["w"] + 5, button["y"] + button["h"] + 5), 
                                     (255, 255, 255), 3)
                        
                        if button["action"] == "reset":
                            reset_canvas()
                        elif button["action"] == "save":  # NEW: Handle save action
                            filename = save_drawing()
                            save_message = f"Saved: {filename}"
                        
                        break
                
                # Color selection
                for color_box in colors:
                    if (color_box["x"] < index_x < color_box["x"] + color_box["w"] and 
                        color_box["y"] < index_y < color_box["y"] + color_box["h"]):
                        current_color = color_box["color"]
                        cv2.rectangle(image, 
                                     (color_box["x"] - 5, color_box["y"] - 5), 
                                     (color_box["x"] + color_box["w"] + 5, color_box["y"] + color_box["h"] + 5), 
                                     (255, 255, 255), 3)
                        break
                
                # Volume control
                if 1000 < middle_x < 1100 and 500 < middle_y < 600:
                    volume_level = 600 - middle_y
                    volume_level = max(0, min(100, volume_level))
                
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Blend canvas
        image = cv2.addWeighted(image, 0.7, canvas, 0.3, 0)
        
        # Status display with NEW save message
        cv2.putText(image, f"Action: {current_action}", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if save_message:
            cv2.putText(image, save_message, (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if drawing_mode:
            cv2.putText(image, "Drawing: ACTIVE", (50, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        cv2.putText(image, f"Saved: {save_count} drawings", (50, 650), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('New Buttons UI - Press Q to quit', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()