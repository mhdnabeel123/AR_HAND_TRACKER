import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ENHANCED: Custom colors and positions
buttons = [
    {"name": "Play", "x": 30, "y": 60, "w": 120, "h": 60, "color": (0, 200, 255), "action": "play"},      # Orange
    {"name": "Pause", "x": 180, "y": 60, "w": 120, "h": 60, "color": (255, 255, 100), "action": "pause"}, # Light Yellow
    {"name": "Stop", "x": 330, "y": 60, "w": 120, "h": 60, "color": (255, 100, 100), "action": "stop"},   # Light Red
    {"name": "Reset", "x": 480, "y": 60, "w": 120, "h": 60, "color": (200, 200, 200), "action": "reset"}  # Light Gray
]

# ENHANCED: More color options with better positions
colors = [
    {"name": "Red", "x": 30, "y": 450, "w": 40, "h": 40, "color": (0, 0, 255)},
    {"name": "Green", "x": 90, "y": 450, "w": 40, "h": 40, "color": (0, 255, 0)},
    {"name": "Blue", "x": 150, "y": 450, "w": 40, "h": 40, "color": (255, 0, 0)},
    {"name": "Yellow", "x": 210, "y": 450, "w": 40, "h": 40, "color": (0, 255, 255)},
    {"name": "Purple", "x": 270, "y": 450, "w": 40, "h": 40, "color": (255, 0, 255)},
    {"name": "Cyan", "x": 330, "y": 450, "w": 40, "h": 40, "color": (255, 255, 0)},
    {"name": "White", "x": 390, "y": 450, "w": 40, "h": 40, "color": (255, 255, 255)}
]

# Drawing canvas
canvas = None
current_color = (255, 255, 255)
drawing_mode = False
current_action = "None"
volume_level = 50

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def reset_canvas():
    global canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

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
        
        # Draw enhanced UI
        for button in buttons:
            cv2.rectangle(image, 
                         (button["x"], button["y"]), 
                         (button["x"] + button["w"], button["y"] + button["h"]), 
                         button["color"], -1)
            cv2.putText(image, button["name"], 
                       (button["x"] + 10, button["y"] + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for color_box in colors:
            cv2.rectangle(image, 
                         (color_box["x"], color_box["y"]), 
                         (color_box["x"] + color_box["w"], color_box["y"] + color_box["h"]), 
                         color_box["color"], -1)
        
        # Volume slider
        cv2.rectangle(image, (1000, 450), (1100, 550), (100, 100, 100), -1)
        cv2.rectangle(image, (1000, 550 - volume_level), (1100, 550), (200, 200, 0), -1)
        cv2.putText(image, f"Vol: {volume_level}%", (1000, 430), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        current_action = "None"
        drawing_mode = False
        
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
                
                cv2.circle(image, (index_x, index_y), 8, (0, 255, 0), -1)
                cv2.circle(image, (thumb_x, thumb_y), 8, (255, 0, 0), -1)
                cv2.circle(image, (middle_x, middle_y), 8, (0, 255, 255), -1)
                
                # Pinch gesture for drawing
                pinch_distance = calculate_distance(index_tip, thumb_tip)
                if pinch_distance < 0.05:
                    drawing_mode = True
                    cv2.circle(canvas, (index_x, index_y), 5, current_color, -1)
                
                # Button interactions
                for button in buttons:
                    if (button["x"] < index_x < button["x"] + button["w"] and 
                        button["y"] < index_y < button["y"] + button["h"]):
                        current_action = button["action"]
                        cv2.rectangle(image, 
                                     (button["x"] - 3, button["y"] - 3), 
                                     (button["x"] + button["w"] + 3, button["y"] + button["h"] + 3), 
                                     (255, 255, 255), 2)
                        if button["action"] == "reset":
                            reset_canvas()
                        break
                
                # Color selection
                for color_box in colors:
                    if (color_box["x"] < index_x < color_box["x"] + color_box["w"] and 
                        color_box["y"] < index_y < color_box["y"] + color_box["h"]):
                        current_color = color_box["color"]
                        cv2.rectangle(image, 
                                     (color_box["x"] - 3, color_box["y"] - 3), 
                                     (color_box["x"] + color_box["w"] + 3, color_box["y"] + color_box["h"] + 3), 
                                     (255, 255, 255), 2)
                        break
                
                # Volume control
                if 1000 < middle_x < 1100 and 450 < middle_y < 550:
                    volume_level = 550 - middle_y
                    volume_level = max(0, min(100, volume_level))
                
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Blend canvas
        image = cv2.addWeighted(image, 0.7, canvas, 0.3, 0)
        
        # Status display
        cv2.putText(image, f"Action: {current_action}", (30, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if drawing_mode:
            cv2.putText(image, "Drawing: ACTIVE", (30, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        cv2.putText(image, "Enhanced Colors & Layout", (30, 650), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Enhanced Colors UI - Press Q to quit', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()