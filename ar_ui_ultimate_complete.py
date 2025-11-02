import cv2
import mediapipe as mp
import math
import numpy as np
import pygame
import os
from datetime import datetime

# Initialize pygame for sounds
pygame.mixer.init()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ULTIMATE: All features combined
buttons = [
    {"name": "Play", "x": 30, "y": 60, "w": 120, "h": 60, "color": (0, 200, 255), "action": "play"},
    {"name": "Pause", "x": 180, "y": 60, "w": 120, "h": 60, "color": (255, 255, 100), "action": "pause"},
    {"name": "Stop", "x": 330, "y": 60, "w": 120, "h": 60, "color": (255, 100, 100), "action": "stop"},
    {"name": "Reset", "x": 480, "y": 60, "w": 120, "h": 60, "color": (200, 200, 200), "action": "reset"},
    {"name": "Save", "x": 630, "y": 60, "w": 120, "h": 60, "color": (255, 0, 255), "action": "save"}
]

colors = [
    {"name": "Red", "x": 30, "y": 450, "w": 40, "h": 40, "color": (0, 0, 255)},
    {"name": "Green", "x": 90, "y": 450, "w": 40, "h": 40, "color": (0, 255, 0)},
    {"name": "Blue", "x": 150, "y": 450, "w": 40, "h": 40, "color": (255, 0, 0)},
    {"name": "Yellow", "x": 210, "y": 450, "w": 40, "h": 40, "color": (0, 255, 255)},
    {"name": "Purple", "x": 270, "y": 450, "w": 40, "h": 40, "color": (255, 0, 255)},
    {"name": "White", "x": 330, "y": 450, "w": 40, "h": 40, "color": (255, 255, 255)}
]

canvas = None
current_color = (255, 255, 255)
current_action = "None"
volume_level = 50
zoom_level = 1.0
save_count = 0
left_hand_active = False
right_hand_active = False
current_gesture = "None"

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def reset_canvas():
    global canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

def save_drawing():
    global save_count
    if not os.path.exists('saved_drawings'):
        os.makedirs('saved_drawings')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_drawings/drawing_{timestamp}_{save_count}.png"
    cv2.imwrite(filename, canvas)
    save_count += 1
    return filename

def detect_gesture(hand_landmarks):
    """Advanced gesture detection"""
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)
    
    # Gesture thresholds
    if (thumb_index_dist < 0.1 and index_middle_dist < 0.1 and 
        middle_ring_dist < 0.1 and ring_pinky_dist < 0.1):
        return "FIST"
    elif (thumb_index_dist > 0.15 and index_middle_dist < 0.1 and 
          middle_ring_dist < 0.1 and ring_pinky_dist < 0.1):
        return "POINT"
    elif (thumb_index_dist < 0.1 and index_middle_dist > 0.15 and 
          middle_ring_dist < 0.1 and ring_pinky_dist < 0.1):
        return "TWO_FINGERS"
    elif (thumb_index_dist > 0.15 and index_middle_dist < 0.1 and 
          middle_ring_dist < 0.1 and ring_pinky_dist > 0.15):
        return "I_LOVE_YOU"
    elif thumb_index_dist < 0.05:
        return "PINCH"
    elif (thumb_index_dist > 0.12 and index_middle_dist > 0.12 and 
          middle_ring_dist > 0.12 and ring_pinky_dist > 0.12):
        return "OPEN_HAND"
    
    return "UNKNOWN"

reset_canvas()

# ULTIMATE: Best settings for all features
with mp_hands.Hands(
    min_detection_confidence=0.6,    # Balanced sensitivity
    min_tracking_confidence=0.6,     # Good tracking
    max_num_hands=2                  # Two-hand support
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
        
        # Volume and Zoom sliders
        cv2.rectangle(image, (1000, 450), (1100, 550), (100, 100, 100), -1)
        cv2.rectangle(image, (1000, 550 - volume_level), (1100, 550), (200, 200, 0), -1)
        cv2.putText(image, f"Vol: {volume_level}%", (1000, 430), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(image, f"Zoom: {zoom_level:.1f}x", (1000, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        current_action = "None"
        current_gesture = "None"
        left_hand_active = False
        right_hand_active = False
        save_message = ""
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Hand type detection
                hand_type = "Unknown"
                if results.multi_handedness:
                    hand_type = results.multi_handedness[hand_idx].classification[0].label
                
                if hand_type == "Left":
                    left_hand_active = True
                    hand_color = (255, 0, 0)  # Blue
                else:
                    right_hand_active = True
                    hand_color = (0, 0, 255)  # Red
                
                # Gesture detection
                gesture = detect_gesture(hand_landmarks)
                current_gesture = gesture
                
                # Get finger positions
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
                
                # Gesture-based coloring
                gesture_colors = {
                    "FIST": (0, 0, 255), "POINT": (0, 255, 0),
                    "TWO_FINGERS": (255, 255, 0), "I_LOVE_YOU": (255, 0, 255),
                    "PINCH": (255, 165, 0), "OPEN_HAND": (0, 255, 255)
                }
                
                circle_color = gesture_colors.get(gesture, hand_color)
                cv2.circle(image, (index_x, index_y), 10, circle_color, -1)
                
                # ULTIMATE: Multi-hand, multi-gesture control system
                if gesture == "PINCH":
                    # Either hand can draw
                    cv2.circle(canvas, (index_x, index_y), 5, current_color, -1)
                    current_action = "Drawing"
                
                elif gesture == "POINT" and hand_type == "Right":
                    # Right hand points for UI interaction
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
                            elif button["action"] == "save":
                                filename = save_drawing()
                                save_message = f"Saved: drawing_{save_count-1}.png"
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
                
                elif gesture == "TWO_FINGERS" and hand_type == "Left":
                    # Left hand controls zoom
                    if 1000 < middle_x < 1100 and 450 < middle_y < 550:
                        zoom_level = (550 - middle_y) / 33.3
                        zoom_level = max(1.0, min(3.0, zoom_level))
                        current_action = f"Zoom: {zoom_level:.1f}x"
                
                elif gesture == "FIST":
                    current_action = "Grab Mode"
                    cv2.putText(image, "GRAB!", (index_x + 15, index_y - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                elif gesture == "I_LOVE_YOU":
                    current_action = "Special Command!"
                    cv2.putText(image, "I LOVE YOU! 💖", (w//2 - 120, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)
                
                # Volume control (left hand middle finger)
                if hand_type == "Left" and 1000 < middle_x < 1100 and 450 < middle_y < 550:
                    volume_level = 550 - middle_y
                    volume_level = max(0, min(100, volume_level))
                
                # Draw hand with labels
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, f"{hand_type}: {gesture}", 
                           (index_x + 15, index_y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 2)
        
        # Blend canvas
        image = cv2.addWeighted(image, 0.7, canvas, 0.3, 0)
        
        # ULTIMATE: Comprehensive status display
        cv2.putText(image, f"Action: {current_action}", (30, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(image, f"Gesture: {current_gesture}", (30, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        hand_status = f"Hands: L[{left_hand_active}] R[{right_hand_active}]"
        cv2.putText(image, hand_status, (30, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if save_message:
            cv2.putText(image, save_message, (30, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(image, f"Saved: {save_count} drawings", (30, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Feature legend
        features_text = "ULTIMATE: Colors+Buttons+Save+2Hands+Gestures+Zoom"
        cv2.putText(image, features_text, (30, 650), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('ULTIMATE AR Hand UI - Press Q to quit', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()