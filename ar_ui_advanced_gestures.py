import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

buttons = [
    {"name": "Play", "x": 50, "y": 80, "w": 140, "h": 70, "color": (0, 255, 0), "action": "play"},
    {"name": "Pause", "x": 250, "y": 80, "w": 140, "h": 70, "color": (255, 255, 0), "action": "pause"},
    {"name": "Stop", "x": 450, "y": 80, "w": 140, "h": 70, "color": (0, 0, 255), "action": "stop"},
    {"name": "Reset", "x": 650, "y": 80, "w": 140, "h": 70, "color": (128, 128, 128), "action": "reset"}
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
current_action = "None"
volume_level = 50
zoom_level = 1.0  # NEW: Zoom functionality
current_gesture = "None"  # NEW: Track current gesture

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def reset_canvas():
    global canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

def detect_gesture(hand_landmarks):
    """NEW: Advanced gesture detection"""
    landmarks = hand_landmarks.landmark
    
    # Get key points
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    # Calculate distances
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)
    
    # 👊 FIST detection (all fingers closed)
    fist_threshold = 0.1
    if (thumb_index_dist < fist_threshold and 
        index_middle_dist < fist_threshold and 
        middle_ring_dist < fist_threshold and 
        ring_pinky_dist < fist_threshold):
        return "FIST"
    
    # 👆 POINT detection (only index finger extended)
    point_threshold = 0.15
    if (thumb_index_dist > point_threshold and 
        index_middle_dist < point_threshold and 
        middle_ring_dist < point_threshold and 
        ring_pinky_dist < point_threshold):
        return "POINT"
    
    # ✌️ TWO FINGERS (peace sign)
    peace_threshold = 0.15
    if (thumb_index_dist < peace_threshold and 
        index_middle_dist > peace_threshold and 
        middle_ring_dist < peace_threshold and 
        ring_pinky_dist < peace_threshold):
        return "TWO_FINGERS"
    
    # 🤟 I LOVE YOU (thumb, index, pinky extended)
    ily_threshold = 0.15
    if (thumb_index_dist > ily_threshold and 
        index_middle_dist < ily_threshold and 
        middle_ring_dist < ily_threshold and 
        ring_pinky_dist > ily_threshold):
        return "I_LOVE_YOU"
    
    # 🤏 PINCH (thumb and index close)
    pinch_threshold = 0.05
    if thumb_index_dist < pinch_threshold:
        return "PINCH"
    
    # ✋ OPEN HAND (all fingers extended)
    open_threshold = 0.12
    if (thumb_index_dist > open_threshold and 
        index_middle_dist > open_threshold and 
        middle_ring_dist > open_threshold and 
        ring_pinky_dist > open_threshold):
        return "OPEN_HAND"
    
    return "UNKNOWN"

reset_canvas()

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2  # Enable two hands for complex gestures
) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Draw UI
        for button in buttons:
            cv2.rectangle(image, 
                         (button["x"], button["y"]), 
                         (button["x"] + button["w"], button["y"] + button["h"]), 
                         button["color"], -1)
            cv2.putText(image, button["name"], 
                       (button["x"] + 20, button["y"] + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
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
        
        # NEW: Zoom display
        cv2.putText(image, f"Zoom: {zoom_level:.1f}x", (1000, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        current_action = "None"
        current_gesture = "None"
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # NEW: Detect gesture
                gesture = detect_gesture(hand_landmarks)
                current_gesture = gesture
                
                # Get finger tips for interaction
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
                
                # Color code based on gesture
                gesture_colors = {
                    "FIST": (0, 0, 255),        # Red
                    "POINT": (0, 255, 0),       # Green
                    "TWO_FINGERS": (255, 255, 0), # Yellow
                    "I_LOVE_YOU": (255, 0, 255), # Purple
                    "PINCH": (255, 165, 0),     # Orange
                    "OPEN_HAND": (0, 255, 255), # Cyan
                    "UNKNOWN": (255, 255, 255)  # White
                }
                
                hand_color = gesture_colors.get(gesture, (255, 255, 255))
                
                cv2.circle(image, (index_x, index_y), 12, hand_color, -1)
                cv2.circle(image, (thumb_x, thumb_y), 10, hand_color, -1)
                cv2.circle(image, (middle_x, middle_y), 10, hand_color, -1)
                
                # GESTURE-BASED ACTIONS
                if gesture == "PINCH":
                    # Drawing mode
                    cv2.circle(canvas, (index_x, index_y), 5, current_color, -1)
                    current_action = "Drawing"
                
                elif gesture == "POINT":
                    # Button and color interactions
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
                
                elif gesture == "TWO_FINGERS":
                    # NEW: Zoom control
                    if 1000 < middle_x < 1100 and 500 < middle_y < 600:
                        zoom_level = (600 - middle_y) / 50  # 1x to 3x zoom
                        zoom_level = max(1.0, min(3.0, zoom_level))
                        current_action = f"Zoom: {zoom_level:.1f}x"
                
                elif gesture == "FIST":
                    # NEW: Grab and move (placeholder for object manipulation)
                    current_action = "Grab Mode"
                    cv2.putText(image, "GRAB!", (index_x + 20, index_y - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                elif gesture == "I_LOVE_YOU":
                    # NEW: Special command
                    current_action = "Special Command!"
                    cv2.putText(image, "I LOVE YOU! 💖", (w//2 - 150, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
                
                elif gesture == "OPEN_HAND":
                    current_action = "Open Hand - Ready"
                
                # Volume control (works with any gesture)
                if 1000 < middle_x < 1100 and 500 < middle_y < 600:
                    volume_level = 600 - middle_y
                    volume_level = max(0, min(100, volume_level))
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Display gesture name near hand
                cv2.putText(image, gesture, (index_x + 20, index_y - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)
        
        # Blend canvas
        image = cv2.addWeighted(image, 0.7, canvas, 0.3, 0)
        
        # Enhanced status display with gestures
        cv2.putText(image, f"Action: {current_action}", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(image, f"Gesture: {current_gesture}", (50, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # NEW: Gesture legend
        cv2.putText(image, "Gestures: 👊Fist ✌️Two 👆Point 🤟Love 🤏Pinch", (50, 650), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Advanced Gestures UI - Press Q to quit', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()