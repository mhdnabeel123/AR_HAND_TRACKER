import cv2
import mediapipe as mp
import math
import numpy as np
import pygame
import time

# Initialize pygame mixer for sounds
pygame.mixer.init()

# Load sound effects
try:
    click_sound = pygame.mixer.Sound(pygame.mixer.Sound(bytes(1000)))  # Placeholder - you'll need actual sound files
    draw_sound = pygame.mixer.Sound(pygame.mixer.Sound(bytes(800)))
    button_sound = pygame.mixer.Sound(pygame.mixer.Sound(bytes(600)))
    print("Sound system initialized!")
except:
    print("Sound files not found - continuing without audio")
    click_sound = draw_sound = button_sound = None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

buttons = [
    {"name": "Play", "x": 50, "y": 80, "w": 140, "h": 70, "color": (0, 255, 0), "action": "play", "sound": True},
    {"name": "Pause", "x": 250, "y": 80, "w": 140, "h": 70, "color": (255, 255, 0), "action": "pause", "sound": True},
    {"name": "Stop", "x": 450, "y": 80, "w": 140, "h": 70, "color": (0, 0, 255), "action": "stop", "sound": True},
    {"name": "Reset", "x": 650, "y": 80, "w": 140, "h": 70, "color": (128, 128, 128), "action": "reset", "sound": True}
]

colors = [
    {"name": "Red", "x": 50, "y": 500, "w": 50, "h": 50, "color": (0, 0, 255), "sound": True},
    {"name": "Green", "x": 120, "y": 500, "w": 50, "h": 50, "color": (0, 255, 0), "sound": True},
    {"name": "Blue", "x": 190, "y": 500, "w": 50, "h": 50, "color": (255, 0, 0), "sound": True},
    {"name": "Yellow", "x": 260, "y": 500, "w": 50, "h": 50, "color": (0, 255, 255), "sound": True},
    {"name": "White", "x": 330, "y": 500, "w": 50, "h": 50, "color": (255, 255, 255), "sound": True}
]

canvas = None
current_color = (255, 255, 255)
drawing_mode = False
current_action = "None"
volume_level = 50
last_sound_time = 0
sound_cooldown = 0.2  # Prevent sound spamming

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def reset_canvas():
    global canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

def play_sound(sound_type):
    global last_sound_time
    current_time = time.time()
    
    # Cooldown to prevent sound spamming
    if current_time - last_sound_time < sound_cooldown:
        return
    
    last_sound_time = current_time
    
    if click_sound:
        try:
            if sound_type == "click":
                click_sound.play()
            elif sound_type == "draw":
                draw_sound.play()
            elif sound_type == "button":
                button_sound.play()
        except:
            pass  # Silently fail if sounds aren't available

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
        
        current_action = "None"
        drawing_mode = False
        sound_played_this_frame = False
        
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
                
                # Pinch gesture with sound
                pinch_distance = calculate_distance(index_tip, thumb_tip)
                if pinch_distance < 0.05:
                    if not drawing_mode:  # Play sound when starting to draw
                        play_sound("draw")
                    drawing_mode = True
                    cv2.circle(canvas, (index_x, index_y), 5, current_color, -1)
                else:
                    drawing_mode = False
                
                # Button interactions with sound
                button_clicked = False
                for button in buttons:
                    if (button["x"] < index_x < button["x"] + button["w"] and 
                        button["y"] < index_y < button["y"] + button["h"]):
                        current_action = button["action"]
                        cv2.rectangle(image, 
                                     (button["x"] - 5, button["y"] - 5), 
                                     (button["x"] + button["w"] + 5, button["y"] + button["h"] + 5), 
                                     (255, 255, 255), 3)
                        
                        if button["sound"] and not sound_played_this_frame:
                            play_sound("button")
                            sound_played_this_frame = True
                        
                        if button["action"] == "reset":
                            reset_canvas()
                            play_sound("click")
                        button_clicked = True
                        break
                
                # Color selection with sound
                if not button_clicked:
                    for color_box in colors:
                        if (color_box["x"] < index_x < color_box["x"] + color_box["w"] and 
                            color_box["y"] < index_y < color_box["y"] + color_box["h"]):
                            if current_color != color_box["color"] and color_box["sound"]:
                                play_sound("click")
                            current_color = color_box["color"]
                            cv2.rectangle(image, 
                                         (color_box["x"] - 5, color_box["y"] - 5), 
                                         (color_box["x"] + color_box["w"] + 5, color_box["y"] + color_box["h"] + 5), 
                                         (255, 255, 255), 3)
                            break
                
                # Volume control
                if 1000 < middle_x < 1100 and 500 < middle_y < 600:
                    new_volume = 600 - middle_y
                    new_volume = max(0, min(100, new_volume))
                    if new_volume != volume_level:
                        volume_level = new_volume
                        # Optional: Play volume change sound
                
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Blend canvas
        image = cv2.addWeighted(image, 0.7, canvas, 0.3, 0)
        
        # Status display with sound indicator
        cv2.putText(image, f"Action: {current_action}", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if drawing_mode:
            cv2.putText(image, "Drawing: ACTIVE", (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Sound status
        sound_status = "Sound: ENABLED" if click_sound else "Sound: DISABLED"
        cv2.putText(image, sound_status, (50, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(image, "Sound Effects Enabled", (50, 650), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Sound Effects UI - Press Q to quit', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()