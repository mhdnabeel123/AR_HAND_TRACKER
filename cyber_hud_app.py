import cv2
import mediapipe as mp
import math
import numpy as np
import os
from datetime import datetime

class CyberneticARHUD:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Cybernetic state
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.current_color = (0, 255, 255)  # Cyan - cyber color
        self.current_action = "SYSTEM ONLINE - AWAITING INPUT"
        self.system_status = "OPERATIONAL"
        self.energy_level = 85
        self.cpu_usage = 23
        self.save_count = 0
        self.current_gesture = "SCANNING..."
        self.save_message = ""
        self.save_message_timer = 0
        self.running = True
        self.scan_line_pos = 0
        self.pulse_alpha = 0
        self.pulse_dir = 1
        
        # Cyber colors
        self.cyber_blue = (255, 200, 0)      # BGR - Cyan
        self.cyber_purple = (255, 0, 255)    # BGR - Purple
        self.cyber_green = (0, 255, 0)       # BGR - Green
        self.cyber_red = (0, 0, 255)         # BGR - Red
        self.cyber_yellow = (0, 255, 255)    # BGR - Yellow
        
        # Create saved drawings directory
        if not os.path.exists('cyber_saves'):
            os.makedirs('cyber_saves')
    
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def reset_canvas(self):
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.current_action = "CANVAS PURGED"
        self.system_status = "RESET COMPLETE"
    
    def save_drawing(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cyber_saves/scan_{timestamp}_{self.save_count}.png"
        cv2.imwrite(filename, self.canvas)
        self.save_count += 1
        self.save_message = f"SCAN ARCHIVED: {self.save_count}"
        self.save_message_timer = 100
        self.system_status = "DATA SAVED"
    
    def detect_gesture(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        
        # Get finger tips
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Calculate distances
        thumb_index = self.calculate_distance(thumb_tip, index_tip)
        index_middle = self.calculate_distance(index_tip, middle_tip)
        middle_ring = self.calculate_distance(middle_tip, ring_tip)
        ring_pinky = self.calculate_distance(ring_tip, pinky_tip)
        
        # Gesture recognition (REMOVED SYSTEM_SHUTDOWN)
        if thumb_index < 0.05:
            return "DATA_INPUT"      # 🤏 Pinch - Drawing
        elif thumb_index > 0.15 and index_middle < 0.1:
            return "INTERFACE_SELECT" # 👆 Point - Selection
        elif thumb_index < 0.1 and index_middle > 0.15:
            return "SYSTEM_CONTROL"   # ✌️ Two - Controls
        elif thumb_index > 0.15 and ring_pinky > 0.15:
            return "SPECIAL_COMMAND"  # 🤟 I Love You
        
        return "SCANNING"
    
    def draw_cyber_background(self, image):
        """Draw cybernetic background elements"""
        h, w = image.shape[:2]
        
        # Pulsing border
        self.pulse_alpha += 0.05 * self.pulse_dir
        if self.pulse_alpha > 1.0:
            self.pulse_alpha = 1.0
            self.pulse_dir = -1
        elif self.pulse_alpha < 0.3:
            self.pulse_alpha = 0.3
            self.pulse_dir = 1
            
        pulse_color = tuple(int(c * self.pulse_alpha) for c in self.cyber_blue)
        
        # Border frame
        cv2.rectangle(image, (5, 5), (w-5, h-5), pulse_color, 2)
        cv2.rectangle(image, (8, 8), (w-8, h-8), self.cyber_purple, 1)
        
        # Corner brackets
        bracket_size = 20
        # Top-left
        cv2.line(image, (10, 10), (10+bracket_size, 10), self.cyber_green, 2)
        cv2.line(image, (10, 10), (10, 10+bracket_size), self.cyber_green, 2)
        # Top-right
        cv2.line(image, (w-10, 10), (w-10-bracket_size, 10), self.cyber_green, 2)
        cv2.line(image, (w-10, 10), (w-10, 10+bracket_size), self.cyber_green, 2)
        # Bottom-left
        cv2.line(image, (10, h-10), (10+bracket_size, h-10), self.cyber_green, 2)
        cv2.line(image, (10, h-10), (10, h-10-bracket_size), self.cyber_green, 2)
        # Bottom-right
        cv2.line(image, (w-10, h-10), (w-10-bracket_size, h-10), self.cyber_green, 2)
        cv2.line(image, (w-10, h-10), (w-10, h-10-bracket_size), self.cyber_green, 2)
        
        # Scanning line
        self.scan_line_pos = (self.scan_line_pos + 2) % h
        cv2.line(image, (0, self.scan_line_pos), (w, self.scan_line_pos), self.cyber_blue, 1)
        if self.scan_line_pos % 20 < 10:
            cv2.line(image, (0, self.scan_line_pos), (w, self.scan_line_pos), self.cyber_purple, 1)
    
    def draw_cyber_ui(self, image):
        """Draw cybernetic UI elements"""
        h, w = image.shape[:2]
        
        # 🎨 DATA PALLETE - Right panel
        colors = [
            {"name": "CYAN", "x": w-80, "y": 150, "w": 50, "h": 30, "color": self.cyber_blue},
            {"name": "PURPLE", "x": w-80, "y": 190, "w": 50, "h": 30, "color": self.cyber_purple},
            {"name": "GREEN", "x": w-80, "y": 230, "w": 50, "h": 30, "color": self.cyber_green},
            {"name": "RED", "x": w-80, "y": 270, "w": 50, "h": 30, "color": self.cyber_red},
            {"name": "YELLOW", "x": w-80, "y": 310, "w": 50, "h": 30, "color": self.cyber_yellow}
        ]
        
        cv2.putText(image, "DATA PALLETE", (w-120, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cyber_blue, 1)
        
        for color in colors:
            # Cyber-style color boxes
            cv2.rectangle(image, (color["x"], color["y"]), 
                         (color["x"] + color["w"], color["y"] + color["h"]), 
                         (40, 40, 40), -1)
            cv2.rectangle(image, (color["x"]+2, color["y"]+2), 
                         (color["x"] + color["w"] - 2, color["y"] + color["h"] - 2), 
                         color["color"], -1)
            
            # Active selection glow
            if tuple(self.current_color) == tuple(color["color"]):
                cv2.rectangle(image, (color["x"]-3, color["y"]-3), 
                             (color["x"] + color["w"] + 3, color["y"] + color["h"] + 3), 
                             self.cyber_green, 2)
                cv2.putText(image, "ACTIVE", (color["x"]-10, color["y"] + color["h"] + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.cyber_green, 1)
        
        # 🎛️ COMMAND INTERFACE - Bottom panel
        commands = [
            {"name": "INIT", "x": 100, "y": h-80, "w": 80, "h": 40, "color": self.cyber_green, "action": "init"},
            {"name": "HALT", "x": 200, "y": h-80, "w": 80, "h": 40, "color": self.cyber_yellow, "action": "halt"},
            {"name": "ABORT", "x": 300, "y": h-80, "w": 80, "h": 40, "color": self.cyber_red, "action": "abort"},
            {"name": "PURGE", "x": 400, "y": h-80, "w": 80, "h": 40, "color": (100, 100, 100), "action": "purge"},
            {"name": "ARCHIVE", "x": 500, "y": h-80, "w": 80, "h": 40, "color": self.cyber_purple, "action": "archive"}
        ]
        
        cv2.putText(image, "COMMAND INTERFACE", (100, h-100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cyber_blue, 1)
        
        for cmd in commands:
            # Cyber buttons with glow effect
            cv2.rectangle(image, (cmd["x"], cmd["y"]), 
                         (cmd["x"] + cmd["w"], cmd["y"] + cmd["h"]), 
                         (30, 30, 30), -1)
            cv2.rectangle(image, (cmd["x"]+1, cmd["y"]+1), 
                         (cmd["x"] + cmd["w"] - 1, cmd["y"] + cmd["h"] - 1), 
                         cmd["color"], -1)
            
            # Button text
            text_size = cv2.getTextSize(cmd["name"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = cmd["x"] + (cmd["w"] - text_size[0]) // 2
            text_y = cmd["y"] + (cmd["h"] + text_size[1]) // 2
            cv2.putText(image, cmd["name"], (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 📊 SYSTEM MONITOR - Left panel
        cv2.putText(image, "SYSTEM MONITOR", (30, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cyber_blue, 1)
        
        # Energy bar
        cv2.rectangle(image, (30, 150), (70, 250), (40, 40, 40), -1)
        energy_height = int((self.energy_level / 100) * 100)
        cv2.rectangle(image, (32, 250 - energy_height), (68, 248), self.cyber_blue, -1)
        cv2.putText(image, f"ENERGY: {self.energy_level}%", (20, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.cyber_green, 1)
        
        # CPU usage
        cv2.rectangle(image, (80, 150), (120, 250), (40, 40, 40), -1)
        cpu_height = int((self.cpu_usage / 100) * 100)
        cv2.rectangle(image, (82, 250 - cpu_height), (118, 248), self.cyber_purple, -1)
        cv2.putText(image, f"CPU: {self.cpu_usage}%", (75, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.cyber_green, 1)
        
        return commands, colors
    
    def draw_hud_data(self, image):
        """Draw HUD data displays"""
        h, w = image.shape[:2]
        
        # Main status display
        cv2.putText(image, ">> CYBERNETIC AR HUD <<", (w//2 - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.cyber_blue, 2)
        
        # System status
        status_color = self.cyber_green if self.system_status == "OPERATIONAL" else self.cyber_red
        cv2.putText(image, f"STATUS: {self.system_status}", (w//2 - 100, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Current action
        cv2.putText(image, f"> {self.current_action}", (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.cyber_yellow, 1)
        
        # Gesture recognition
        cv2.putText(image, f"GESTURE: {self.current_gesture}", (50, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cyber_purple, 1)
        
        # Save status
        if self.save_message_timer > 0:
            cv2.putText(image, f"ARCHIVE: {self.save_message}", (50, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cyber_green, 1)
            self.save_message_timer -= 1
        
        # Data logs
        cv2.putText(image, f"DATA FILES: {self.save_count}", (w-200, h-120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cyber_blue, 1)
        
        # Command guide (UPDATED - Removed SYSTEM_SHUTDOWN)
        commands = [
            "COMMAND GUIDE:",
            "INTERFACE_SELECT - Choose options",
            "DATA_INPUT - Write data", 
            "SYSTEM_CONTROL - Adjust systems",
            "SPECIAL_COMMAND - Execute special",
            "Press Q to exit system"
        ]
        
        for i, cmd in enumerate(commands):
            cv2.putText(image, cmd, (w-250, 400 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.cyber_yellow, 1)
    
    def process_cyber_hand(self, hand_landmarks, hand_type, image, commands, colors):
        """Process hand interactions with cyber theme"""
        # Get finger positions
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]
        
        h, w, _ = image.shape
        index_x = int(index_tip.x * w)
        index_y = int(index_tip.y * h)
        middle_x = int(middle_tip.x * w)
        middle_y = int(middle_tip.y * h)
        
        # Detect gesture
        gesture = self.detect_gesture(hand_landmarks)
        self.current_gesture = gesture
        
        # Draw cyber fingertip
        gesture_colors = {
            "INTERFACE_SELECT": self.cyber_green,
            "DATA_INPUT": self.cyber_blue,
            "SYSTEM_CONTROL": self.cyber_purple,
            "SPECIAL_COMMAND": self.cyber_yellow
        }
        
        tip_color = gesture_colors.get(gesture, self.cyber_red)
        
        # Cyber fingertip with glow
        cv2.circle(image, (index_x, index_y), 12, tip_color, -1)
        cv2.circle(image, (index_x, index_y), 15, tip_color, 2)
        cv2.circle(image, (index_x, index_y), 8, (255, 255, 255), -1)
        
        # Handle cyber gestures
        if gesture == "DATA_INPUT":
            # Drawing mode
            cv2.circle(self.canvas, (index_x, index_y), 4, self.current_color, -1)
            self.current_action = "DATA STREAM ACTIVE"
            self.cpu_usage = min(100, self.cpu_usage + 1)
        
        elif gesture == "INTERFACE_SELECT":
            # Check commands
            for cmd in commands:
                if (cmd["x"] < index_x < cmd["x"] + cmd["w"] and 
                    cmd["y"] < index_y < cmd["y"] + cmd["h"]):
                    self.current_action = f"COMMAND: {cmd['name']}"
                    self.system_status = "EXECUTING"
                    
                    # Command execution
                    if cmd["action"] == "purge":
                        self.reset_canvas()
                    elif cmd["action"] == "archive":
                        self.save_drawing()
                    
                    # Highlight command
                    cv2.rectangle(image, (cmd["x"]-4, cmd["y"]-4), 
                                 (cmd["x"] + cmd["w"] + 4, cmd["y"] + cmd["h"] + 4), 
                                 self.cyber_green, 3)
                    return
            
            # Check colors
            for color in colors:
                if (color["x"] < index_x < color["x"] + color["w"] and 
                    color["y"] < index_y < color["y"] + color["h"]):
                    self.current_color = color["color"]
                    self.current_action = f"COLOR: {color['name']}"
                    self.system_status = "PALLETE UPDATED"
                    return
        
        elif gesture == "SYSTEM_CONTROL":
            # System controls
            if 30 < middle_x < 70 and 150 < middle_y < 250:  # Energy
                self.energy_level = (250 - middle_y)
                self.energy_level = max(0, min(100, self.energy_level))
                self.current_action = f"ENERGY: {self.energy_level}%"
            elif 80 < middle_x < 120 and 150 < middle_y < 250:  # CPU
                self.cpu_usage = (250 - middle_y)
                self.cpu_usage = max(0, min(100, self.cpu_usage))
                self.current_action = f"CPU: {self.cpu_usage}%"
        
        elif gesture == "SPECIAL_COMMAND":
            self.current_action = "SPECIAL COMMAND EXECUTED"
            self.system_status = "SPECIAL MODE"
            cv2.putText(image, "SPECIAL MODE ENGAGED", (w//2-120, h//2-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.cyber_yellow, 2)
    
    def run(self):
        print("🤖 CYBERNETIC AR HUD SYSTEM")
        print("=====================================")
        print(">> SYSTEM: ONLINE")
        print(">> GESTURE COMMANDS:")
        print("   INTERFACE_SELECT - Choose options")
        print("   DATA_INPUT       - Write data") 
        print("   SYSTEM_CONTROL   - Adjust systems")
        print("   SPECIAL_COMMAND  - Execute special")
        print(">> Press Q to exit system")
        print("=====================================")
        
        try:
            while self.cap.isOpened() and self.running:
                success, image = self.cap.read()
                if not success:
                    continue
                
                # Flip and get dimensions
                image = cv2.flip(image, 1)
                h, w = image.shape[:2]
                
                # Process hands
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)
                
                # Draw cyber background
                self.draw_cyber_background(image)
                
                # Draw UI and get command/color positions
                commands, colors = self.draw_cyber_ui(image)
                
                # Process cyber hands
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        hand_type = "BIOMETRIC_INPUT"
                        self.process_cyber_hand(hand_landmarks, hand_type, image, commands, colors)
                        
                        # Draw cyber hand landmarks
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=self.cyber_blue, thickness=2, circle_radius=3),
                            self.mp_drawing.DrawingSpec(color=self.cyber_purple, thickness=2)
                        )
                
                # Blend data canvas
                image = cv2.addWeighted(image, 0.8, self.canvas, 0.2, 0)
                
                # Draw HUD data
                self.draw_hud_data(image)
                
                # Display
                cv2.imshow('CYBERNETIC AR HUD v2.0', image)
                
                # Keyboard controls (ONLY way to exit now)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_canvas()
                elif key == ord('s'):
                    self.save_drawing()
        
        except Exception as e:
            print(f"SYSTEM ERROR: {e}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print(">> SYSTEM: OFFLINE")
            print("👋 Cybernetic AR HUD terminated")

def main():
    try:
        hud = CyberneticARHUD()
        hud.run()
    except Exception as e:
        print(f"STARTUP ERROR: {e}")

if __name__ == "__main__":
    main()