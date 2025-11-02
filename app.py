import cv2
import mediapipe as mp
import math
import numpy as np
import os
from datetime import datetime

class ARHandController:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands=2
        )
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Application state
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.current_color = (255, 255, 255)  # Start with white
        self.current_action = "Show your hand to start!"
        self.volume_level = 50
        self.save_count = 0
        self.current_gesture = "None"
        self.save_message = ""
        self.save_message_timer = 0
        self.running = True
        
        # Smooth gesture tracking
        self.gesture_history = []
        self.gesture_smoothness = 5
        
        # Create saved drawings directory
        if not os.path.exists('saved_drawings'):
            os.makedirs('saved_drawings')
        
        # UI Configuration
        self.setup_ui()
    
    def setup_ui(self):
        """Setup beautiful, organized UI elements"""
        
        # 🎨 COLOR PALETTE - Top right corner
        self.colors = [
            {"name": "Red", "x": 1150, "y": 20, "w": 40, "h": 40, "color": (0, 0, 255)},
            {"name": "Green", "x": 1200, "y": 20, "w": 40, "h": 40, "color": (0, 255, 0)},
            {"name": "Blue", "x": 1150, "y": 70, "w": 40, "h": 40, "color": (255, 0, 0)},
            {"name": "Yellow", "x": 1200, "y": 70, "w": 40, "h": 40, "color": (0, 255, 255)},
            {"name": "Purple", "x": 1150, "y": 120, "w": 40, "h": 40, "color": (255, 0, 255)},
            {"name": "White", "x": 1200, "y": 120, "w": 40, "h": 40, "color": (255, 255, 255)}
        ]
        
        # 🎛️ CONTROL BUTTONS - Bottom center
        button_width, button_height = 110, 60
        start_x = 150
        button_y = 620  # Bottom area
        
        self.buttons = [
            {"name": "PLAY", "x": start_x, "y": button_y, "w": button_width, "h": button_height, "color": (50, 200, 50), "action": "play"},
            {"name": "PAUSE", "x": start_x + 130, "y": button_y, "w": button_width, "h": button_height, "color": (50, 150, 255), "action": "pause"},
            {"name": "STOP", "x": start_x + 260, "y": button_y, "w": button_width, "h": button_height, "color": (50, 50, 255), "action": "stop"},
            {"name": "RESET", "x": start_x + 390, "y": button_y, "w": button_width, "h": button_height, "color": (100, 100, 100), "action": "reset"},
            {"name": "SAVE", "x": start_x + 520, "y": button_y, "w": button_width, "h": button_height, "color": (200, 50, 200), "action": "save"}
        ]
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def reset_canvas(self):
        """Clear the drawing canvas"""
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.current_action = "Canvas cleared!"
    
    def save_drawing(self):
        """Save current canvas as image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_drawings/drawing_{timestamp}_{self.save_count}.png"
        cv2.imwrite(filename, self.canvas)
        self.save_count += 1
        self.save_message = f"Saved: drawing_{self.save_count-1}.png"
        self.save_message_timer = 100
        return filename
    
    def smooth_gesture(self, new_gesture):
        """Smooth gesture detection to reduce flickering"""
        self.gesture_history.append(new_gesture)
        if len(self.gesture_history) > self.gesture_smoothness:
            self.gesture_history.pop(0)
        
        # Return the most common gesture in history
        if len(self.gesture_history) == self.gesture_smoothness:
            return max(set(self.gesture_history), key=self.gesture_history.count)
        return new_gesture
    
    def detect_gesture(self, hand_landmarks):
        """Improved gesture detection with better thresholds"""
        landmarks = hand_landmarks.landmark
        
        # Get finger tips and bases
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Get finger bases for extended detection
        thumb_base = landmarks[2]
        index_base = landmarks[5]
        middle_base = landmarks[9]
        ring_base = landmarks[13]
        pinky_base = landmarks[17]
        
        # Calculate distances between finger tips and bases
        thumb_extended = self.calculate_distance(thumb_tip, thumb_base) > 0.15
        index_extended = self.calculate_distance(index_tip, index_base) > 0.15
        middle_extended = self.calculate_distance(middle_tip, middle_base) > 0.15
        ring_extended = self.calculate_distance(ring_tip, ring_base) > 0.15
        pinky_extended = self.calculate_distance(pinky_tip, pinky_base) > 0.15
        
        # Count extended fingers
        extended_fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        extended_count = sum(extended_fingers)
        
        # 🚪 CLOSE APP GESTURE: All fingers closed (fist)
        if extended_count == 0:
            return "CLOSE_APP"
        
        # Gesture recognition
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        
        if extended_count == 1 and index_extended:
            return "POINT"         # 👆
        elif extended_count == 2 and index_extended and middle_extended:
            return "TWO_FINGERS"   # ✌️
        elif extended_count == 3 and index_extended and pinky_extended and thumb_extended:
            return "I_LOVE_YOU"    # 🤟
        elif thumb_index_dist < 0.05 and extended_count >= 2:
            return "PINCH"         # 🤏
        elif extended_count == 5:
            return "OPEN_HAND"     # ✋
        elif extended_count == 0:
            return "FIST"          # 👊
        
        return "UNKNOWN"
    
    def draw_ui(self, image):
        """Draw beautiful, organized UI elements"""
        h, w = image.shape[:2]
        
        # 🎨 Draw color palette (top right)
        for color_box in self.colors:
            # Color box with shadow effect
            cv2.rectangle(image, 
                         (color_box["x"] + 2, color_box["y"] + 2), 
                         (color_box["x"] + color_box["w"] + 2, color_box["y"] + color_box["h"] + 2), 
                         (50, 50, 50), -1)
            cv2.rectangle(image, 
                         (color_box["x"], color_box["y"]), 
                         (color_box["x"] + color_box["w"], color_box["y"] + color_box["h"]), 
                         color_box["color"], -1)
            
            # Highlight current color
            if tuple(self.current_color) == tuple(color_box["color"]):
                cv2.rectangle(image, 
                             (color_box["x"] - 3, color_box["y"] - 3), 
                             (color_box["x"] + color_box["w"] + 3, color_box["y"] + color_box["h"] + 3), 
                             (255, 255, 255), 3)
        
        # 🎛️ Draw control buttons (bottom center)
        for button in self.buttons:
            # Button with shadow
            cv2.rectangle(image, 
                         (button["x"] + 3, button["y"] + 3), 
                         (button["x"] + button["w"] + 3, button["y"] + button["h"] + 3), 
                         (30, 30, 30), -1)
            cv2.rectangle(image, 
                         (button["x"], button["y"]), 
                         (button["x"] + button["w"], button["y"] + button["h"]), 
                         button["color"], -1)
            
            # Button border
            cv2.rectangle(image, 
                         (button["x"], button["y"]), 
                         (button["x"] + button["w"], button["y"] + button["h"]), 
                         (255, 255, 255), 2)
            
            # Button text (centered)
            text_size = cv2.getTextSize(button["name"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = button["x"] + (button["w"] - text_size[0]) // 2
            text_y = button["y"] + (button["h"] + text_size[1]) // 2
            cv2.putText(image, button["name"], (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 📊 Draw volume indicator (left side)
        cv2.rectangle(image, (30, 150), (60, 350), (80, 80, 80), -1)
        cv2.rectangle(image, (30, 350 - self.volume_level * 2), (60, 350), (0, 200, 255), -1)
        cv2.putText(image, f"VOL: {self.volume_level}%", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def draw_status(self, image):
        """Draw beautiful status information"""
        # Main status (top center)
        cv2.putText(image, self.current_action, (400, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Gesture info (top left)
        cv2.putText(image, f"Gesture: {self.current_gesture}", (30, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save message (temporary)
        if self.save_message_timer > 0:
            cv2.putText(image, self.save_message, (30, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.save_message_timer -= 1
        
        # Save counter
        cv2.putText(image, f"Saved: {self.save_count}", (30, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Instructions (bottom left)
        instructions = [
            "GESTURE CONTROLS:",
            "👆 POINT - Click buttons & colors",
            "🤏 PINCH - Draw on canvas",
            "✌️ TWO - Volume control",
            "✋ OPEN - Clear canvas",
            "👊 FIST - Close app"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(image, instruction, (30, 450 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    
    def process_hand_interactions(self, hand_landmarks, hand_type, image):
        """Process interactions for a single hand"""
        try:
            # Get finger positions
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            middle_tip = hand_landmarks.landmark[12]
            
            h, w, _ = image.shape
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)
            middle_x = int(middle_tip.x * w)
            middle_y = int(middle_tip.y * h)
            
            # Detect and smooth gesture
            raw_gesture = self.detect_gesture(hand_landmarks)
            gesture = self.smooth_gesture(raw_gesture)
            self.current_gesture = gesture
            
            # 🚪 CLOSE APP GESTURE
            if gesture == "CLOSE_APP":
                self.current_action = "Closing app..."
                cv2.putText(image, "CLOSING APP...", (w//2 - 150, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                self.running = False
                return gesture
            
            # Gesture color mapping
            gesture_colors = {
                "FIST": (0, 0, 255), "POINT": (0, 255, 0),
                "TWO_FINGERS": (255, 255, 0), "I_LOVE_YOU": (255, 0, 255),
                "PINCH": (255, 165, 0), "OPEN_HAND": (0, 255, 255),
                "CLOSE_APP": (255, 0, 0)
            }
            
            circle_color = gesture_colors.get(gesture, (255, 255, 255))
            
            # Draw fingertip with gesture color
            cv2.circle(image, (index_x, index_y), 10, circle_color, -1)
            cv2.circle(image, (index_x, index_y), 10, (255, 255, 255), 2)
            
            # Handle gestures
            if gesture == "PINCH":
                # Drawing mode
                cv2.circle(self.canvas, (index_x, index_y), 6, self.current_color, -1)
                self.current_action = "Drawing..."
            
            elif gesture == "POINT":
                # UI interactions
                button_interacted = False
                
                # Check buttons
                for button in self.buttons:
                    if (button["x"] < index_x < button["x"] + button["w"] and 
                        button["y"] < index_y < button["y"] + button["h"]):
                        self.current_action = f"{button['action'].title()}"
                        # Highlight button
                        cv2.rectangle(image, 
                                     (button["x"] - 5, button["y"] - 5), 
                                     (button["x"] + button["w"] + 5, button["y"] + button["h"] + 5), 
                                     (255, 255, 0), 4)
                        
                        if button["action"] == "reset":
                            self.reset_canvas()
                        elif button["action"] == "save":
                            self.save_drawing()
                        button_interacted = True
                        break
                
                # Color selection (if no button was clicked)
                if not button_interacted:
                    for color_box in self.colors:
                        if (color_box["x"] < index_x < color_box["x"] + color_box["w"] and 
                            color_box["y"] < index_y < color_box["y"] + color_box["h"]):
                            self.current_color = color_box["color"]
                            self.current_action = f"Color: {color_box['name']}"
                            break
            
            elif gesture == "TWO_FINGERS":
                # Volume control
                if 30 < middle_x < 60 and 150 < middle_y < 350:
                    self.volume_level = (350 - middle_y) // 2
                    self.volume_level = max(0, min(100, self.volume_level))
                    self.current_action = f"Volume: {self.volume_level}%"
            
            elif gesture == "OPEN_HAND":
                # Clear canvas
                self.reset_canvas()
                self.current_action = "Canvas cleared!"
            
            elif gesture == "I_LOVE_YOU":
                self.current_action = "I Love You! 💖"
                cv2.putText(image, "I LOVE YOU!", (w//2 - 120, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
            
            # Draw hand label
            cv2.putText(image, f"{hand_type}: {gesture}", 
                       (index_x + 15, index_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 2)
            
            return gesture
            
        except Exception as e:
            print(f"Error in hand processing: {e}")
            return "ERROR"
    
    def run(self):
        """Main application loop"""
        print("🚀 AR Hand Controller - Stable Edition")
        print("=====================================")
        print("🎯 Controls:")
        print("  👆 POINT    - Click buttons & select colors")
        print("  🤏 PINCH    - Draw on canvas")  
        print("  ✌️ TWO      - Control volume")
        print("  ✋ OPEN      - Clear canvas")
        print("  👊 FIST     - Close app (all fingers closed)")
        print("  Q - Quit | R - Reset | S - Save")
        print("=====================================")
        
        try:
            while self.cap.isOpened() and self.running:
                success, image = self.cap.read()
                if not success:
                    print("Failed to read camera frame")
                    continue
                
                # Flip image for mirror effect
                image = cv2.flip(image, 1)
                h, w = image.shape[:2]
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process hand detection
                results = self.hands.process(image_rgb)
                
                # Draw beautiful UI elements
                self.draw_ui(image)
                
                # Process detected hands
                if results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Determine hand type
                        hand_type = "Unknown"
                        if results.multi_handedness:
                            hand_type = results.multi_handedness[hand_idx].classification[0].label
                        
                        # Process hand interactions
                        gesture = self.process_hand_interactions(hand_landmarks, hand_type, image)
                        
                        # If close app gesture detected, break immediately
                        if gesture == "CLOSE_APP":
                            break
                        
                        # Draw hand landmarks (beautiful)
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                
                # Blend drawing canvas with camera feed
                image = cv2.addWeighted(image, 0.8, self.canvas, 0.2, 0)
                
                # Draw beautiful status information
                self.draw_status(image)
                
                # App title
                cv2.putText(image, "AR HAND CONTROLLER", (w//2 - 180, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                # Display the image
                cv2.imshow('AR Hand Controller - Stable Edition', image)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_canvas()
                elif key == ord('s'):
                    self.save_drawing()
                
                # Check if we should exit
                if not self.running:
                    break
        
        except Exception as e:
            print(f"Application error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            print("👋 Application closed gracefully")

def main():
    """Main function to run the application"""
    try:
        app = ARHandController()
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your camera is connected and accessible")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()