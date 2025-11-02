import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Set page config
st.set_page_config(
    page_title="Cyber AR Hand Controller",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cyber theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cyber-card {
        background: rgba(0, 0, 0, 0.8);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .status-indicator {
        color: #00ff00;
        font-family: 'Courier New', monospace;
    }
    .gesture-display {
        font-size: 1.5rem;
        color: #ff00ff;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class HandGestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        self.current_gesture = "SCANNING..."
        self.current_action = "Ready"
        self.canvas = None
        self.current_color = (0, 255, 255)  # Cyan
        
        # Cyber colors
        self.cyber_blue = (255, 200, 0)
        self.cyber_purple = (255, 0, 255)
        self.cyber_green = (0, 255, 0)
        self.cyber_red = (0, 0, 255)
        self.cyber_yellow = (0, 255, 255)

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def detect_gesture(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        thumb_index = self.calculate_distance(thumb_tip, index_tip)
        index_middle = self.calculate_distance(index_tip, middle_tip)
        
        if thumb_index < 0.05:
            return "DATA_INPUT"
        elif thumb_index > 0.15 and index_middle < 0.1:
            return "INTERFACE_SELECT"
        elif thumb_index < 0.1 and index_middle > 0.15:
            return "SYSTEM_CONTROL"
        elif thumb_index > 0.15 and self.calculate_distance(ring_tip, pinky_tip) > 0.15:
            return "SPECIAL_COMMAND"
        
        return "SCANNING"

    def draw_cyber_ui(self, image):
        h, w = image.shape[:2]
        
        # Draw cyber border
        cv2.rectangle(image, (10, 10), (w-10, h-10), self.cyber_blue, 2)
        
        # Draw corner brackets
        bracket_size = 15
        corners = [(10, 10), (w-10, 10), (10, h-10), (w-10, h-10)]
        for x, y in corners:
            cv2.line(image, (x, y), (x+bracket_size, y), self.cyber_green, 2)
            cv2.line(image, (x, y), (x, y+bracket_size), self.cyber_green, 2)
        
        # Draw status text
        cv2.putText(image, f"GESTURE: {self.current_gesture}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.cyber_purple, 2)
        cv2.putText(image, f"ACTION: {self.current_action}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.cyber_yellow, 2)
        
        # Draw color palette
        colors = [self.cyber_blue, self.cyber_purple, self.cyber_green, self.cyber_red, self.cyber_yellow]
        for i, color in enumerate(colors):
            x = w - 60
            y = 100 + i * 40
            cv2.rectangle(image, (x, y), (x+40, y+30), color, -1)
            if tuple(self.current_color) == tuple(color):
                cv2.rectangle(image, (x-2, y-2), (x+42, y+32), (255, 255, 255), 2)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # Initialize canvas
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Flip for mirror effect
        img = cv2.flip(img, 1)
        
        # Process hand detection
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)
        
        # Draw cyber UI
        self.draw_cyber_ui(img)
        
        # Process hand gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Detect gesture
                gesture = self.detect_gesture(hand_landmarks)
                self.current_gesture = gesture
                
                # Get finger positions
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                
                index_x = int(index_tip.x * w)
                index_y = int(index_tip.y * h)
                
                # Handle gestures
                if gesture == "DATA_INPUT":
                    cv2.circle(self.canvas, (index_x, index_y), 5, self.current_color, -1)
                    self.current_action = "DRAWING"
                elif gesture == "INTERFACE_SELECT":
                    self.current_action = "SELECTING"
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=self.cyber_blue, thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=self.cyber_purple, thickness=2)
                )
                
                # Draw fingertip
                cv2.circle(img, (index_x, index_y), 10, self.cyber_green, -1)
        
        # Blend canvas with video
        img = cv2.addWeighted(img, 0.8, self.canvas, 0.2, 0)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 CYBERNETIC AR HAND CONTROLLER</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎮 CONTROL PANEL")
        
        # Color selection
        st.markdown("#### 🎨 COLOR SELECTION")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔵 CYAN", use_container_width=True):
                st.session_state.current_color = (255, 200, 0)
        with col2:
            if st.button("🟣 PURPLE", use_container_width=True):
                st.session_state.current_color = (255, 0, 255)
        with col3:
            if st.button("🟢 GREEN", use_container_width=True):
                st.session_state.current_color = (0, 255, 0)
        
        col4, col5 = st.columns(2)
        with col4:
            if st.button("🔴 RED", use_container_width=True):
                st.session_state.current_color = (0, 0, 255)
        with col5:
            if st.button("🟡 YELLOW", use_container_width=True):
                st.session_state.current_color = (0, 255, 255)
        
        # Actions
        st.markdown("#### ⚡ ACTIONS")
        if st.button("🧹 CLEAR CANVAS", use_container_width=True):
            st.session_state.clear_canvas = True
        
        if st.button("💾 SAVE DRAWING", use_container_width=True):
            st.session_state.save_drawing = True
        
        # System info
        st.markdown("#### 📊 SYSTEM STATUS")
        st.markdown('<p class="status-indicator">🟢 SYSTEM: ONLINE</p>', unsafe_allow_html=True)
        st.progress(75, text="🖥️ CPU USAGE: 75%")
        st.progress(60, text="🔋 ENERGY: 60%")
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎥 LIVE FEED")
        
        # WebRTC streamer
        ctx = webrtc_streamer(
            key="hand-gesture",
            video_processor_factory=HandGestureProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if ctx.video_processor:
            # Display current status
            st.markdown(f'<div class="cyber-card"><p class="gesture-display">Current Gesture: {ctx.video_processor.current_gesture}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cyber-card"><p class="gesture-display">Action: {ctx.video_processor.current_action}</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 GESTURE GUIDE")
        
        gestures = {
            "🤏 PINCH": "Draw on the canvas",
            "👆 POINT": "Select interface elements", 
            "✌️ TWO FINGERS": "System controls",
            "🤟 SPECIAL": "Activate commands"
        }
        
        for gesture, description in gestures.items():
            with st.container():
                st.markdown(f"**{gesture}**")
                st.caption(description)
                st.divider()
        
        st.markdown("### 💡 TIPS")
        st.info("• Ensure good lighting for better hand detection")
        st.info("• Keep your hand within the camera frame")
        st.info("• Make clear, deliberate gestures")
        st.info("• Use a plain background for best results")

    # Footer
    st.markdown("---")
    st.markdown("### 🚀 ENHANCEMENTS ROADMAP")
    
    features = [
        "🎭 Face detection and emotion recognition",
        "🗣️ Voice command integration", 
        "🌌 Virtual background effects",
        "📷 Photo capture and gallery",
        "🎵 Sound effects and audio feedback",
        "🔢 Number gesture recognition (1-5 fingers)",
        "🔄 Gesture macros and automation"
    ]
    
    for feature in features:
        st.write(f"• {feature}")

if __name__ == "__main__":
    # Initialize session state
    if 'current_color' not in st.session_state:
        st.session_state.current_color = (255, 200, 0)
    if 'clear_canvas' not in st.session_state:
        st.session_state.clear_canvas = False
    if 'save_drawing' not in st.session_state:
        st.session_state.save_drawing = False
    
    main()