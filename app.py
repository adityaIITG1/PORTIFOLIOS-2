import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# ============================================================
#   YOGA AI - STREAMLIT PREMIUM EDITION
# ============================================================

# Page Config
st.set_page_config(
    page_title="Yoga AI Premium",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Dark UI
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #e6edf3;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.4);
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ===================== CONSTANTS =======================

CHAKRA_NAMES = [
    "Root", "Sacral", "Solar Plexus", "Heart",
    "Throat", "Third Eye", "Crown"
]

CHAKRA_COLORS = [
    (0,   0, 255),   # Root    - Red
    (0, 140, 255),   # Sacral  - Orange
    (0, 255, 255),   # Solar   - Yellow
    (0, 255,   0),   # Heart   - Green
    (255, 0,   0),   # Throat  - Blue (BGR)
    (255, 0, 255),   # Third   - Violet
    (255, 255, 255)  # Crown   - White
]

MUDRA_INFO = {
    "Gyan": [
        "GYAN MUDRA (Wisdom)", "Benefits:", "- Improves concentration", "- Sharpens memory", "- Reduces stress"
    ],
    "Prana": [
        "PRANA MUDRA (Vitality)", "Benefits:", "- Boosts energy", "- Improves vision", "- Activates root chakra"
    ],
    "Apana": [
        "APANA MUDRA (Detox)", "Benefits:", "- Detoxifies body", "- Improves digestion", "- Inner balance"
    ],
    "Surya": [
        "SURYA MUDRA (Fire)", "Benefits:", "- Boosts metabolism", "- Generates heat", "- Weight loss aid"
    ],
    "Varun": [
        "VARUN MUDRA (Water)", "Benefits:", "- Hydrates skin", "- Balances fluids", "- Improves circulation"
    ],
    "Anjali": [
        "ANJALI MUDRA (Prayer)", "Benefits:", "- Inner peace", "- Brain balance", "- Gratitude"
    ]
}

# ===================== HELPER FUNCTIONS =======================

def draw_text_with_bg(frame, text, x, y, font_scale=0.6, color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), bg_alpha=0.6):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_universe(frame, t):
    h, w, _ = frame.shape
    cx = int(w * 0.15) # Left side
    cy = int(h * 0.5)

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), 140, (40, 0, 60), -1)
    cv2.circle(overlay, (cx, cy), 90, (80, 0, 120), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    cv2.circle(frame, (cx, cy), 26, (0, 255, 255), -1)

    orbit_radii = [55, 90, 125]
    speeds = [0.9, 0.6, 0.35]
    colors = [(255, 200, 0), (255, 255, 255), (0, 215, 255)]

    for r, spd, col in zip(orbit_radii, speeds, colors):
        angle = t * spd
        px = int(cx + r * math.cos(angle))
        py = int(cy + r * math.sin(angle))
        cv2.circle(frame, (px, py), 10, col, -1)
        cv2.ellipse(frame, (cx, cy), (r, r), 0, 0, 360, (90, 90, 120), 1)

def draw_chakras(frame, center_x, top_y, bottom_y, active_index, energies, aura_color, breath_factor, t):
    num_chakras = 7
    ys = np.linspace(bottom_y, top_y, num_chakras)
    music_pulse = 0.8 + 0.35 * math.sin(2.0 * t)

    for i in range(num_chakras):
        chakra_name = CHAKRA_NAMES[i]
        base_color = CHAKRA_COLORS[i]
        energy = energies[i]

        wobble = 8 * math.sin(t * 1.4 + i * 0.9)
        cy = int(ys[i] + wobble)
        base_radius = 18 + int(energy * 22)
        radius = int(base_radius * breath_factor * music_pulse)
        center = (center_x, cy)

        aura_radius = int(radius * (1.5 + 0.3 * music_pulse))
        aura_alpha = min(0.9, 0.25 + 0.5 * energy * music_pulse)

        overlay = frame.copy()
        cv2.circle(overlay, center, aura_radius, aura_color, -1)
        cv2.addWeighted(overlay, aura_alpha, frame, 1 - aura_alpha, 0, frame)

        cv2.circle(frame, center, radius, base_color, -1)

        orbit_r = int(radius * 1.6)
        dot_angle = t * 2.5 + i
        dot_x = int(center[0] + orbit_r * math.cos(dot_angle))
        dot_y = int(center[1] + orbit_r * math.sin(dot_angle))
        cv2.circle(frame, (dot_x, dot_y), 4, (255, 255, 255), -1)

        if i == active_index:
            cv2.circle(frame, center, radius + 6, (255, 255, 255), 2)

        cv2.putText(frame, chakra_name, (center[0] + 30, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def draw_revolving_aura(frame, center_x, center_y, radius, t):
    golden_color = (0, 215, 255)
    overlay = frame.copy()
    cv2.circle(overlay, (center_x, center_y), radius, golden_color, -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    num_particles = 8
    for i in range(num_particles):
        angle = t * 2.0 + (i * (2 * math.pi / num_particles))
        orbit_rx = radius * 1.2
        orbit_ry = radius * 0.4
        px = int(center_x + orbit_rx * math.cos(angle))
        py = int(center_y + orbit_ry * math.sin(angle) - radius * 0.5)
        cv2.circle(frame, (px, py), 6, (255, 255, 255), -1)
        cv2.circle(frame, (px, py), 10, golden_color, 2)

def draw_gyan_sparkles(frame, center_x, center_y, radius):
    overlay = frame.copy()
    for _ in range(35):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(radius * 0.6, radius * 1.1)
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle))
        cv2.circle(overlay, (x, y), random.randint(2, 4), (0, 215, 255), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

def draw_mini_hand(frame, cx, cy, mudra_name, scale=1.0):
    colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (0, 140, 255), (255, 0, 255)]
    wrist_color = (255, 255, 255)
    s = 25 * scale
    pts = {
        'wrist': (0, s*1.5),
        'thumb_base': (-s*0.6, s*0.8), 'thumb_tip': (-s*1.2, -s*0.2),
        'index_base': (-s*0.3, -s*0.5), 'index_tip': (-s*0.4, -s*1.8),
        'mid_base': (0, -s*0.6),      'mid_tip': (0, -s*2.0),
        'ring_base': (s*0.3, -s*0.5),   'ring_tip': (s*0.4, -s*1.8),
        'pinky_base': (s*0.5, -s*0.3),  'pinky_tip': (s*0.7, -s*1.4)
    }
    
    if mudra_name == "Gyan":
        pts['index_tip'] = (-s*0.8, -s*0.5); pts['thumb_tip'] = (-s*0.8, -s*0.5)
    elif mudra_name == "Prana":
        pts['ring_tip'] = (-s*0.5, s*0.2); pts['pinky_tip'] = (-s*0.5, s*0.2); pts['thumb_tip'] = (-s*0.5, s*0.2)
    elif mudra_name == "Apana":
        pts['mid_tip'] = (-s*0.5, s*0.2); pts['ring_tip'] = (-s*0.5, s*0.2); pts['thumb_tip'] = (-s*0.5, s*0.2)
    elif mudra_name == "Surya":
        pts['ring_tip'] = pts['thumb_base']; pts['thumb_tip'] = pts['thumb_base']
    elif mudra_name == "Varun":
        pts['pinky_tip'] = (-s*0.6, 0); pts['thumb_tip'] = (-s*0.6, 0)
    elif mudra_name == "Anjali":
        pts['thumb_tip'] = (-s*0.2, -s*0.5); pts['index_tip'] = (0, -s*1.8)
        pts['mid_tip'] = (s*0.1, -s*1.9); pts['ring_tip'] = (s*0.2, -s*1.8); pts['pinky_tip'] = (s*0.3, -s*1.6)

    def dline(p1_name, p2_name, color):
        p1 = (int(cx + pts[p1_name][0]), int(cy + pts[p1_name][1]))
        p2 = (int(cx + pts[p2_name][0]), int(cy + pts[p2_name][1]))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(frame, p2, 3, color, -1, cv2.LINE_AA)

    dline('wrist', 'thumb_base', wrist_color); dline('thumb_base', 'thumb_tip', colors[0])
    dline('wrist', 'index_base', wrist_color); dline('index_base', 'index_tip', colors[1])
    dline('wrist', 'mid_base', wrist_color); dline('mid_base', 'mid_tip', colors[2])
    dline('wrist', 'ring_base', wrist_color); dline('ring_base', 'ring_tip', colors[3])
    dline('wrist', 'pinky_base', wrist_color); dline('pinky_base', 'pinky_tip', colors[4])

def draw_mudra_info_panel(frame, mudra_name=None):
    h, w, _ = frame.shape
    sidebar_w = 280
    panel_w = sidebar_w - 20
    panel_h = 140
    x = w - sidebar_w + 10
    y = 570 
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (40, 50, 60), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (0, 255, 0), 2)
    
    lines = MUDRA_INFO.get(mudra_name, ["CHAKRA AI FLOW", "Guide:", "- Sit in **Lotus Pose**", "- Show **Hand Mudras**", "- Close **Eyes**", "- Focus on **Breath**"])
    title_color = (0, 255, 0) if mudra_name else (0, 255, 255)
    
    cv2.putText(frame, lines[0], (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2, cv2.LINE_AA)
    py = y + 60
    for line in lines[1:]:
        col = (200, 255, 200) if line.startswith("Benefits:") or line.startswith("Guide:") else (255, 255, 255)
        scale = 0.5 if line.startswith("Benefits:") or line.startswith("Guide:") else 0.45
        
        parts = line.split('**')
        px = x + 15
        for i, part in enumerate(parts):
            curr_col = (0, 255, 255) if i % 2 == 1 else col
            cv2.putText(frame, part, (px, py), cv2.FONT_HERSHEY_SIMPLEX, scale, curr_col, 1, cv2.LINE_AA)
            (txt_w, _), _ = cv2.getTextSize(part, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            px += txt_w
        py += 25

def draw_mudra_sidebar(frame, active_mudra):
    h, w, _ = frame.shape
    sidebar_w = 280 
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (60, 70, 80), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (w - sidebar_w, 0), (w - sidebar_w, h), (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, "Mudra Guide", (w - sidebar_w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    mudras = [("Gyan", "Wisdom"), ("Prana", "Vitality"), ("Apana", "Detox"), ("Surya", "Fire/Wt"), ("Varun", "Water"), ("Anjali", "Prayer")]
    y = 90
    for name, desc in mudras:
        is_active = (active_mudra == name + " Mudra") if active_mudra else False
        bg_col = (50, 200, 50) if is_active else (255, 255, 255)
        bg_alpha = 0.6 if is_active else 0.15
        
        if is_active:
            cv2.line(frame, (w - sidebar_w, y - 25), (w - sidebar_w, y + 45), (0, 255, 0), 5, cv2.LINE_AA)
            
        overlay_item = frame.copy()
        cv2.rectangle(overlay_item, (w - sidebar_w + 2, y - 25), (w, y + 45), bg_col, -1)
        cv2.addWeighted(overlay_item, bg_alpha, frame, 1 - bg_alpha, 0, frame)
        
        cv2.putText(frame, name, (w - sidebar_w + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (w - sidebar_w + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220) if not is_active else (255, 255, 255), 1, cv2.LINE_AA)
        draw_mini_hand(frame, w - 60, y + 10, name, scale=0.8)
        y += 80
        
    draw_mudra_info_panel(frame, active_mudra.split()[0] if active_mudra else None)

class PostureAnalyzer:
    def assess(self, pose_landmarks):
        if not pose_landmarks: return 0.0, "No body"
        lm = pose_landmarks.landmark
        ls, rs = lm[11], lm[12]
        lh, rh = lm[23], lm[24]
        dx = ((ls.x+rs.x)*0.5) - ((lh.x+rh.x)*0.5)
        dy = ((ls.y+rs.y)*0.5) - ((lh.y+rh.y)*0.5) + 1e-6
        spine_angle = abs(math.degrees(math.atan2(dx, dy)))
        shoulder_level = abs(ls.y - rs.y)
        score = 1.0
        if spine_angle > 10: score -= min(0.5, (spine_angle - 10) / 40)
        if shoulder_level > 0.03: score -= min(0.3, (shoulder_level - 0.03) / 0.1)
        score = max(0.0, min(1.0, score))
        if score > 0.8: label = "Aligned"
        elif score > 0.6: label = "Slight tilt"
        else: label = "Poor posture"
        return score, label

# ===================== VIDEO PROCESSOR =======================

class YogaProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.chakra_energies = [0.4] * 7
        self.anim_time = 0.0
        self.last_frame_time = time.time()
        self.posture_analyzer = PostureAnalyzer()
        self.alignment_mode = False
        self.alignment_progress = 0.0
        self.alignment_start_time = 0
        self.eye_closed_frames = 0
        self.was_eyes_closed = False
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process
        hand_res = self.mp_hands.process(rgb)
        face_res = self.mp_face.process(rgb)
        pose_res = self.mp_pose.process(rgb)
        
        # Time Delta
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        
        # --- Logic ---
        center_x = w // 2
        top_y = int(h * 0.25)
        bottom_y = int(h * 0.85)
        
        # Face & Meditation
        eye_open = 1.0
        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0].landmark
            left_eye_top = lm[159]
            left_eye_bottom = lm[145]
            eye_open = math.sqrt((left_eye_top.x - left_eye_bottom.x)**2 + (left_eye_top.y - left_eye_bottom.y)**2)
            center_x = int(lm[1].x * w)
            
            if eye_open < 0.022:
                self.eye_closed_frames += 1
                if self.eye_closed_frames > 15 and not self.alignment_mode:
                    self.alignment_mode = True
                    self.alignment_start_time = time.time()
            else:
                self.eye_closed_frames = 0
                
        if self.alignment_mode:
            self.alignment_progress = min(1.0, self.alignment_progress + 0.01)
            target = 1.0 * self.alignment_progress
            for i in range(7): self.chakra_energies[i] = max(self.chakra_energies[i], target)
            if time.time() - self.alignment_start_time > 8: self.alignment_mode = False
            
        # Pose
        posture_score = 0.0
        if pose_res.pose_landmarks:
            posture_score, _ = self.posture_analyzer.assess(pose_res.pose_landmarks)
            
        # Hands
        detected_mudra = None
        detected_mudra_name = None
        gyan_active = False
        
        if hand_res.multi_hand_landmarks:
            # Simple Namaste Check
            if len(hand_res.multi_hand_landmarks) == 2:
                h1 = hand_res.multi_hand_landmarks[0].landmark[0]
                h2 = hand_res.multi_hand_landmarks[1].landmark[0]
                if math.sqrt((h1.x-h2.x)**2 + (h1.y-h2.y)**2) < 0.2:
                    detected_mudra_name = "Anjali Mudra"
                    detected_mudra = 4
            
            # Single Hand Mudras (Simplified logic for brevity)
            if not detected_mudra_name:
                for hl in hand_res.multi_hand_landmarks:
                    lm = hl.landmark
                    # Gyan: Index tip near Thumb tip
                    if math.sqrt((lm[4].x-lm[8].x)**2 + (lm[4].y-lm[8].y)**2) < 0.05:
                        detected_mudra_name = "Gyan Mudra"; detected_mudra = 6; gyan_active = True; break
        
        # Energy Update
        if detected_mudra is not None:
            for i in range(7):
                if i == detected_mudra: self.chakra_energies[i] = min(1.0, self.chakra_energies[i] + 0.02)
                else: self.chakra_energies[i] = max(0.1, self.chakra_energies[i] - 0.004)
        elif not self.alignment_mode:
             for i in range(7): self.chakra_energies[i] = max(0.05, self.chakra_energies[i] - 0.015)
             
        # --- Drawing ---
        is_yoga_active = (posture_score > 0.1) or gyan_active
        speed_multiplier = 3.0 if self.alignment_mode else (2.0 if is_yoga_active else 1.0)
        self.anim_time += dt * speed_multiplier
        
        draw_universe(img, self.anim_time)
        
        # Aura
        center_y_aura = int(h * 0.55)
        aura_radius = int(min(w, h) * 0.4)
        aura_color = (0, 215, 255) if (is_yoga_active and not self.alignment_mode) else (255, 255, 255)
        
        overlay_bg = img.copy()
        cv2.circle(overlay_bg, (center_x, center_y_aura), aura_radius, aura_color, -1)
        cv2.addWeighted(overlay_bg, 0.15, img, 0.85, 0, img)
        
        if is_yoga_active and not self.alignment_mode:
            draw_revolving_aura(img, center_x, center_y_aura - 100, 120, self.anim_time)
            
        if gyan_active and not self.alignment_mode:
            draw_gyan_sparkles(img, center_x, center_y_aura, aura_radius)
            
        draw_chakras(img, center_x, top_y, bottom_y, detected_mudra, self.chakra_energies, aura_color, 1.0, self.anim_time)
        
        # Hands & Face
        if hand_res.multi_hand_landmarks:
             for hl in hand_res.multi_hand_landmarks:
                 self.mp_drawing.draw_landmarks(img, hl, self.mp_hands.HAND_CONNECTIONS)
                 
        draw_mudra_sidebar(img, detected_mudra_name)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== MAIN UI =======================

st.title("🧘 Yoga AI Premium")
st.write("Experience the power of AI-guided meditation and chakra balancing.")

col1, col2 = st.columns([3, 1])

with col1:
    webrtc_streamer(
        key="yoga-ai",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=YogaProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### 🛠️ Settings")
    st.slider("Sensitivity", 0.0, 1.0, 0.5)
    st.markdown("### 📊 Live Stats")
    st.metric("Session Time", "00:00")
    st.info("Align your body and show mudras to begin.")

