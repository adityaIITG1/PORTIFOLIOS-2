import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pygame
import os
import random
import speech_recognition as sr
import pyttsx3
import serial
import serial.tools.list_ports
import ai_explainer

# ============================================================
#   AI CHAKRAFLOW — FULL VERSION (MUSIC + VOICE + SUMMARY)
#   BILINGUAL (Hindi + English)
# ============================================================

CAM_INDEX = 0
# Slightly larger 16:9 frame; still modest to keep lag low.
FRAME_WIDTH = 1120
FRAME_HEIGHT = 630

# ---- Path to your Adiyogi music ----
MUSIC_PATH = r"C:\Users\ASUS\OneDrive\Desktop\Yoga_AI\Adiyogi The Source of Yog-320kbps.mp3"

# Chakra definitions (bottom to top)
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

# Short scripture-like snippets per chakra for AI explainer (safe, brief)
CHAKRA_SCRIPTURES = [
    {
        "id": "root_balance",
        "source": "Yoga wisdom",
        "sanskrit": "Sthiram sukham asanam",
        "hinglish": "Stay steady like a mountain",
        "meaning": "Ground yourself and find steadiness."
    },
    {
        "id": "sacral_flow",
        "source": "Yoga wisdom",
        "sanskrit": "Jala tattva",
        "hinglish": "Gentle flow, soft breath",
        "meaning": "Let movement be smooth and creative."
    },
    {
        "id": "solar_fire",
        "source": "Yoga wisdom",
        "sanskrit": "Tejas",
        "hinglish": "Inner fire with calm mind",
        "meaning": "Strength with kindness—no force."
    },
    {
        "id": "heart_compassion",
        "source": "Yoga wisdom",
        "sanskrit": "Anahata",
        "hinglish": "Open heart, light shoulders",
        "meaning": "Balance effort with softness and care."
    },
    {
        "id": "throat_truth",
        "source": "Yoga wisdom",
        "sanskrit": "Satya",
        "hinglish": "Speak softly, breathe freely",
        "meaning": "Align breath and voice with honesty."
    },
    {
        "id": "third_eye_focus",
        "source": "Yoga wisdom",
        "sanskrit": "Dhyana",
        "hinglish": "Drishti shant rakho",
        "meaning": "Calm gaze, clear mind, steady breath."
    },
    {
        "id": "crown_stillness",
        "source": "Yoga wisdom",
        "sanskrit": "Shanti",
        "hinglish": "Sukoon se baitho",
        "meaning": "Sit in quiet awareness; no hurry, no pressure."
    },
]
# Thresholds
# Thresholds
EYE_CLOSED_THRESHOLD = 0.30 # EAR Ratio (Increased to 0.30 for very robust detection)
EYE_CLOSED_FRAMES_REQUIRED = 15 # ~0.5-1 sec.5s

AI_REFRESH_SECS = 6  # refresh AI tip every few seconds

# ---------------- Pygame audio init -----------------
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# ---------------- TTS (Indian-ish female voice) -----------------
tts = pyttsx3.init()
voices = tts.getProperty('voices')
for v in voices:
    name_lower = v.name.lower()
    if "female" in name_lower or "zira" in name_lower or "hindi" in v.id.lower():
        tts.setProperty('voice', v.id)
        break
tts.setProperty('rate', 150) # Slower for better clarity
tts.setProperty('volume', 1.0)

# ---------------- Mouse State -----------------
mouse_x, mouse_y = 0, 0
last_speak_time = 0
current_speaking_graph = None

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

import threading
import queue

# [FIX] TTS Worker to prevent freezing
# Uses a dedicated thread and queue for speech
class TTSWorker:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def _run(self):
        # Initialize engine ONCE in the worker thread
        try:
            engine = pyttsx3.init()
            # Voice Settings
            voices = engine.getProperty('voices')
            for v in voices:
                name_lower = v.name.lower()
                if "female" in name_lower or "zira" in name_lower or "hindi" in v.id.lower():
                    engine.setProperty('voice', v.id)
                    break
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            while True:
                text = self.queue.get()
                if text is None: break # Sentinel to stop
                
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[TTS Error] {e}")
                    
                self.queue.task_done()
        except Exception as e:
            print(f"[TTS Init Error] {e}")

# Initialize Global TTS Worker
tts_worker = TTSWorker()

def speak_threaded(text):
    """Adds text to the TTS queue."""
    tts_worker.queue.put(text)

def check_hover_and_speak(w, h):
    global last_speak_time, current_speaking_graph
    
    # [FIX] Debounce logic
    # Allow re-speaking if enough time has passed (e.g., 5 seconds)
    # OR if hovering a NEW graph
    
    # Panel Coordinates (UPDATED to match draw_heart_rate_panel)
    panel_w = 350
    panel_h = 650
    panel_x = 140 
    panel_y = 20
    
    gy = panel_y + 145
    gh = 45
    gap = 8
    
    # Regions
    regions = {
        "Heart Rhythm": (panel_x + 20, gy, panel_w - 40, gh, 
                         "Yeh Heart Rhythm graph hai. Yeh aapke dil ki dhadkan ki gati aur lay ko dikhata hai. Agar yeh graph smooth hai, toh aapka dil swasth hai aur aap relax hain."),
        "Stress Level": (panel_x + 20, gy + gh + gap, panel_w - 40, gh, 
                         "Yeh Stress Level graph hai. Yeh batata hai ki aap kitne tanav mein hain. Isse neeche rakhne ke liye gehri saans lein aur relax karein."),
        "Prana Energy": (panel_x + 20, gy + 2*(gh + gap), panel_w - 40, gh, 
                         "Yeh Prana Energy graph hai. Yeh aapki andar ki urja ko mapta hai. Yoga aur Pranayam se yeh badhta hai aur aapko shaktishali banata hai."),
        "Focus Level": (panel_x + 20, gy + 3*(gh + gap), panel_w - 40, gh, 
                        "Yeh Focus Level graph hai. Yeh dikhata hai ki aapka dhyan kitna kendrit hai. Jitna zyaada focus, utna behtar result."),
        "HRV Index": (panel_x + 20, gy + 4*(gh + gap), panel_w - 40, gh, 
                      "Yeh HRV Index hai. Yeh aapke dil ki lachilapan aur swasthya ka sabse bada sanket hai. High HRV ka matlab hai aapka dil jawan aur healthy hai."),
        "Nadi Pariksha": (panel_x + 20, gy + 5*(gh + gap) + 15, panel_w - 40, 50,
                          "Yeh Nadi Pariksha hai. Yahan hum aapke Vata, Pitta aur Kapha doshas ko track kar rahe hain. Vata movement hai, Pitta energy hai, aur Kapha stability hai."),
    }
    
    hovered_something = False

    # Check Rectangular Regions
    for name, (rx, ry, rw, rh, text) in regions.items():
        if rx < mouse_x < rx + rw and ry < mouse_y < ry + rh:
            hovered_something = True
            # Speak if:
            # 1. New graph hovered
            # 2. OR Same graph but time > 5s
            if current_speaking_graph != name or (time.time() - last_speak_time > 5.0):
                print(f"[SPEAK] {name}")
                speak_threaded(text)
                last_speak_time = time.time()
                current_speaking_graph = name
            return

    # Check Circular Region (Energy Coherence)
    tg_y = gy + 5*(gh + gap) + 15
    guide_y = tg_y + 50
    cx = panel_x + panel_w // 2
    cy = guide_y + 45
    r = 35
    
    dist = ((mouse_x - cx)**2 + (mouse_y - cy)**2)**0.5
    if dist < r:
        hovered_something = True
        if current_speaking_graph != "Energy Coherence" or (time.time() - last_speak_time > 5.0):
            print("[SPEAK] Energy Coherence")
            speak_threaded("Yeh Energy Coherence radar hai. Yeh aapki poori body aur mind ka balance dikhata hai. Jab yeh bada aur gol hota hai, iska matlab aapki energy peak par hai aur aap poori tarah se aligned hain.")
            last_speak_time = time.time()
            current_speaking_graph = "Energy Coherence"
        return
        
    # [FIX] Reset if not hovering anything
    # This allows re-triggering immediately if user leaves and comes back
    if not hovered_something:
        current_speaking_graph = None

# ---------------- Mediapipe -----------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ===================== HELPERS =======================

def draw_status_panel(frame, center_x, center_y, radius, status):
    """
    Draws a centralized status panel inside the yellow circle.
    status: dict with keys 'title', 'subtitle', 'hint', 'extra'
    """
    # Calculate vertical spacing
    # We want to center the block of text vertically within the circle
    # Title (Large) + Subtitle (Medium) + Hint (Small) + Extra (Small)
    
    lines = []
    # [FIX] Enhanced Visibility for ALL lines (Thicker font + Stronger Outline)
    if status.get('title'): lines.append({'text': status['title'], 'scale': 0.8, 'thick': 2, 'color': (0, 255, 255)})
    if status.get('subtitle'): lines.append({'text': status['subtitle'], 'scale': 0.65, 'thick': 2, 'color': (0, 255, 255)}) # Thicker
    if status.get('hint'): lines.append({'text': status['hint'], 'scale': 0.6, 'thick': 2, 'color': (0, 255, 255)}) # Thicker
    if status.get('extra'): lines.append({'text': status['extra'], 'scale': 0.5, 'thick': 1, 'color': (0, 255, 255)})
    
    if not lines: return

    # Calculate total height and max width to center it
    total_h = 0
    max_w = 0
    line_heights = []
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line['text'], cv2.FONT_HERSHEY_SIMPLEX, line['scale'], line['thick'])
        lh = h + baseline + 10 # 10px padding
        line_heights.append(lh)
        total_h += lh
        max_w = max(max_w, w)
        
    # [FIX] Move to Bottom Center
    # Start Y position: Bottom of screen minus total height minus padding
    h, w, _ = frame.shape
    current_y = h - total_h - 20 
    
    # [FIX] Removed Background Box
    # (Deleted rectangle drawing code)
    
    for i, line in enumerate(lines):
        text = line['text']
        scale = line['scale']
        thick = line['thick']
        color = line['color']
        lh = line_heights[i]
        
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = center_x - (w // 2)
        
        # Draw Shadow/Outline for better visibility
        # [FIX] Much thicker outline (4px minimum)
        cv2.putText(frame, text, (x, current_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
        # Draw Main Text
        cv2.putText(frame, text, (x, current_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        
        current_y += lh

def draw_text_with_bg(frame, text, x, y, font_scale=0.6, color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), bg_alpha=0.6):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    overlay = frame.copy()
    # Draw background rectangle
    cv2.rectangle(overlay, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_paragraph_with_bg(frame, text, x, y, max_width=600, font_scale=0.5, color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), bg_alpha=0.6):
    """
    Draws text wrapped into a paragraph with a background box.
    x, y: Bottom-left corner of the *first line* (or roughly the anchor).
    Actually, let's make x, y the center-bottom anchor for the whole block to stack upwards easily.
    """
    words = text.split()
    lines = []
    line = ""
    
    # Simple wrapping
    for w in words:
        test_line = line + (" " if line else "") + w
        (w_px, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if w_px > max_width:
            lines.append(line)
            line = w
        else:
            line = test_line
    if line:
        lines.append(line)
        
    if not lines: return

    # Calculate total height
    (txt_w, txt_h), baseline = cv2.getTextSize("Test", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    line_height = txt_h + baseline + 10 # Spacing
    total_h = line_height * len(lines)
    
    # Calculate max width of the block
    max_line_w = 0
    for l in lines:
        (lw, _), _ = cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_line_w = max(max_line_w, lw)
        
    # Anchor: (x, y) is the bottom center of the block
    # Top-left of the box
    box_x = x - max_line_w // 2 - 10
    box_y = y - total_h - 10
    box_w = max_line_w + 20
    box_h = total_h + 20
    
    # Draw Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    
    # Draw Text
    cur_y = box_y + 10 + txt_h # First line baseline
    for l in lines:
        # Center each line
        (lw, _), _ = cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lx = x - lw // 2
        cv2.putText(frame, l, (lx, cur_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        cur_y += line_height

def get_finger_states(hand_landmarks, image_width, image_height):
    lm = hand_landmarks.landmark

    def to_pixel(idx):
        return int(lm[idx].x * image_width), int(lm[idx].y * image_height)

    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    states = {}

    thumb_tip_x, thumb_tip_y = to_pixel(4)
    thumb_pip_x, thumb_pip_y = to_pixel(3)
    states["thumb"] = thumb_tip_x > thumb_pip_x

    finger_names = ["index", "middle", "ring", "pinky"]
    for name, tip_idx, pip_idx in zip(finger_names, tips[1:], pips[1:]):
        tip_x, tip_y = to_pixel(tip_idx)
        pip_x, pip_y = to_pixel(pip_idx)
        states[name] = tip_y < pip_y

    return states


def detect_gyan_mudra(hand_landmarks, frame=None, width=0, height=0):
    lm = hand_landmarks.landmark
    thumb_tip = lm[4]
    index_tip = lm[8]
    wrist = lm[0]
    mid_tip = lm[12]
    
    # Normalize by hand size (wrist to middle fingertip) for scale tolerance
    hand_scale = math.sqrt((wrist.x - mid_tip.x) ** 2 + (wrist.y - mid_tip.y) ** 2) + 1e-6
    d = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) / hand_scale
    
    # Visual Debugging (Draw line between thumb and index)
    if frame is not None:
        tx, ty = int(thumb_tip.x * width), int(thumb_tip.y * height)
        ix, iy = int(index_tip.x * width), int(index_tip.y * height)
        
        # Color code: Green=Active, Yellow=Close, Red=Far
        # Debug info
        cv2.putText(frame, f"Gyan Dist: {d:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if d < 0.28: # Further relaxed from 0.22
            col = (0, 255, 0)
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1) 
        elif d < 0.35:
            col = (0, 255, 255)
        else:
            col = (0, 0, 255)
            
        cv2.line(frame, (tx, ty), (ix, iy), col, 2)

    return d < 0.28 # Further relaxed threshold

def detect_prana_mudra(hand_landmarks):
    # Ring and Pinky tips touch Thumb tip. Index and Middle straight.
    lm = hand_landmarks.landmark
    thumb_tip = lm[4]
    ring_tip = lm[16]
    pinky_tip = lm[20]
    wrist = lm[0]
    mid_tip = lm[12]
    
    hand_scale = math.sqrt((wrist.x - mid_tip.x) ** 2 + (wrist.y - mid_tip.y) ** 2) + 1e-6
    
    d_ring = math.sqrt((thumb_tip.x - ring_tip.x)**2 + (thumb_tip.y - ring_tip.y)**2) / hand_scale
    d_pinky = math.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2) / hand_scale
    
    # Check if index and middle are extended (tip further from wrist than pip)
    # Using simple y check might fail if hand is rotated, better to check distance from wrist
    def dist_sq(p1, p2): return (p1.x-p2.x)**2 + (p1.y-p2.y)**2
    
    index_ext = dist_sq(lm[8], wrist) > dist_sq(lm[6], wrist)
    middle_ext = dist_sq(lm[12], wrist) > dist_sq(lm[10], wrist)

    index_ext = dist_sq(lm[8], wrist) > dist_sq(lm[6], wrist)
    middle_ext = dist_sq(lm[12], wrist) > dist_sq(lm[10], wrist)

    return d_ring < 0.28 and d_pinky < 0.28 and index_ext and middle_ext

def detect_apana_mudra(hand_landmarks):
    # Middle and Ring tips touch Thumb tip. Index and Pinky straight.
    lm = hand_landmarks.landmark
    thumb_tip = lm[4]
    mid_tip = lm[12]
    ring_tip = lm[16]
    wrist = lm[0]
    
    hand_scale = math.sqrt((wrist.x - mid_tip.x) ** 2 + (wrist.y - mid_tip.y) ** 2) + 1e-6
    
    d_mid = math.sqrt((thumb_tip.x - mid_tip.x)**2 + (thumb_tip.y - mid_tip.y)**2) / hand_scale
    d_ring = math.sqrt((thumb_tip.x - ring_tip.x)**2 + (thumb_tip.y - ring_tip.y)**2) / hand_scale
    
    def dist_sq(p1, p2): return (p1.x-p2.x)**2 + (p1.y-p2.y)**2
    index_ext = dist_sq(lm[8], wrist) > dist_sq(lm[6], wrist)
    pinky_ext = dist_sq(lm[20], wrist) > dist_sq(lm[18], wrist)
    
    index_ext = dist_sq(lm[8], wrist) > dist_sq(lm[6], wrist)
    pinky_ext = dist_sq(lm[20], wrist) > dist_sq(lm[18], wrist)
    
    return d_mid < 0.28 and d_ring < 0.28 and index_ext and pinky_ext

def detect_surya_mudra(hand_landmarks):
    # Ring finger bent, thumb pressing it.
    # This is hard to detect perfectly. Approximation: Ring tip close to thumb base (CMC or MCP), Thumb tip close to Ring PIP.
    lm = hand_landmarks.landmark
    thumb_tip = lm[4]
    ring_tip = lm[16]
    ring_pip = lm[14]
    wrist = lm[0]
    mid_tip = lm[12]
    
    hand_scale = math.sqrt((wrist.x - mid_tip.x) ** 2 + (wrist.y - mid_tip.y) ** 2) + 1e-6
    
    # Check if ring finger is folded
    ring_folded = dist_sq(ring_tip, wrist) < dist_sq(ring_pip, wrist)
    
    # Check if thumb is over ring finger (distance between thumb tip and ring pip/dip)
    d_thumb_ring = math.sqrt((thumb_tip.x - ring_pip.x)**2 + (thumb_tip.y - ring_pip.y)**2) / hand_scale
    
    # Check if thumb is over ring finger (distance between thumb tip and ring pip/dip)
    d_thumb_ring = math.sqrt((thumb_tip.x - ring_pip.x)**2 + (thumb_tip.y - ring_pip.y)**2) / hand_scale
    
    return ring_folded and d_thumb_ring < 0.30 # Further relaxed from 0.25

def detect_varun_mudra(hand_landmarks):
    # Pinky tip touches Thumb tip. Others straight.
    lm = hand_landmarks.landmark
    thumb_tip = lm[4]
    pinky_tip = lm[20]
    wrist = lm[0]
    mid_tip = lm[12]
    
    hand_scale = math.sqrt((wrist.x - mid_tip.x) ** 2 + (wrist.y - mid_tip.y) ** 2) + 1e-6
    
    d_pinky = math.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2) / hand_scale
    
    def dist_sq(p1, p2): return (p1.x-p2.x)**2 + (p1.y-p2.y)**2
    index_ext = dist_sq(lm[8], wrist) > dist_sq(lm[6], wrist)
    mid_ext = dist_sq(lm[12], wrist) > dist_sq(lm[10], wrist)
    ring_ext = dist_sq(lm[16], wrist) > dist_sq(lm[14], wrist)
    
    mid_ext = dist_sq(lm[12], wrist) > dist_sq(lm[10], wrist)
    ring_ext = dist_sq(lm[16], wrist) > dist_sq(lm[14], wrist)
    
    return d_pinky < 0.28 and index_ext and mid_ext and ring_ext

def dist_sq(p1, p2):
    return (p1.x-p2.x)**2 + (p1.y-p2.y)**2

def detect_open_palm(finger_states):
    return all(finger_states.values())


def detect_fist(finger_states):
    return not any(finger_states.values())


def detect_peace(hand_landmarks):
    # Index/middle extended, ring/pinky folded
    lm = hand_landmarks.landmark
    states = get_finger_states(hand_landmarks, 1, 1)
    return states["index"] and states["middle"] and not states["ring"] and not states["pinky"]


def classify_chakra_gesture(finger_states, hand_landmarks=None):
    # Enhanced classification using specific mudra functions
    if hand_landmarks:
        if detect_gyan_mudra(hand_landmarks): return "Gyan Mudra"
        if detect_prana_mudra(hand_landmarks): return "Prana Mudra"
        if detect_apana_mudra(hand_landmarks): return "Apana Mudra"
        if detect_surya_mudra(hand_landmarks): return "Surya Mudra"
        if detect_varun_mudra(hand_landmarks): return "Varun Mudra"
    
    # Fallback to finger states for basic chakra mapping if needed, or return None
    t = finger_states["thumb"]
    i = finger_states["index"]
    m = finger_states["middle"]
    r = finger_states["ring"]
    p = finger_states["pinky"]

    if not t and not i and not m and not r and not p:
        return "Root (Fist)"
    
    return None


def wrap_text(text, max_chars=60):
    words = text.split()
    lines = []
    line = ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line += (" " if line else "") + w
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def analyze_face(face_landmarks, img_w, img_h):
    if not face_landmarks:
        return (255, 255, 255), "No face", 0.02, 0.02

    lm = face_landmarks.landmark
    upper_lip = lm[13]
    lower_lip = lm[14]
    left_eye_top = lm[159]
    left_eye_bottom = lm[145]

    def dist(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    mouth_open = dist(upper_lip, lower_lip)
    eye_open = dist(left_eye_top, left_eye_bottom)

    # Gaze & Eye State Detection
    gaze_label = "Center"
    eye_ratio = 0.3 # Default Open
    gaze_x = 0.0 # -1.0 (Left) to 1.0 (Right)
    
    if len(lm) > 470: # Check if iris landmarks exist
        # Right Eye (User's Right, Screen Left)
        # Inner: 362, Outer: 263, Top: 386, Bottom: 374
        r_inner = lm[362]
        r_outer = lm[263]
        r_top = lm[386]
        r_bottom = lm[374]
        r_iris = lm[473]
        
        # Calculate Eye Aspect Ratio (EAR) - Scale Invariant
        h_dist = dist(r_inner, r_outer)
        v_dist = dist(r_top, r_bottom)
        
        if h_dist > 0:
            eye_ratio = v_dist / h_dist
            
            # Calculate Gaze X (Relative to center)
            eye_center_x = (r_inner.x + r_outer.x) / 2
            # Normalize: deviation / (half_width)
            # Factor 4.0 to make it more sensitive
            gaze_x = (r_iris.x - eye_center_x) / (h_dist / 2) * 4.0
        
        # Thresholds (tuned for mirrored/webcam view)
        # In mirrored view: Looking Left (Screen Left) -> Iris moves Left (smaller x)
        if gaze_x < -0.3:
            gaze_label = "Right" # Actually Screen Left (User's Right?) - Naming is confusing, let's rely on gaze_x
        elif gaze_x > 0.3:
            gaze_label = "Left" # Screen Right
            
    if mouth_open > 0.035:
        aura_color = (0, 255, 255)
        mood = "Expressive / Happy"
    elif eye_ratio < EYE_CLOSED_THRESHOLD:
        aura_color = (255, 128, 0)
        mood = "Calm / Meditative"
    else:
        aura_color = (255, 255, 255)
        mood = "Neutral"

    return aura_color, mood, eye_ratio, mouth_open, gaze_label, gaze_x


class BreathingTracker:
    def __init__(self, smoothing=0.9):
        self.prev_y = None
        self.smoothed = 0.0
        self.smoothing = smoothing
        self.last_time = time.time()
        self.breath_phase = 0.0

    def update(self, nose_y):
        if self.prev_y is None:
            self.prev_y = nose_y
            return
        dy = nose_y - self.prev_y
        self.prev_y = nose_y
        self.smoothed = self.smoothing * self.smoothed + (1 - self.smoothing) * dy
        dt = max(1e-3, time.time() - self.last_time)
        self.last_time = time.time()
        self.breath_phase += self.smoothed * 50
        if self.breath_phase > 2 * math.pi:
            self.breath_phase -= 2 * math.pi
        if self.breath_phase < 0:
            self.breath_phase += 2 * math.pi

    def get_breath_factor(self):
        return 1.0 + 0.3 * math.sin(self.breath_phase)


def draw_chakras(frame, center_x, top_y, bottom_y,
                 active_index, energies, aura_color,
                 breath_factor, t):
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

        if i == active_index:
            cv2.circle(frame, center, radius + 6, (255, 255, 255), 2)

        cv2.putText(frame, chakra_name, (center[0] + 30, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_chakra_meter(frame, energies):
    # PREMIUM CHAKRA METER - Highly Visible
    bar_w = 25  # Keep width same
    bar_h = 65  # [FIX] Reduced slightly (was 75) to fit lower start position
    gap = 10    # [FIX] Reduced gap (was 12)
    x0 = 30
    y0 = 130    # [FIX] Moved down (was 70) to make room for Indian Flag in Top Left
    
    for i, energy in enumerate(energies):
        y_top = y0 + i * (bar_h + gap)
        color = CHAKRA_COLORS[i]
        
        # PREMIUM GLOW EFFECT - Outer aura
        for glow_i in range(4):
            alpha = 0.1 - (glow_i * 0.025)
            offset = 4 - glow_i
            overlay_glow = frame.copy()
            cv2.rectangle(overlay_glow, (x0 - offset, y_top - offset), 
                         (x0 + bar_w + offset, y_top + bar_h + offset), color, -1)
            cv2.addWeighted(overlay_glow, alpha, frame, 1 - alpha, 0, frame)
        
        # Dark Background with Premium Border
        cv2.rectangle(frame, (x0, y_top), (x0 + bar_w, y_top + bar_h),
                      (15, 15, 20), -1)  # Very dark background
        # Bright colored border
        cv2.rectangle(frame, (x0, y_top), (x0 + bar_w, y_top + bar_h),
                      color, 2)  # Thicker colored border
                      
        # Filled Energy with GRADIENT
        filled_h = int(bar_h * energy)
        y_fill = y_top + (bar_h - filled_h)
        
        # Draw gradient fill
        for j in range(filled_h):
            ratio = j / max(1, filled_h)
            # Darken color at top, brighten at bottom
            b, g, r = color
            dark_factor = 0.5 + (0.5 * ratio)  # 0.5 to 1.0
            gradient_color = (int(b * dark_factor), int(g * dark_factor), int(r * dark_factor))
            cv2.line(frame, (x0 + 1, y_fill + j), (x0 + bar_w - 1, y_fill + j), gradient_color, 1)
        
        # Inner glow on filled portion
        if filled_h > 0:
            overlay_fill = frame.copy()
            cv2.rectangle(overlay_fill, (x0 + 1, y_fill), (x0 + bar_w - 1, y_top + bar_h - 1),
                         (255, 255, 255), -1)
            cv2.addWeighted(overlay_fill, 0.15, frame, 0.85, 0, frame)
                      
        # Percentage Text - Brighter and Bolder
        text_y = y_top + int(bar_h * 0.4)
        # [FIX] Black Border for Contrast
        cv2.putText(frame, f"{int(energy * 100)}%", (x0 + 35, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        # [FIX] Bright Yellow Text
        cv2.putText(frame, f"{int(energy * 100)}%", (x0 + 35, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Chakra Name - Colored to match chakra
        # [FIX] Black Border
        cv2.putText(frame, CHAKRA_NAMES[i].split()[0], (x0 + 35, text_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        # [FIX] Bright Yellow Text (User Request)
        cv2.putText(frame, CHAKRA_NAMES[i].split()[0], (x0 + 35, text_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


def generate_smart_coach_message(energies, mood_label, alignment_mode, gyan_active):
    if gyan_active:
        return "Gyan Mudra detected — Deep Meditation Mode."
    weakest_idx = int(np.argmin(energies))
    strongest_idx = int(np.argmax(energies))
    weakest_name = CHAKRA_NAMES[weakest_idx]
    strongest_name = CHAKRA_NAMES[strongest_idx]
    if alignment_mode:
        return "✨ Alignment Mode: All chakras are being gently balanced..."
    if energies[weakest_idx] < 0.3:
        return f"Tip: {weakest_name} is low. Try its gesture to recharge. ({mood_label})"
    if all(e > 0.7 for e in energies):
        return f"Beautiful! Your energy looks balanced. Stay with your breath. ({mood_label})"
    return f"Focus on breath. {strongest_name} is strong, {weakest_name} needs love. ({mood_label})"


def draw_gyan_sparkles(frame, center_x, center_y, radius):
    overlay = frame.copy()
    for _ in range(35):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(radius * 0.6, radius * 1.1)
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle))
        cv2.circle(overlay, (x, y), random.randint(2, 4), (0, 215, 255), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)


def draw_revolving_aura(frame, center_x, center_y, radius, t):
    """
    Draws a golden revolving aura around the head.
    """
    golden_color = (0, 215, 255) # BGR: Gold/Orange-ish
    
    # Draw main glowing halo
    overlay = frame.copy()
    cv2.circle(overlay, (center_x, center_y), radius, golden_color, -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    # Revolving particles
    num_particles = 8
    for i in range(num_particles):
        angle = t * 2.0 + (i * (2 * math.pi / num_particles))
        # Elliptical orbit to look like it's "over head"
        orbit_rx = radius * 1.2
        orbit_ry = radius * 0.4
        
        px = int(center_x + orbit_rx * math.cos(angle))
        py = int(center_y + orbit_ry * math.sin(angle) - radius * 0.5) # Shifted up slightly
        
        cv2.circle(frame, (px, py), 6, (255, 255, 255), -1) # White core
        cv2.circle(frame, (px, py), 10, golden_color, 2) # Golden rim


class PostureAnalyzer:
    def __init__(self):
        self.last_label = "Unknown"

    def assess(self, pose_landmarks):
        if not pose_landmarks:
            self.last_label = "No body"
            return 0.0, self.last_label

        lm = pose_landmarks.landmark
        ls, rs = lm[11], lm[12]
        lh, rh = lm[23], lm[24]

        mid_shoulders = ((ls.x + rs.x) * 0.5, (ls.y + rs.y) * 0.5)
        mid_hips = ((lh.x + rh.x) * 0.5, (lh.y + rh.y) * 0.5)

        dx = mid_shoulders[0] - mid_hips[0]
        dy = mid_shoulders[1] - mid_hips[1] + 1e-6
        spine_angle = abs(math.degrees(math.atan2(dx, dy)))  # 0 is vertical

        shoulder_level = abs(ls.y - rs.y)
        hips_level = abs(lh.y - rh.y)

        score = 1.0
        if spine_angle > 10:
            score -= min(0.5, (spine_angle - 10) / 40)
        if shoulder_level > 0.03:
            score -= min(0.3, (shoulder_level - 0.03) / 0.1)
        if hips_level > 0.03:
            score -= min(0.2, (hips_level - 0.03) / 0.1)

        score = max(0.0, min(1.0, score))
        if score > 0.8:
            label = "Aligned"
        elif score > 0.6:
            label = "Slight tilt"
        elif score > 0.4:
            label = "Adjust spine/shoulders"
        else:
            label = "Poor posture"
        self.last_label = label
        return score, label


def draw_smart_tracking(frame, hand_results, face_results, yoga_mode=False):
    """
    Draws 'smart' tracking overlays:
    - 21 hand points with connections
    - Face mesh (contours)
    - Dynamic style based on yoga_mode
    """
    if not yoga_mode:
        # Subtle mode
        hand_style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
        face_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        conn_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
    else:
        # "Extra Smart" Yoga Mode - Golden/High-tech look
        # Gold BGR: (0, 215, 255)
        hand_style = mp_drawing.DrawingSpec(color=(0, 215, 255), thickness=2, circle_radius=3)
        face_style = mp_drawing.DrawingSpec(color=(0, 215, 255), thickness=1, circle_radius=1)
        conn_style = mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2)

    # Draw Hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_style,
                conn_style
            )

    # Draw Face Mesh
    # Draw Face Mesh (Hidden for cleaner visuals, but tracking remains active)
    # if face_results.multi_face_landmarks:
    #     for face_landmarks in face_results.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(
    #             frame,
    #             face_landmarks,
    #             mp_face.FACEMESH_TESSELATION,
    #             landmark_drawing_spec=None,
    #             connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=1)
    #         )
    #         mp_drawing.draw_landmarks(
    #             frame,
    #             face_landmarks,
    #             mp_face.FACEMESH_CONTOURS,
    #             landmark_drawing_spec=None,
    #             connection_drawing_spec=face_style
    #         )

def detect_namaste(hand_results):
    """
    Detects if two hands are present and close together (Namaste/Anjali gesture).
    """
    if not hand_results.multi_hand_landmarks or len(hand_results.multi_hand_landmarks) < 2:
        return False
    
    # Simple check: distance between wrist points
    h1 = hand_results.multi_hand_landmarks[0].landmark[0]
    h2 = hand_results.multi_hand_landmarks[1].landmark[0]
    
    dist = math.sqrt((h1.x - h2.x)**2 + (h1.y - h2.y)**2)
    return dist < 0.40  # [FIX] Relaxed threshold (was 0.30) for easier detection

def draw_mini_hand(frame, cx, cy, mudra_name, scale=1.0):
    """
    Draws a stylized colorful hand skeleton representing the mudra.
    cx, cy: Center of the visualization
    """
    # Colors for fingers (Thumb, Index, Middle, Ring, Pinky)
    # High Visibility Rainbow Theme
    colors = [
        (0, 0, 255),    # Thumb (Red)
        (0, 255, 0),    # Index (Green)
        (0, 255, 255),  # Middle (Yellow)
        (0, 140, 255),  # Ring (Orange)
        (255, 0, 255)   # Pinky (Magenta)
    ]
    wrist_color = (255, 255, 255)
    
    # Base offsets for an open right hand (relative to cx, cy)
    # Coordinates: (x, y) where -y is up
    # Scale factor
    s = 25 * scale
    
    # Default Open Palm positions
    # Wrist, ThumbCMC, ThumbMCP, ThumbIP, ThumbTip
    # IndexMCP, IndexPIP, IndexDIP, IndexTip ...
    
    # Simplified: Wrist -> MCP -> Tip
    pts = {
        'wrist': (0, s*1.5),
        'thumb_base': (-s*0.6, s*0.8), 'thumb_tip': (-s*1.2, -s*0.2),
        'index_base': (-s*0.3, -s*0.5), 'index_tip': (-s*0.4, -s*1.8),
        'mid_base': (0, -s*0.6),      'mid_tip': (0, -s*2.0),
        'ring_base': (s*0.3, -s*0.5),   'ring_tip': (s*0.4, -s*1.8),
        'pinky_base': (s*0.5, -s*0.3),  'pinky_tip': (s*0.7, -s*1.4)
    }
    
    # Modify positions based on Mudra
    if mudra_name == "Gyan":
        # Index tip touches Thumb tip
        pts['index_tip'] = (-s*0.8, -s*0.5)
        pts['thumb_tip'] = (-s*0.8, -s*0.5)
        
    elif mudra_name == "Prana":
        # Ring & Pinky touch Thumb
        pts['ring_tip'] = (-s*0.5, s*0.2)
        pts['pinky_tip'] = (-s*0.5, s*0.2)
        pts['thumb_tip'] = (-s*0.5, s*0.2)
        
    elif mudra_name == "Apana":
        # Middle & Ring touch Thumb
        pts['mid_tip'] = (-s*0.5, s*0.2)
        pts['ring_tip'] = (-s*0.5, s*0.2)
        pts['thumb_tip'] = (-s*0.5, s*0.2)
        
    elif mudra_name == "Surya":
        # Ring folded to thumb base, Thumb presses it
        pts['ring_tip'] = pts['thumb_base'] # Ring to thumb base
        pts['thumb_tip'] = pts['thumb_base'] # Thumb presses down
        
    elif mudra_name == "Varun":
        # Pinky touches Thumb
        pts['pinky_tip'] = (-s*0.6, 0)
        pts['thumb_tip'] = (-s*0.6, 0)
        
    elif mudra_name == "Anjali":
        # Draw two hands meeting (Simplified as one vertical hand for icon)
        # Or better: Draw left and right hands meeting
        # For simplicity in this function, we'll just draw the Right hand in "Prayer" half
        # which means vertical, fingers together.
        pts['thumb_tip'] = (-s*0.2, -s*0.5)
        pts['index_tip'] = (0, -s*1.8)
        pts['mid_tip'] = (s*0.1, -s*1.9)
        pts['ring_tip'] = (s*0.2, -s*1.8)
        pts['pinky_tip'] = (s*0.3, -s*1.6)

    # Helper to draw line
    def dline(p1_name, p2_name, color):
        p1 = (int(cx + pts[p1_name][0]), int(cy + pts[p1_name][1]))
        p2 = (int(cx + pts[p2_name][0]), int(cy + pts[p2_name][1]))
        cv2.line(frame, p1, p2, color, 2)
        cv2.circle(frame, p2, 3, color, -1)

    # Draw Connections
    # Thumb
    dline('wrist', 'thumb_base', wrist_color)
    dline('thumb_base', 'thumb_tip', colors[0])
    
    # Index
    dline('wrist', 'index_base', wrist_color)
    dline('index_base', 'index_tip', colors[1])
    
    # Middle
    dline('wrist', 'mid_base', wrist_color)
    dline('mid_base', 'mid_tip', colors[2])
    
    # Ring
    dline('wrist', 'ring_base', wrist_color)
    dline('ring_base', 'ring_tip', colors[3])
    
    # Pinky
    dline('wrist', 'pinky_base', wrist_color)
    dline('pinky_base', 'pinky_tip', colors[4])


def draw_mudra_sidebar(frame, active_mudra):
    h, w, _ = frame.shape
    sidebar_w = 280 
    
    # --- Premium Glow Background ---
    overlay = frame.copy()
    
    # 1. Ultra Dark Glass Background (Almost Black with slight blue tint)
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (2, 2, 5), -1)
    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
    
    # 2. Glowing Left Border (Cyan/Gold Gradient Effect)
    # Stronger Glow
    for i in range(15):
        alpha = 0.15 - (i * 0.01)
        thickness = 30 - (i * 2)
        if thickness < 1: thickness = 1
        overlay_glow = frame.copy()
        cv2.line(overlay_glow, (w - sidebar_w, 0), (w - sidebar_w, h), (0, 255, 255), thickness)
        cv2.addWeighted(overlay_glow, alpha, frame, 1 - alpha, 0, frame)
        
    # Solid thin line for sharpness
    cv2.line(frame, (w - sidebar_w, 0), (w - sidebar_w, h), (0, 255, 255), 2)

    # Title with Glow - TRIPLEX Font for Premium Look
    cv2.putText(frame, "MUDRA GUIDE", (w - sidebar_w + 30, 50), 
                cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 255, 255), 2) # Yellow
    # Subtitle
    cv2.putText(frame, "Hand Yoga System", (w - sidebar_w + 35, 80), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 255, 255), 1) # Light Yellow
    
    mudras = [
        ("Gyan", "Wisdom"),
        ("Prana", "Vitality"),
        ("Apana", "Detox"),
        ("Surya", "Fire/Wt"),
        ("Varun", "Water"),
        ("Anjali", "Prayer")
    ]
    
    y = 100 # [FIX] Started Higher (was 120)
    for name, desc in mudras:
        color = (200, 255, 255) # Off-white/Yellowish
        thickness = 1
        bg_col = (30, 30, 20)
        border_col = (70, 70, 40)
        font = cv2.FONT_HERSHEY_DUPLEX 
        
        if active_mudra and name in active_mudra:
            color = (0, 255, 255) # Yellow
            thickness = 2
            bg_col = (50, 50, 0) # Dark Yellow background
            border_col = (0, 255, 255) # Yellow border
            
            # Glow effect for active item
            overlay_item = frame.copy()
            cv2.rectangle(overlay_item, (w - sidebar_w + 10, y - 25), (w - 10, y + 45), (0, 255, 255), -1)
            cv2.addWeighted(overlay_item, 0.2, frame, 0.8, 0, frame)
            
        # Background for item
        cv2.rectangle(frame, (w - sidebar_w + 10, y - 25), (w - 10, y + 45), bg_col, -1)
        cv2.rectangle(frame, (w - sidebar_w + 10, y - 25), (w - 10, y + 45), border_col, 1)
            
        cv2.putText(frame, f"{name}", (w - sidebar_w + 25, y), 
                    font, 0.7, color, thickness)
        cv2.putText(frame, f"{desc}", (w - sidebar_w + 25, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 220), 1)
        
        # Draw Mini Hand Visual
        mx = w - 60
        my = y + 10
        draw_mini_hand(frame, mx, my, name, scale=0.8)
        
        y += 65 # [FIX] Compacted More (was 75)
            
    # --- Premium Footer Box (CHAKRA AI FLOW) ---
    # [FIX] Moved UP significantly (was h-190)
    footer_y = h - 230
    # Glow Box
    cv2.rectangle(frame, (w - sidebar_w + 10, footer_y), (w - 10, h - 10), (0, 255, 255), 2)
    
    # Header
    cv2.putText(frame, "CHAKRA AI FLOW", (w - sidebar_w + 25, footer_y + 35), 
                cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 255), 1) # Yellow Text
                
    # Steps
    steps = [
        "- Sit in Lotus Pose", 
        "- Show Hand Mudras", 
        "- Close Eyes",
        "- Touch Nose (Kumbhaka)",
        "- Namaste -> Screenshot",
        "- Press 'q' -> Quit"   
    ]
    sy = footer_y + 65
    for step in steps:
        # [FIX] Green Color for specific instructions
        col = (230, 255, 255)
        if "Touch Nose" in step or "Namaste" in step or "Press 'q'" in step:
            col = (0, 255, 0) # Green
            
        cv2.putText(frame, step, (w - sidebar_w + 25, sy), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, col, 1)
        sy += 25


def draw_revolving_aura(frame, center_x, center_y, radius, t):
    """
    Draws a golden revolving aura around the head.
    """
    golden_color = (0, 215, 255) # BGR: Gold/Orange-ish
    
    # Draw main glowing halo
    overlay = frame.copy()
    cv2.circle(overlay, (center_x, center_y), radius, golden_color, -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    # Revolving particles
    num_particles = 8
    for i in range(num_particles):
        angle = t * 2.0 + (i * (2 * math.pi / num_particles))
        # Elliptical orbit to look like it's "over head"
        orbit_rx = radius * 1.2
        orbit_ry = radius * 0.4
        
        px = int(center_x + orbit_rx * math.cos(angle))
        py = int(center_y + orbit_ry * math.sin(angle) - radius * 0.5) # Shifted up slightly
        
        cv2.circle(frame, (px, py), 6, (255, 255, 255), -1) # White core
        cv2.circle(frame, (px, py), 10, golden_color, 2) # Golden rim


class PostureAnalyzer:
    def __init__(self):
        self.last_label = "Unknown"

    def assess(self, pose_landmarks):
        if not pose_landmarks:
            self.last_label = "No body"
            return 0.0, self.last_label

        lm = pose_landmarks.landmark
        ls, rs = lm[11], lm[12]
        lh, rh = lm[23], lm[24]

        mid_shoulders = ((ls.x + rs.x) * 0.5, (ls.y + rs.y) * 0.5)
        mid_hips = ((lh.x + rh.x) * 0.5, (lh.y + rh.y) * 0.5)

        dx = mid_shoulders[0] - mid_hips[0]
        dy = mid_shoulders[1] - mid_hips[1] + 1e-6
        spine_angle = abs(math.degrees(math.atan2(dx, dy)))  # 0 is vertical

        shoulder_level = abs(ls.y - rs.y)
        hips_level = abs(lh.y - rh.y)

        score = 1.0
        if spine_angle > 10:
            score -= min(0.5, (spine_angle - 10) / 40)
        if shoulder_level > 0.03:
            score -= min(0.3, (shoulder_level - 0.03) / 0.1)
        if hips_level > 0.03:
            score -= min(0.2, (hips_level - 0.03) / 0.1)

        score = max(0.0, min(1.0, score))
        if score > 0.8:
            label = "Aligned"
        elif score > 0.6:
            label = "Slight tilt"
        elif score > 0.4:
            label = "Adjust spine/shoulders"
        else:
            label = "Poor posture"
        self.last_label = label
        return score, label


def draw_smart_tracking(frame, hand_results, face_results, yoga_mode=False):
    """
    Draws 'smart' tracking overlays:
    - 21 hand points with connections
    - Face mesh (contours)
    - Dynamic style based on yoga_mode
    """
    if not yoga_mode:
        # Subtle mode
        hand_style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
        face_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        conn_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
    else:
        # "Extra Smart" Yoga Mode - Glowing/High-tech look
        hand_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
        face_style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
        conn_style = mp_drawing.DrawingSpec(color=(50, 205, 50), thickness=2)

    # Draw Hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_style,
                conn_style
            )

    # Draw Face Mesh
    # Draw Face Mesh (Hidden for cleaner visuals, but tracking remains active)
    # if face_results.multi_face_landmarks:
    #     for face_landmarks in face_results.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(
    #             frame,
    #             face_landmarks,
    #             mp_face.FACEMESH_TESSELATION,
    #             landmark_drawing_spec=None,
    #             connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=1)
    #         )
    #         mp_drawing.draw_landmarks(
    #             frame,
    #             face_landmarks,
    #             mp_face.FACEMESH_CONTOURS,
    #             landmark_drawing_spec=None,
    #             connection_drawing_spec=face_style
    #         )

def detect_namaste(hand_results):
    """
    Detects if two hands are present and close together (Namaste/Anjali gesture).
    """
    if not hand_results.multi_hand_landmarks or len(hand_results.multi_hand_landmarks) < 2:
        return False
    
    # Simple check: distance between wrist points
    h1 = hand_results.multi_hand_landmarks[0].landmark[0]
    h2 = hand_results.multi_hand_landmarks[1].landmark[0]
    
    dist = math.sqrt((h1.x - h2.x)**2 + (h1.y - h2.y)**2)
    return dist < 0.20  # Relaxed threshold

def draw_mini_hand(frame, cx, cy, mudra_name, scale=1.0):
    """
    Draws a stylized colorful hand skeleton representing the mudra.
    cx, cy: Center of the visualization
    """
    # Colors for fingers (Thumb, Index, Middle, Ring, Pinky)
    # High Visibility Rainbow Theme
    colors = [
        (0, 0, 255),    # Thumb (Red)
        (0, 255, 0),    # Index (Green)
        (0, 255, 255),  # Middle (Yellow)
        (0, 140, 255),  # Ring (Orange)
        (255, 0, 255)   # Pinky (Magenta)
    ]
    wrist_color = (255, 255, 255)
    
    # Base offsets for an open right hand (relative to cx, cy)
    # Coordinates: (x, y) where -y is up
    # Scale factor
    s = 25 * scale
    
    # Default Open Palm positions
    # Wrist, ThumbCMC, ThumbMCP, ThumbIP, ThumbTip
    # IndexMCP, IndexPIP, IndexDIP, IndexTip ...
    
    # Simplified: Wrist -> MCP -> Tip
    pts = {
        'wrist': (0, s*1.5),
        'thumb_base': (-s*0.6, s*0.8), 'thumb_tip': (-s*1.2, -s*0.2),
        'index_base': (-s*0.3, -s*0.5), 'index_tip': (-s*0.4, -s*1.8),
        'mid_base': (0, -s*0.6),      'mid_tip': (0, -s*2.0),
        'ring_base': (s*0.3, -s*0.5),   'ring_tip': (s*0.4, -s*1.8),
        'pinky_base': (s*0.5, -s*0.3),  'pinky_tip': (s*0.7, -s*1.4)
    }
    
    # Modify positions based on Mudra
    if mudra_name == "Gyan":
        # Index tip touches Thumb tip
        pts['index_tip'] = (-s*0.8, -s*0.5)
        pts['thumb_tip'] = (-s*0.8, -s*0.5)
        
    elif mudra_name == "Prana":
        # Ring & Pinky touch Thumb
        pts['ring_tip'] = (-s*0.5, s*0.2)
        pts['pinky_tip'] = (-s*0.5, s*0.2)
        pts['thumb_tip'] = (-s*0.5, s*0.2)
        
    elif mudra_name == "Apana":
        # Middle & Ring touch Thumb
        pts['mid_tip'] = (-s*0.5, s*0.2)
        pts['ring_tip'] = (-s*0.5, s*0.2)
        pts['thumb_tip'] = (-s*0.5, s*0.2)
        
    elif mudra_name == "Surya":
        # Ring folded to thumb base, Thumb presses it
        pts['ring_tip'] = pts['thumb_base'] # Ring to thumb base
        pts['thumb_tip'] = pts['thumb_base'] # Thumb presses down
        
    elif mudra_name == "Varun":
        # Pinky touches Thumb
        pts['pinky_tip'] = (-s*0.6, 0)
        pts['thumb_tip'] = (-s*0.6, 0)
        
    elif mudra_name == "Anjali":
        # Draw two hands meeting (Simplified as one vertical hand for icon)
        # Or better: Draw left and right hands meeting
        # For simplicity in this function, we'll just draw the Right hand in "Prayer" half
        # which means vertical, fingers together.
        pts['thumb_tip'] = (-s*0.2, -s*0.5)
        pts['index_tip'] = (0, -s*1.8)
        pts['mid_tip'] = (s*0.1, -s*1.9)
        pts['ring_tip'] = (s*0.2, -s*1.8)
        pts['pinky_tip'] = (s*0.3, -s*1.6)

    # Helper to draw line - REFINED
    def dline(p1_name, p2_name, color):
        p1 = (int(cx + pts[p1_name][0]), int(cy + pts[p1_name][1]))
        p2 = (int(cx + pts[p2_name][0]), int(cy + pts[p2_name][1]))
        # Thinner, crisp lines
        cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
        # Glowing joints (small white center, colored rim)
        cv2.circle(frame, p2, 2, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p2, 1, (255, 255, 255), -1, cv2.LINE_AA)

    # Draw Connections
    # Thumb
    dline('wrist', 'thumb_base', wrist_color)
    dline('thumb_base', 'thumb_tip', colors[0])
    
    # Index
    dline('wrist', 'index_base', wrist_color)
    dline('index_base', 'index_tip', colors[1])
    
    # Middle
    dline('wrist', 'mid_base', wrist_color)
    dline('mid_base', 'mid_tip', colors[2])
    
    # Ring
    dline('wrist', 'ring_base', wrist_color)
    dline('ring_base', 'ring_tip', colors[3])
    
    # Pinky
    dline('wrist', 'pinky_base', wrist_color)
    dline('pinky_base', 'pinky_tip', colors[4])


def draw_mudra_sidebar(frame, active_mudra):
    h, w, _ = frame.shape
    sidebar_w = 280 
    
    # Glassmorphism Background - HIGH VISIBILITY
    overlay = frame.copy()
    # Much lighter background (Dark Grey/Blue) for contrast
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (60, 70, 80), -1)
    # High alpha (0.85) to block out background noise
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Vertical Accent Line (Left Edge)
    cv2.line(frame, (w - sidebar_w, 0), (w - sidebar_w, h), (100, 255, 100), 2, cv2.LINE_AA)
    
    # [NEW] Animated Instruction (Premium UI)
    # "Touch nose to enable breathing exercise please touch heart rate sensor at this time"
    import time
    pulse = abs(math.sin(time.time() * 3)) # Smooth pulse
    
    # Dynamic Colors & Scale
    text_color = (0, 255, 255) # Cyan
    if pulse > 0.7: text_color = (0, 215, 255) # Gold peak
    
    scale = 0.45 + (0.05 * pulse) # Subtle pop (0.45 to 0.50)
    
    # Background Box for Visibility
    box_y = 15
    box_h = 90 # Increased height (was 75)
    cv2.rectangle(frame, (w - sidebar_w + 5, box_y), (w - 5, box_y + box_h), (20, 30, 40), -1) # Dark BG
    cv2.rectangle(frame, (w - sidebar_w + 5, box_y), (w - 5, box_y + box_h), (0, 255, 255), 1) # Border
    
    # [NEW] Arrow pointing Left (towards Face/Nose)
    # Tip at (w - sidebar_w - 10, box_y + 30)
    arrow_tip = (w - sidebar_w - 15, box_y + 35)
    arrow_base = (w - sidebar_w + 5, box_y + 35)
    cv2.arrowedLine(frame, arrow_base, arrow_tip, (0, 255, 255), 2, tipLength=0.3)
    
    # Text Lines (Centered in Sidebar)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Line 1: "Touch Nose to Enable"
    cv2.putText(frame, "Touch Nose to Enable", (w - sidebar_w + 20, box_y + 25), 
                font, scale, (255, 255, 255), 1, cv2.LINE_AA)
                
    # Line 2: "Breathing Exercise"
    cv2.putText(frame, "Breathing Exercise", (w - sidebar_w + 35, box_y + 50), # Spaced out
                font, scale, text_color, 1, cv2.LINE_AA)
                
    # Line 3: "Touch Heart Sensor"
    cv2.putText(frame, "Touch Heart Sensor", (w - sidebar_w + 30, box_y + 75), # Spaced out
                font, scale, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Title
    cv2.putText(frame, "Mudra Guide", (w - sidebar_w + 20, 135), # Moved down significantly (was 90)
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    mudras = [
        ("Gyan", "Wisdom"),
        ("Prana", "Vitality"),
        ("Apana", "Detox"),
        ("Surya", "Fire/Wt"),
        ("Varun", "Water"),
        ("Anjali", "Prayer")
    ]
    
    y = 170 # [FIX] Moved Down to avoid title overlap (was 140)
    for name, desc in mudras:
        is_active = (active_mudra and name in active_mudra)
        
        # Item Background
        # Default: visible but subtle
        bg_col = (255, 255, 255)
        bg_alpha = 0.15
        text_col = (255, 255, 255) # Pure White
        desc_col = (220, 220, 220) # Light Grey
        
        if is_active:
            # Active: Bright Green Highlight
            bg_col = (50, 200, 50) # Green
            bg_alpha = 0.6 # Stronger highlight
            text_col = (255, 255, 255)
            desc_col = (255, 255, 255)
            
            # Glowing Left Border for active item - GREEN
            cv2.line(frame, (w - sidebar_w, y - 25), (w - sidebar_w, y + 45), (0, 255, 0), 5, cv2.LINE_AA)
            
        # Draw Item Background
        overlay_item = frame.copy()
        cv2.rectangle(overlay_item, (w - sidebar_w + 2, y - 25), (w, y + 45), bg_col, -1)
        cv2.addWeighted(overlay_item, bg_alpha, frame, 1 - bg_alpha, 0, frame)
            
        # Text
        cv2.putText(frame, f"{name}", (w - sidebar_w + 20, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{desc}", (w - sidebar_w + 20, y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, desc_col, 1, cv2.LINE_AA)
        
        # Draw Mini Hand Visual
        mx = w - 60
        my = y + 10
        draw_mini_hand(frame, mx, my, name, scale=0.8)
            
        y += 65 # [FIX] Compacted spacing (was 70)

    # Draw Info Panel
    # If active mudra, show its info. Else show Default Guide.
    shown_mudra = None
    if active_mudra:
        for m in active_mudra:
            shown_mudra = m
            break
            
    draw_mudra_info_panel(frame, shown_mudra)

# Detailed Benefits Dictionary
MUDRA_INFO = {
    "Gyan": [
        "GYAN MUDRA (Wisdom)",
        "Benefits:",
        "- Improves concentration & memory",
        "- Sharpens brain power",
        "- Reduces stress & anxiety",
        "- Promotes spiritual growth"
    ],
    "Prana": [
        "PRANA MUDRA (Vitality)",
        "Benefits:",
        "- Boosts energy & immunity",
        "- Reduces fatigue",
        "- Improves vision",
        "- Activates root chakra"
    ],
    "Apana": [
        "APANA MUDRA (Detox)",
        "Benefits:",
        "- Detoxifies the body",
        "- Improves digestion",
        "- Strengthens excretion system",
        "- Promotes inner balance"
    ],
    "Surya": [
        "SURYA MUDRA (Fire)",
        "Benefits:",
        "- Boosts metabolism",
        "- Helps in weight loss",
        "- Generates body heat",
        "- Improves digestion"
    ],
    "Varun": [
        "VARUN MUDRA (Water)",
        "Benefits:",
        "- Balances water content",
        "- Hydrates skin & body",
        "- Cures skin diseases",
        "- Improves blood circulation"
    ],
    "Anjali": [
        "ANJALI MUDRA (Prayer)",
        "Benefits:",
        "- Promotes inner peace",
        "- Connects left & right brain",
        "- Expresses respect & gratitude",
        "- Calms the mind"
    ]
}

def draw_mudra_info_panel(frame, mudra_name=None):
    """
    Draws a stylish info panel vertically stacked BELOW the Anjali mudra in the sidebar.
    """
    h, w, _ = frame.shape
    sidebar_w = 280
    
    # Panel Size - Fit within sidebar width
    panel_w = sidebar_w - 20 # 260
    panel_h = 120 # [FIX] Increased height (was 110)
    
    # Position: Inside Sidebar Column, Below Anjali
    # Anjali ends at ~610. Next slot is safe at 630.
    x = w - sidebar_w + 10
    y = 580 # [FIX] Moved UP to avoid bottom cutoff (was 630)
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (40, 50, 60), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (0, 255, 0), 2)
    
    lines = []
    title_color = (0, 255, 0)
    
    if mudra_name and mudra_name in MUDRA_INFO:
        # Specific Mudra Info
        lines = MUDRA_INFO[mudra_name]
    else:
        # Default Guide
        lines = [
            # [REMOVED] Header & Hint per user request
            "- Sit in **Lotus Pose**",
            "- Show **Hand Mudras**",
            "- Close **Eyes**",
            "- Focus on **Breath**"
        ]
        title_color = (0, 255, 255) # Cyan for general guide
    
    # Title (First line) - REMOVED
    # Body
    py = y + 25 # [FIX] Start higher (was +30)
    for line in lines: # Iterate all lines
        col = (255, 255, 255)
        scale = 0.4 # [FIX] Smaller font (was 0.45)
        thick = 1
        
        if line.startswith("Benefits:") or line.startswith("Guide:"):
            col = (200, 255, 200)
            scale = 0.45 # Slightly larger header
            thick = 1
            cv2.putText(frame, line, (x + 15, py), 
                        cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
        elif line.startswith("Hint:"):
            # [FIX] Bright Yellow with Border for Hint
            cv2.putText(frame, line, (x + 15, py), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA) # Border
            cv2.putText(frame, line, (x + 15, py), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA) # Yellow
        else:
            # Keyword Highlighting Logic
            parts = line.split('**')
            px = x + 15
            
            for i, part in enumerate(parts):
                curr_col = col
                curr_thick = thick
                
                if i % 2 == 1: # Highlighted
                    curr_col = (0, 255, 255) 
                    curr_thick = 1 # Thinner for smaller text
                
                cv2.putText(frame, part, (px, py), 
                            cv2.FONT_HERSHEY_SIMPLEX, scale, curr_col, curr_thick, cv2.LINE_AA)
                
                (txt_w, _), _ = cv2.getTextSize(part, cv2.FONT_HERSHEY_SIMPLEX, scale, curr_thick)
                px += txt_w
                
        py += 20 # [FIX] Tighter spacing (was 25)


class AnalyticsTracker:
    def __init__(self):
        self.mudra_counts = {"Gyan": 0, "Fist": 0, "OpenPalm": 0, "Peace": 0}
        self.posture_alerts = 0
        self.posture_samples = []
        self.chakra_time = [0.0] * 7
        self.last_chakra_idx = None
        self.last_chakra_time = time.time()

    def record_chakra(self, idx):
        now = time.time()
        if self.last_chakra_idx is not None:
            self.chakra_time[self.last_chakra_idx] += now - self.last_chakra_time
        self.last_chakra_idx = idx
        self.last_chakra_time = now

    def record_mudra(self, name):
        if name in self.mudra_counts:
            self.mudra_counts[name] += 1

    def record_posture(self, score):
        self.posture_samples.append(score)
        if score < 0.5:
            self.posture_alerts += 1

    def summary(self):
        avg_posture = sum(self.posture_samples) / len(self.posture_samples) if self.posture_samples else 0.0
        return {
            "chakra_time": self.chakra_time,
            "mudras": self.mudra_counts,
            "posture_alerts": self.posture_alerts,
            "avg_posture": avg_posture,
        }



class HeartRateMonitor:
    def __init__(self, baud_rate=115200):
        self.ser = None
        self.heart_rate = 0
        self.spo2 = 0
        self.last_beat_time = 0
        self.connected = False
        self.buffer = ""
        self.baud_rate = baud_rate
        self.beat_detected_flag = False # New flag for visualizer
        self.hr_history = [] # History for stability analysis
        
        # [FIX] Reconnection Logic
        self.last_data_time = time.time()
        self.last_reconnect_attempt = 0
        
        self.connect()

    def connect(self):
        try:
            if self.connected and self.ser:
                self.ser.close()
            
            # Auto-detect Arduino
            ports = list(serial.tools.list_ports.comports())
            # print(f"[DEBUG] Found ports: {[p.description for p in ports]}") # DEBUG PRINT
            
            arduino_port = None
            for p in ports:
                # Added "Serial" generic match for some clones
                if "Arduino" in p.description or "CH340" in p.description or "USB Serial" in p.description or "Serial" in p.description:
                    arduino_port = p.device
                    break
            
            if arduino_port:
                self.ser = serial.Serial(arduino_port, self.baud_rate, timeout=0.1)
                self.connected = True
                self.last_data_time = time.time() # Reset timeout timer
                print(f"[INFO] Heart Rate Sensor connected on {arduino_port}")
            else:
                # print("[WARN] No Arduino found for Heart Rate Sensor.")
                self.connected = False
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            self.connected = False

    def update(self):
        self.beat_detected_flag = False # Reset flag each frame
        
        # [FIX] Auto-Reconnection Logic
        if not self.connected:
            if time.time() - self.last_reconnect_attempt > 2.0:
                self.last_reconnect_attempt = time.time()
                # print("[INFO] Attempting to reconnect sensor...")
                self.connect()
            return

        # [FIX] Timeout Detection (Sensor Freeze)
        # If no data received for 5 seconds, assume connection is dead
        if time.time() - self.last_data_time > 5.0:
            print("[WARN] Sensor timeout (no data for 5s). Resetting connection...")
            self.connected = False
            if self.ser:
                try:
                    self.ser.close()
                except:
                    pass
            self.ser = None
            return

        try:
            while self.ser.in_waiting:
                char = self.ser.read().decode('utf-8', errors='ignore')
                if char == '\n':
                    self.parse_data(self.buffer)
                    self.buffer = ""
                    self.last_data_time = time.time() # [FIX] Update timestamp
                else:
                    self.buffer += char
        except Exception as e:
            print(f"[WARN] Serial read error: {e}")
            self.connected = False
            if self.ser:
                try:
                    self.ser.close()
                except:
                    pass
            self.ser = None

    def parse_data(self, line):
        line = line.strip()
        if line == "BEAT":
            self.last_beat_time = time.time()
            self.beat_detected_flag = True
        elif line.startswith("HR:"):
            # Format: HR:75;SpO2:98
            try:
                parts = line.split(';')
                for part in parts:
                    if "HR:" in part:
                        raw_val = float(part.split(':')[1])
                        
                        # [ACCURACY MODE] Direct Passthrough
                        # Trust the Arduino's sophisticated processing.
                        
                        # [FIX] Handle "0" from Arduino immediately
                        if raw_val == 0:
                            self.heart_rate = 0
                            self.spo2 = 0
                            self.hr_history = []
                        elif 55 < raw_val < 115:
                            self.heart_rate = raw_val
                            self.last_beat_time = time.time()
                        
                    if "SpO2:" in part:
                        val = float(part.split(':')[1])
                        self.spo2 = min(val, 100.0) # Clamp to 100%
            except:
                pass

    def get_data(self):
        # TIMEOUT LOGIC: If no beat for 3.0 seconds (was 1.5), reset data (Synced with Arduino)
        # This ensures "Instant Reset" in the UI but allows for slower heart rates.
        if time.time() - self.last_beat_time > 3.0:
            self.heart_rate = 0
            self.spo2 = 0
            self.hr_history = [] # Clear history
            self.beat_detected_flag = False
            
        return self.heart_rate, self.spo2, self.last_beat_time, self.beat_detected_flag, self.hr_history

class PulseWaveVisualizer:
    def __init__(self):
        self.data = [0.0] * 100
        self.pulse_timer = 0.0
    
    def update(self, beat_detected):
        # Shift data
        self.data.pop(0)
        
        val = 0.0
        
        # Trigger pulse
        if beat_detected:
            self.pulse_timer = 1.0
            
        # Generate pulse shape
        if self.pulse_timer > 0:
            t = 1.0 - self.pulse_timer
            # QRS complex approximation
            if t < 0.2: val = math.sin(t * 10) * 0.2 # P wave
            elif t < 0.4: val = -0.5 + (t-0.2)*10 # Q dip
            elif t < 0.5: val = 1.0 - (t-0.4)*20 # R spike up
            elif t < 0.6: val = -0.5 + (t-0.5)*10 # S dip
            elif t < 0.8: val = math.sin((t-0.6)*10) * 0.3 # T wave
            
            self.pulse_timer -= 0.08 # Decay speed
        else:
            # Baseline noise
            val = (random.random() - 0.5) * 0.1
            
        self.data.append(val)

    def draw(self, frame, x, y, w, h, color):
        # Draw background
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
        
        # Draw wave
        points = []
        for i, val in enumerate(self.data):
            px = int(x + (i / len(self.data)) * w)
            # Map val (-1 to 1) to height
            py = int(y + h/2 - (val * h * 0.4))
            points.append((px, py))
            
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points)], False, color, 2, cv2.LINE_AA)

class MindWaveVisualizer:
    def __init__(self):
        self.data = [0.0] * 100
        self.phase = 0.0
        self.current_state = "Neutral"
        self.state_timer = 0
        
    def update(self, hr, hr_history):
        self.data.pop(0)
        
        # Determine State (with Hysteresis/Smoothing)
        self.state_timer += 1
        if self.state_timer > 30: # Only update state every ~1 second (30 frames)
            self.state_timer = 0
            variance = 0
            if len(hr_history) > 5:
                variance = np.var(hr_history)
                
            # Logic Refined for Yoga Context:
            # Relaxation: HR < 75 (Deep calm)
            # Concentration: 75 <= HR <= 95 AND Stable (Low Variance)
            # Anxiety/Stress: HR > 95 OR Unstable (High Variance)
            
            if hr > 95 or variance > 10:
                self.current_state = "Anxiety/Stress"
            elif 75 <= hr <= 95 and variance < 3:
                self.current_state = "Concentration"
            elif hr < 75:
                self.current_state = "Relaxation"
            else:
                self.current_state = "Neutral"
            
        val = 0.0
        self.phase += 0.1
        
        if self.current_state == "Anxiety/Stress":
            # Jagged, fast, chaotic
            val = math.sin(self.phase * 3) * 0.5 + (random.random()-0.5) * 0.8
        elif self.current_state == "Concentration":
            # Stable, consistent beam
            val = math.sin(self.phase) * 0.8
        elif self.current_state == "Relaxation":
            # Slow, smooth
            val = math.sin(self.phase * 0.5) * 0.6
        else:
            # Idle
            val = (random.random()-0.5) * 0.05
            
        self.data.append(val)
        return self.current_state

    def draw(self, frame, x, y, w, h, state):
        # Color based on state
        color = (200, 200, 200) # Grey
        if state == "Anxiety/Stress": color = (255, 0, 255) # Purple (to differ from Heart Red)
        elif state == "Concentration": color = (255, 200, 0) # Cyan/Blue-ish
        elif state == "Relaxation": color = (0, 255, 0) # Green
        
        # Draw background
        cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 20, 20), -1)
        
        # Draw wave
        points = []
        for i, val in enumerate(self.data):
            px = int(x + (i / len(self.data)) * w)
            py = int(y + h/2 - (val * h * 0.4))
            points.append((px, py))
            
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points)], False, color, 2, cv2.LINE_AA)
            
        return color

def draw_heart(frame, x, y, size, color, outline=False):
    # Draw a heart shape using two ellipses and a triangle
    # Left lobe
    cv2.ellipse(frame, (x - size//2, y - size//4), 
                (size//2, size//2), 0, 30, 210, color, -1)
    # Right lobe
    cv2.ellipse(frame, (x + size//2, y - size//4), 
                (size//2, size//2), 0, 150, 330, color, -1)
    # Triangle (Bottom)
    pts = np.array([
        [int(x - size*0.95), int(y - size*0.05)],
        [int(x + size*0.95), int(y - size*0.05)],
        [x, y + size]
    ], np.int32)
    cv2.fillPoly(frame, [pts], color)
    
    if outline:
        white = (255, 255, 255)
        thick = 2
        cv2.ellipse(frame, (x - size//2, y - size//4), (size//2, size//2), 0, 30, 210, white, thick)
        cv2.ellipse(frame, (x + size//2, y - size//4), (size//2, size//2), 0, 150, 330, white, thick)
        cv2.polylines(frame, [pts], True, white, thick)

# --- KUMBHAKA (BREATH RETENTION) MASTER ---
class KumbhakaTracker:
    def __init__(self):
        self.hr_buffer = []
        self.buffer_size = 20 # Approx 2-3 seconds of data
        self.prana_level = 0.0 # 0 to 100
        self.is_holding = False
        self.max_prana = 100.0
        self.charge_rate = 0.8 # How fast it fills per frame
        self.decay_rate = 2.0  # How fast it drops when breathing
        
    def update(self, current_hr, is_touching_nose=False):
        if current_hr <= 10: # Ignore noise/zeros
            self.is_holding = False
            self.prana_level = max(0, self.prana_level - self.decay_rate)
            return self.prana_level, self.is_holding
            
        self.hr_buffer.append(current_hr)
        if len(self.hr_buffer) > self.buffer_size:
            self.hr_buffer.pop(0)
            
        # Calculate Stability (Standard Deviation)
        stdev = 100.0
        if len(self.hr_buffer) >= 5:
            avg = sum(self.hr_buffer) / len(self.hr_buffer)
            variance = sum([((x - avg) ** 2) for x in self.hr_buffer]) / len(self.hr_buffer)
            stdev = variance ** 0.5
            
        # Logic: Hold ONLY if Nose Touch Detected (User Request)
        if is_touching_nose: 
            self.is_holding = True
            self.prana_level = min(self.max_prana, self.prana_level + self.charge_rate)
        else:
            self.is_holding = False
            self.prana_level = max(0, self.prana_level - self.decay_rate)
            
        return self.prana_level, self.is_holding

kumbhaka_tracker = KumbhakaTracker()

def draw_kumbhaka_bar(frame, prana_level, is_holding):
    h, w, _ = frame.shape
    
    # Only show if there is some Prana or actively holding
    if prana_level < 1.0 and not is_holding:
        return

    # Position: Center Bottom Overlay
    bar_w = 400
    bar_h = 30
    x = (w - bar_w) // 2
    y = h - 150
    
    # Background
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (20, 20, 40), -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (100, 100, 100), 1)
    
    # Fill (Cosmic Blue/Purple Gradient) - CHANGED from Orange
    fill_w = int((prana_level / 100.0) * bar_w)
    if fill_w > 0:
        # Cosmic Blue (Neon Cyan/Blue)
        # BGR: (255, 200, 0) -> Cyan-ish Blue
        cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), (255, 200, 0), -1) 
        
        # Glow Effect
        if is_holding:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + fill_w, y + bar_h), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
    # Text
    status = "KUMBHAKA ACTIVE - CHARGING PRANA" if is_holding else "Releasing..."
    col = (255, 200, 0) if is_holding else (200, 200, 200)
    
    cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(frame, f"{int(prana_level)}%", (x + bar_w + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

# --- ELEMENTAL MASTERY EFFECTS ---
class ElementalEffects:
    def __init__(self):
        self.particles = [] # List of [x, y, vx, vy, life, type, color]
        self.max_particles = 100
        
        # Load Brain Image for Gyan Mudra
        self.brain_img = None
        if os.path.exists("brain_glow.png"):
            self.brain_img = cv2.imread("brain_glow.png", cv2.IMREAD_UNCHANGED)
            # Ensure alpha
            if self.brain_img is not None and self.brain_img.shape[2] != 4:
                self.brain_img = cv2.cvtColor(self.brain_img, cv2.COLOR_BGR2BGRA)
        else:
            print("[WARN] brain_glow.png not found for Gyan Mudra effect.")
        
        # Load Sun Image for Surya Mudra
        self.sun_img = None
        if os.path.exists("sun_glow.png"):
            self.sun_img = cv2.imread("sun_glow.png", cv2.IMREAD_UNCHANGED)
            if self.sun_img is not None and self.sun_img.shape[2] != 4:
                self.sun_img = cv2.cvtColor(self.sun_img, cv2.COLOR_BGR2BGRA)
        else:
            print("[WARN] sun_glow.png not found for Surya Mudra effect.")
        
    def update_and_draw(self, frame, mudra_name, hand_landmarks_list, face_landmarks):
        h, w, _ = frame.shape
        
        # 1. Spawn Particles based on Mudra
        if mudra_name == "Surya Mudra": # FIRE (Sun in Palm)
            # [FIX] Replaced Fire Particles with Glowing Sun
            self._draw_glowing_sun(frame, hand_landmarks_list, w, h)
        elif mudra_name == "Varun Mudra": # WATER
            self._spawn_water(hand_landmarks_list, w, h)
        elif mudra_name == "Prana Mudra": # NATURE
            self._spawn_nature(hand_landmarks_list, w, h)
        elif mudra_name == "Gyan Mudra": # KNOWLEDGE (Brain in Palm)
            # [FIX] Replaced Book with Glowing Brain in Palm
            self._draw_glowing_brain(frame, hand_landmarks_list, w, h)
            
        # 2. Update & Draw Particles
        new_particles = []
        for p in self.particles:
            x, y, vx, vy, life, p_type, col = p
            
            # Physics
            x += vx
            y += vy
            life -= 0.05
            
            # Render
            if life > 0:
                alpha = life
                overlay = frame.copy()
                
                if p_type == "fire":
                    # Rise up, flicker
                    vy -= 0.5 
                    radius = int(life * 15)
                    cv2.circle(overlay, (int(x), int(y)), radius, col, -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    
                elif p_type == "water":
                    # Rise slowly, wobble
                    vy = -2 + math.sin(time.time() * 10 + x) 
                    radius = int(life * 10)
                    cv2.circle(overlay, (int(x), int(y)), radius, col, 1) # Bubble
                    cv2.circle(overlay, (int(x - radius*0.3), int(y - radius*0.3)), int(radius*0.2), (255, 255, 255), -1) # Highlight
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                elif p_type == "nature":
                    # Float around
                    vx = math.sin(time.time() * 5 + y*0.1) * 2
                    vy = math.cos(time.time() * 5 + x*0.1) * 2
                    radius = int(life * 8)
                    # Draw Leaf shape (ellipse)
                    cv2.ellipse(overlay, (int(x), int(y)), (radius, radius//2), int(time.time()*100), 0, 360, col, -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                new_particles.append([x, y, vx, vy, life, p_type, col])
                
        self.particles = new_particles

    def _spawn_fire(self, hand_list, w, h):
        # [DEPRECATED] Replaced by Sun Image
        pass

    def _spawn_water(self, hand_list, w, h):
        if not hand_list: return
        for hand_lm in hand_list:
            lm = hand_lm.landmark[0] # Wrist/Palm
            cx, cy = int(lm.x * w), int(lm.y * h)
            if random.random() < 0.2:
                # Water Colors: Blue, Cyan, White
                col = random.choice([(255, 0, 0), (255, 255, 0), (255, 255, 255)])
                self.particles.append([cx + random.randint(-40, 40), cy, 0, 0, 1.0, "water", col])

    def _spawn_nature(self, hand_list, w, h):
        if not hand_list: return
        for hand_lm in hand_list:
            lm = hand_lm.landmark[8] # Index tip (or any)
            cx, cy = int(lm.x * w), int(lm.y * h)
            if random.random() < 0.2:
                # Nature Colors: Green, Lime
                col = random.choice([(0, 255, 0), (50, 205, 50), (0, 255, 127)])
                self.particles.append([cx, cy, 0, 0, 1.0, "nature", col])
    
    def _draw_glowing_sun(self, frame, hand_list, w, h):
        if not hand_list or self.sun_img is None: return
        
        for hand_lm in hand_list:
            # Calculate Palm Center
            wrist = hand_lm.landmark[0]
            middle_mcp = hand_lm.landmark[9]
            
            cx = int((wrist.x + middle_mcp.x) / 2 * w)
            cy = int((wrist.y + middle_mcp.y) / 2 * h)
            
            # Size of Sun Icon
            size = 90
            x1 = cx - size // 2
            y1 = cy - size // 2
            x2 = x1 + size
            y2 = y1 + size
            
            # Bounds Check
            if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h: continue
            
            # Resize Sun
            sun_resized = cv2.resize(self.sun_img, (size, size))
            
            # Alpha Blending
            roi = frame[y1:y2, x1:x2]
            b_b, b_g, b_r, b_a = cv2.split(sun_resized)
            mask = b_a / 255.0
            inv_mask = 1.0 - mask
            
            for c in range(3):
                roi[:, :, c] = (mask * sun_resized[:, :, c] + inv_mask * roi[:, :, c])
                
            frame[y1:y2, x1:x2] = roi
            
            # Add extra glow (Yellow)
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), size // 2 + 15, (0, 255, 255), -1) 
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    def _draw_glowing_brain(self, frame, hand_list, w, h):
        if not hand_list or self.brain_img is None: return
        
        for hand_lm in hand_list:
            # Calculate Palm Center (Approx between Wrist 0 and Middle MCP 9)
            wrist = hand_lm.landmark[0]
            middle_mcp = hand_lm.landmark[9]
            
            cx = int((wrist.x + middle_mcp.x) / 2 * w)
            cy = int((wrist.y + middle_mcp.y) / 2 * h)
            
            # Size of Brain Icon
            size = 80
            x1 = cx - size // 2
            y1 = cy - size // 2
            x2 = x1 + size
            y2 = y1 + size
            
            # Bounds Check
            if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h: continue
            
            # Resize Brain
            brain_resized = cv2.resize(self.brain_img, (size, size))
            
            # Alpha Blending
            roi = frame[y1:y2, x1:x2]
            
            # Separate channels
            b_b, b_g, b_r, b_a = cv2.split(brain_resized)
            
            # Create mask
            mask = b_a / 255.0
            inv_mask = 1.0 - mask
            
            # Blend
            for c in range(3):
                roi[:, :, c] = (mask * brain_resized[:, :, c] + inv_mask * roi[:, :, c])
                
            frame[y1:y2, x1:x2] = roi
            
            # Add extra glow (simple circle behind)
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), size // 2 + 10, (255, 255, 0), -1) # Cyan/Yellow glow
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

elemental_effects = ElementalEffects()

# --- THIRD EYE INTERFACE ---
class ThirdEyeController:
    def __init__(self):
        self.dwell_time = 0
        self.target = None # "Left", "Right", None
        self.beam_color = (255, 0, 255) # Purple default
        
    def update_and_draw(self, frame, face_landmarks, gaze_x, chakra_energies):
        if not face_landmarks: return
        
        h, w, _ = frame.shape
        
        # Forehead (Third Eye) Landmark: 10
        forehead = face_landmarks.landmark[10]
        fx, fy = int(forehead.x * w), int(forehead.y * h)
        
        # Calculate Beam Target based on Gaze X
        # gaze_x is -1.0 (Left/Screen Left) to 1.0 (Right/Screen Right)
        # Map to screen width
        # Center is w/2.
        # Target X = w/2 + gaze_x * (w/2)
        
        target_x = int(w/2 + gaze_x * (w * 0.8)) # 0.8 factor to reach edges easily
        target_y = int(h * 0.6) # Look slightly down/center
        
        # Clamp
        target_x = max(0, min(w, target_x))
        
        # Interaction Logic
        current_target = None
        if target_x < w * 0.2: # Looking at Left Zone (Chakra Meter)
            current_target = "Left"
        elif target_x > w * 0.8: # Looking at Right Zone (Stats)
            current_target = "Right"
            
        # Dwell Logic
        if current_target and current_target == self.target:
            self.dwell_time += 1
        else:
            self.dwell_time = 0
            self.target = current_target
            
        # Action Trigger (Dwell > 30 frames ~ 1 sec)
        is_locked = self.dwell_time > 20
        
        # Visuals
        # ALWAYS draw the Divine Glow at Third Eye (Forehead)
        # Concentric circles for Glow Effect
        # White/Gold color
        glow_color = (255, 255, 255) 
        if is_locked:
             glow_color = (0, 255, 0) # Green tint when active
             
        # Draw Glow
        cv2.circle(frame, (fx, fy), 5, (255, 255, 255), -1) # Core
        cv2.circle(frame, (fx, fy), 10, glow_color, 2)
        cv2.circle(frame, (fx, fy), 20, glow_color, 1)
        
        # Add a "Light" overlay for bloom
        overlay = frame.copy()
        cv2.circle(overlay, (fx, fy), 40, glow_color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Only perform actions if there is a target (Left or Right)
        if current_target:
            if is_locked:
                # Perform Action
                if self.target == "Left":
                    # Charge Weakest Chakra
                    weakest_idx = int(np.argmin(chakra_energies))
                    chakra_energies[weakest_idx] = min(1.0, chakra_energies[weakest_idx] + 0.01)
                    cv2.putText(frame, f"Charging {CHAKRA_NAMES[weakest_idx]}...", (target_x + 20, target_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                               
                elif self.target == "Right":
                    cv2.putText(frame, "Focusing...", (target_x - 150, target_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw Target Reticle (Subtle)
            # No beam, just the target indicator
            cv2.circle(frame, (target_x, target_y), 10, glow_color, 2)
            if is_locked:
                 cv2.circle(frame, (target_x, target_y), 5, (255, 255, 255), -1)

third_eye = ThirdEyeController()

class MultiGraphVisualizer:
    def __init__(self, max_len=100):
        self.max_len = max_len
        self.hrv_data = [0.0] * max_len
        self.prana_data = [0.0] * max_len
        self.focus_data = [0.0] * max_len
        self.hrv_index_data = [0.0] * max_len # [NEW] HRV Index
        self.pulse_data = [0.0] * max_len
        self.phase = 0.0
        
    def update(self, hr, hr_history, spo2, posture_score, beat_detected, avg_energy=0.5, hrv_val=50.0):
        self.phase += 0.2 # Faster animation
        
        # 1. Heart Rhythm (ECG Style)
        if beat_detected:
            self.pulse_data[-4:] = [-0.2, 1.0, -0.5, 0.1]
        else:
            self.pulse_data.append(random.uniform(-0.02, 0.02))
        
        if len(self.pulse_data) > self.max_len:
            self.pulse_data.pop(0)
            
        # 2. Stress (HRV) -> "EQ Bars"
        # Generate varied data 0.0 to 1.0
        # Stress factor makes bars higher and more erratic
        stress_factor = max(0.0, min(1.0, (hr - 60) / 40))
        
        # Base noise
        noise = random.uniform(0.0, 0.3)
        # Add stress spikes
        if random.random() < stress_factor:
            noise += random.uniform(0.3, 0.7)
            
        self.hrv_data.append(noise)
        if len(self.hrv_data) > self.max_len: self.hrv_data.pop(0)

        # 3. Prana (Energy) -> "Double Wave"
        # Now linked to REAL Chakra Energy (avg_energy)
        # Base level = avg_energy
        # Add "Breathing" sine wave on top
        breath_wave = 0.1 * math.sin(self.phase * 0.15)
        val_prana = avg_energy + breath_wave
        
        # Clamp 0.0 to 1.0
        val_prana = max(0.0, min(1.0, val_prana))
        
        self.prana_data.append(val_prana)
        if len(self.prana_data) > self.max_len: self.prana_data.pop(0)
        
        # 4. Focus -> "Glow Beam"
        # Value near 0.5 (center). 
        # Focused = Tight line. Distracted = Wide wobble.
        focus_score = 1.0 - stress_factor
        wobble = (1.0 - focus_score) * 0.3
        val_focus = 0.5 + math.sin(self.phase * 0.3) * wobble + random.uniform(-0.05, 0.05)
        self.focus_data.append(val_focus)
        if len(self.focus_data) > self.max_len: self.focus_data.pop(0)

        # 5. HRV Index -> "Filled Area"
        # Normalize HRV (typically 20-100ms) to 0.0-1.0
        norm_hrv = max(0.0, min(1.0, (hrv_val - 20) / 80))
        self.hrv_index_data.append(norm_hrv)
        if len(self.hrv_index_data) > self.max_len: self.hrv_index_data.pop(0)

    def draw_graph(self, frame, x, y, w, h, data, color, label, fill=False, style="line"):
        # Background
        cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 15, 20), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 1)
        
        # Label
        cv2.putText(frame, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if style == "bars":
            # EQ Bar Style
            bar_w = max(1, w // len(data))
            for i, val in enumerate(data):
                bx = int(x + i * bar_w)
                bh = int(val * h * 0.8) # Scale to height
                by = int(y + h - bh)
                
                # Color gradient based on height
                c = color
                if val > 0.6: c = (0, 0, 255) # Red tips for high stress
                elif val > 0.3: c = (0, 255, 255) # Yellow mid
                
                cv2.rectangle(frame, (bx, by), (bx + bar_w - 1, y + h), c, -1)
                
        elif style == "double_wave":
            # Double Sine Wave
            points1 = []
            points2 = []
            for i, val in enumerate(data):
                px = int(x + (i / self.max_len) * w)
                
                # Wave 1 (Main)
                py1 = int(y + h - (val * h * 0.9))
                points1.append((px, py1))
                
                # Wave 2 (Phase shifted, smaller)
                val2 = val * 0.8 + 0.1 * math.sin(i * 0.2 + self.phase)
                py2 = int(y + h - (val2 * h * 0.9))
                points2.append((px, py2))
                
            if len(points1) > 1:
                # Fill between waves? Or just draw two lines
                cv2.polylines(frame, [np.array(points1)], False, color, 2, cv2.LINE_AA)
                cv2.polylines(frame, [np.array(points2)], False, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Fill bottom
                poly_pts = np.vstack([np.array(points1), [x+w, y+h], [x, y+h]])
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly_pts], color)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        elif style == "glow_beam":
            # Glowing Line
            points = []
            for i, val in enumerate(data):
                px = int(x + (i / self.max_len) * w)
                py = int(y + h - (val * h * 0.9))
                points.append((px, py))
                
            if len(points) > 1:
                pts_arr = np.array(points, dtype=np.int32)
                # Outer Glow
                cv2.polylines(frame, [pts_arr], False, color, 6, cv2.LINE_AA) # Thick colored
                # Inner Core
                cv2.polylines(frame, [pts_arr], False, (255, 255, 255), 2, cv2.LINE_AA) # White core

        elif style == "ecg":
            # ECG look
            points = []
            for i, val in enumerate(data):
                px = int(x + (i / self.max_len) * w)
                py = int(y + h/2 - (val * h * 0.4))
                points.append((px, py))
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points)], False, color, 2, cv2.LINE_AA)
                
        elif style == "filled_area":
            # Filled Area Graph
            points = []
            for i, val in enumerate(data):
                px = int(x + (i / self.max_len) * w)
                py = int(y + h - (val * h * 0.9))
                points.append((px, py))
            
            if len(points) > 1:
                # Close the polygon
                poly_pts = np.vstack([np.array(points), [x+w, y+h], [x, y+h]])
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly_pts], color)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.polylines(frame, [np.array(points)], False, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            # Default Line
            points = []
            for i, val in enumerate(data):
                px = int(x + (i / self.max_len) * w)
                py = int(y + h - (val * h * 0.9))
                points.append((px, py))
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points)], False, color, 2, cv2.LINE_AA)

class PhysiologyEngine:
    def __init__(self):
        self.history_bpm = []
        self.history_ibi = []
        self.last_beat_time = time.time()
        self.min_ibi = 300  # 200 BPM
        self.max_ibi = 1500 # 40 BPM
        
        # [NEW] Nadi Pariksha History
        self.history_vata = []
        self.history_pitta = []
        self.history_kapha = []
        
        # [NEW] Insight Timer
        self.last_insight_time = 0
        self.current_insight = "Scanning bio-rhythms..."
        
    def _get_tiny_graph(self, data, length=7):
        if not data or len(data) < 2: return "       "
        # Unicode bars:   ▂ ▃ ▄ ▅ ▆ ▇ █
        bars = "  ▂▃▄▅▆▇█"
        # Normalize last 'length' points
        recent = data[-length:]
        if not recent: return "       "
        mn, mx = min(recent), max(recent)
        if mx == mn: return "▃" * len(recent)
        
        graph = ""
        for v in recent:
            idx = int((v - mn) / (mx - mn + 1e-6) * (len(bars) - 1))
            graph += bars[idx]
        return graph.ljust(length)

    def analyze(self, bpm, beat_detected, gaze_label="Center"):
        # [FIX] Handle No Sensor Input
        if bpm <= 0:
            return {
                'stress_score': 0.0,
                'calm_score': 0.0,
                'focus_score': 0.0,
                'insight_text': "Waiting for Sensor...",
                'tiny_graphs': {'vata': [], 'pitta': [], 'kapha': []}
            }

        now = time.time()
        
        # 1. Calculate IBI
        if beat_detected:
            ibi = (now - self.last_beat_time) * 1000.0 # ms
            self.last_beat_time = now
            if self.min_ibi < ibi < self.max_ibi:
                self.history_ibi.append(ibi)
                if len(self.history_ibi) > 20: self.history_ibi.pop(0)
        
        # Update BPM history
        if bpm > 0:
            self.history_bpm.append(bpm)
            if len(self.history_bpm) > 20: self.history_bpm.pop(0)
            
        # 2. Calculate HRV (RMSSD)
        hrv_rmssd = 0.0
        if len(self.history_ibi) > 2:
            diffs = np.diff(self.history_ibi)
            sq_diffs = diffs ** 2
            mean_sq = np.mean(sq_diffs)
            hrv_rmssd = math.sqrt(mean_sq)
            
        # 3. Derive Metrics
        # Stress: High BPM + Low HRV
        # Normalize BPM (60-100) -> 0-1
        norm_bpm = max(0, min(1, (bpm - 60) / 40)) if bpm > 0 else 0
        # Normalize HRV (10-100) -> 0-1 (Higher is better)
        norm_hrv = max(0, min(1, (hrv_rmssd - 10) / 90))
        
        stress_score = (norm_bpm * 0.7) + ((1.0 - norm_hrv) * 0.3)
        stress_score = max(0, min(1, stress_score)) * 100
        
        calm_score = 100 - stress_score
        
        # Focus: Stability of BPM (Inverse of BPM variance)
        bpm_var = np.var(self.history_bpm) if len(self.history_bpm) > 5 else 10
        focus_score = max(0, min(100, 100 - bpm_var))
        
        # [FIX] Gaze Influence on Focus
        if gaze_label == "Center":
            focus_score = max(80.0, focus_score) # Ensure high focus
        else:
            focus_score = min(40.0, focus_score) # Cap low focus
        
        # [NEW] Calculate Doshas (Nadi Pariksha)
        # Vata (Air): Linked to Variability/Movement -> Proportional to HRV
        vata_score = min(100, norm_hrv * 100)
        
        # Pitta (Fire): Linked to Intensity/Heat -> Proportional to HR
        pitta_score = min(100, norm_bpm * 100)
        
        # Kapha (Water): Linked to Stability/Calm -> Inverse of HR & HRV
        # High Kapha = Slow, steady pulse
        kapha_score = min(100, (1.0 - norm_bpm) * 80 + (1.0 - norm_hrv) * 20)
        
        # Update History
        self.history_vata.append(vata_score)
        self.history_pitta.append(pitta_score)
        self.history_kapha.append(kapha_score)
        
        # Keep history short (20 frames)
        if len(self.history_vata) > 20: self.history_vata.pop(0)
        if len(self.history_pitta) > 20: self.history_pitta.pop(0)
        if len(self.history_kapha) > 20: self.history_kapha.pop(0)
        
        # [NEW] Determine Dominant Dosha & Finding
        doshas = {'Vata': vata_score, 'Pitta': pitta_score, 'Kapha': kapha_score}
        dominant = max(doshas, key=doshas.get)
        
        finding = "Scanning..."
        if len(self.history_bpm) > 5:
            if dominant == 'Vata':
                finding = "Dominant: Vata (High Movement/Anxiety)"
            elif dominant == 'Pitta':
                finding = "Dominant: Pitta (High Energy/Heat)"
            elif dominant == 'Kapha':
                finding = "Dominant: Kapha (High Stability/Lethargy)"
                
            # Check for Balance (if all are close)
            avg_d = sum(doshas.values()) / 3
            if all(abs(v - avg_d) < 15 for v in doshas.values()):
                finding = "Finding: Tridosha Balanced (Excellent)"
        
        # 4. Generate Insight (Every 15 Seconds)
        if now - self.last_insight_time > 15.0:
            self.last_insight_time = now
            
            if len(self.history_bpm) > 5:
                if calm_score > 80:
                    self.current_insight = "Deep state of relaxation detected."
                elif calm_score > 60:
                    self.current_insight = "Heart rhythm is steady and calm."
                elif stress_score > 80:
                    self.current_insight = "High arousal. Focus on slow exhalations."
                elif stress_score > 60:
                    self.current_insight = "Slight tension. Soften your shoulders."
                elif focus_score > 80:
                    self.current_insight = "Excellent physiological coherence."
                else:
                    self.current_insight = "Breathing is syncing with heart rate."
            else:
                self.current_insight = "Scanning bio-rhythms..."

        return {
            "heart_rate": bpm,
            "hrv_rmssd_ms": hrv_rmssd,
            "stress_score": stress_score,
            "calm_score": calm_score,
            "focus_score": focus_score,
            "tiny_graphs": {'vata': self.history_vata, 'pitta': self.history_pitta, 'kapha': self.history_kapha},
            "insight_text": self.current_insight,
            "finding": finding # [NEW]
        }

multi_visualizer = MultiGraphVisualizer()
# pulse_visualizer = PulseWaveVisualizer() # Replaced
# mind_visualizer = MindWaveVisualizer() # Replaced
physio_engine = PhysiologyEngine() # [NEW] Physiology Engine

def draw_mini_bars(frame, x, y, data, color, h=20, w=100):
    if not data: return
    bar_w = w / len(data)
    # [FIX] Use Absolute Scaling (0-100) for "Real" Graph feel
    # Do not auto-scale min/max, otherwise noise looks like huge spikes
    for i, val in enumerate(data):
        norm = max(0, min(1, val / 100.0)) # Assume data is 0-100
        bh = int(norm * h)
        bx = int(x + i * bar_w)
        by = int(y + h - bh)
        cv2.rectangle(frame, (bx, by), (bx + int(bar_w)-1, y + h), color, -1)

def draw_heart_rate_panel(frame, hr_monitor, meditation_stage, posture_score=0.0, avg_energy=0.5, gaze_label="Center"):
    hr, spo2, last_beat, beat_detected, hr_history = hr_monitor.get_data()
    
    h, w, _ = frame.shape
    
    # Premium Large Panel
    panel_w = 350
    panel_h = 720 # Reduced to 720 for smaller screens
    # Moved to LEFT side next to Chakra Meter (x=40) to avoid Mudra Sidebar overlap
    panel_x = 140 
    panel_y = 20 # Moved UP closer to top
    
    if not hr_monitor.connected:
        cv2.putText(frame, "Sensor: Not Connected", (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return

    # [NEW] Physiology Analysis
    physio_metrics = physio_engine.analyze(hr, beat_detected, gaze_label)
    hrv_val = physio_metrics.get('hrv_rmssd_ms', 50.0)

    # Update Visualizer with HRV Index
    multi_visualizer.update(hr, hr_history, spo2, posture_score, beat_detected, avg_energy, hrv_val)
    
    # Panel Background (Premium Dark Glass)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (5, 8, 15), -1) 
    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame) 
    
    # Border (Gold/Cyan Gradient feel)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 215, 255), 2)

    # Header
    cv2.putText(frame, "BIO-ANALYTICS ENGINE", (panel_x + 60, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Main Stats Row
    # Heart
    hx, hy = panel_x + 40, panel_y + 70 # Shifted right for bigger icon
    beat_scale = 1.0
    
    # [FIX] Sync animation with Arduino "BEAT" signal
    # Make it pop: Scale 1.4x for 200ms
    if time.time() - last_beat < 0.20: 
        beat_scale = 1.4 
        
    # Draw Heart Icon (Custom Shape) - MUCH BIGGER & VISIBLE
    # Color shifts to Bright Red when beating
    heart_col = (0, 0, 255) if beat_scale > 1.0 else (0, 0, 200)
    draw_heart(frame, hx, hy - 5, int(28 * beat_scale), heart_col, outline=True)
    
    # HR Value
    cv2.putText(frame, f"{int(hr)}", (hx + 50, hy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    # BPM Label - Moved below for better visibility
    cv2.putText(frame, "BPM", (hx + 55, hy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1)
    
    # SpO2 - Moved DOWN to avoid overlap with BPM
    cv2.putText(frame, f"Oxygen: {int(spo2)}%", (hx, hy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)

    # Graphs - COMPACT MODE
    gy = panel_y + 145 # Pushed DOWN to make room for Oxygen
    gh = 55 # Reduced height slightly
    gap = 10 # Reduced gap slightly
    
    # 1. Heart Rhythm (Red) -> ECG Style
    multi_visualizer.draw_graph(frame, panel_x + 20, gy, panel_w - 40, gh, multi_visualizer.pulse_data, (0, 0, 255), "Heart Rhythm", style="ecg")
    
    # 2. HRV / Stress (Purple) -> EQ Bars
    curr_stress = multi_visualizer.hrv_data[-1]
    stress_status = "Relaxed" if curr_stress < 0.3 else "High"
    multi_visualizer.draw_graph(frame, panel_x + 20, gy + gh + gap, panel_w - 40, gh, multi_visualizer.hrv_data, (255, 0, 255), f"Stress: {stress_status}", style="bars")
    
    # 3. Prana Energy (Gold) -> Double Wave
    curr_prana = multi_visualizer.prana_data[-1]
    prana_status = "High" if curr_prana > 0.6 else "Building"
    multi_visualizer.draw_graph(frame, panel_x + 20, gy + 2*(gh + gap), panel_w - 40, gh, multi_visualizer.prana_data, (0, 215, 255), f"Prana: {prana_status}", style="double_wave")
    
    # 4. Focus (Blue) -> Glow Beam
    curr_focus = multi_visualizer.focus_data[-1]
    focus_status = "Sharp" if curr_focus > 0.6 else "Drifting"
    multi_visualizer.draw_graph(frame, panel_x + 20, gy + 3*(gh + gap), panel_w - 40, gh, multi_visualizer.focus_data, (255, 200, 0), f"Focus: {focus_status}", style="glow_beam")

    # 5. [NEW] HRV Index (Teal) -> Filled Area
    # Use real HRV value for label
    hrv_label = f"HRV Index: {int(hrv_val)} ms"
    multi_visualizer.draw_graph(frame, panel_x + 20, gy + 4*(gh + gap), panel_w - 40, gh, multi_visualizer.hrv_index_data, (200, 255, 200), hrv_label, style="filled_area")

    # --- NEW: Physiology Engine Tiny Graphs & Bot ---
    # [NEW] Nadi Pariksha (Pulse Diagnosis) Tiny Graphs
    tg_y = gy + 5*(gh + gap) + 15
    
    cv2.putText(frame, "Nadi Pariksha (Doshas):", (panel_x + 20, tg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # [FIX] Move Text ABOVE Bars and Add Spacing
    # Vata
    cv2.putText(frame, "Vata", (panel_x + 20, tg_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
    draw_mini_bars(frame, panel_x + 20, tg_y + 25, physio_metrics['tiny_graphs']['vata'], (255, 200, 100), w=60, h=15)
    
    # Pitta
    cv2.putText(frame, "Pitta", (panel_x + 100, tg_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    draw_mini_bars(frame, panel_x + 100, tg_y + 25, physio_metrics['tiny_graphs']['pitta'], (0, 0, 255), w=60, h=15)
    
    # Kapha
    cv2.putText(frame, "Kapha", (panel_x + 180, tg_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    draw_mini_bars(frame, panel_x + 180, tg_y + 25, physio_metrics['tiny_graphs']['kapha'], (0, 255, 0), w=60, h=15)

    # [NEW] Findings Guide
    finding_text = physio_metrics.get('finding', "Scanning...")
    cv2.putText(frame, finding_text, (panel_x + 20, tg_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 255), 1)

    # [NEW] Horizontal Guide Box (Bottom of Panel)
    guide_y = tg_y + 65
    # Removed Legend as per user request

    # [NEW] Advanced Premium Graph: "Energy Coherence" (Circular Radar)
    # CONNECTED TO ENERGY: Size & Color changes with avg_energy
    
    coherence_y = guide_y + 45 
    coherence_x = panel_x + panel_w // 2
    
    # Base radius + Energy expansion
    # avg_energy is 0.0 to 1.0 (usually)
    # Radius: 20 (min) to 50 (max)
    radius = int(20 + avg_energy * 30)
    
    # Color Logic: Blue (Low) -> Gold (Med) -> White/Purple (High)
    if avg_energy < 0.3:
        coh_color = (255, 0, 0) # Blue
    elif avg_energy < 0.7:
        coh_color = (0, 215, 255) # Gold
    else:
        coh_color = (255, 0, 255) # Purple/Whiteish
        
    # Draw Radar Background (Static)
    cv2.circle(frame, (coherence_x, coherence_y), 50, (30, 30, 30), 1)
    cv2.line(frame, (coherence_x - 50, coherence_y), (coherence_x + 50, coherence_y), (30, 30, 30), 1)
    cv2.line(frame, (coherence_x, coherence_y - 50), (coherence_x, coherence_y + 50), (30, 30, 30), 1)
    
    # Dynamic Coherence Shape
    pts = []
    num_pts = 36
    
    for i in range(num_pts):
        angle = math.radians(i * (360/num_pts))
        # Pulse factor
        pulse = 1.0 + 0.05 * math.sin(time.time() * 4 + i * 0.5)
        
        r = radius * pulse
        px = int(coherence_x + r * math.cos(angle))
        py = int(coherence_y + r * math.sin(angle))
        pts.append([px, py])
        
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Draw Filled Coherence Blob
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], coh_color)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Label with Value
    energy_pct = int(avg_energy * 100)
    cv2.putText(frame, f"Energy Coherence: {energy_pct}%", (coherence_x - 60, coherence_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # [NEW] Data Analysis Bot
    # Position relative to Coherence Graph to ensure visibility
    # [FIX] Position relative to Coherence Graph (y ~680)
    bot_y = coherence_y + radius + 30
    bot_x = panel_x + 50 # Moved right slightly
    
    # Draw Bot Icon (Animated)
    t = time.time()
    
    # Blinking Logic (Every 3s for 0.15s)
    is_blink = (t % 3.0) > 2.85
    
    # Look Direction (Cycle: Center -> Left -> Center -> Right)
    look_cycle = t % 8.0
    eye_dx = 0
    if 2.0 < look_cycle < 3.5: eye_dx = -4 # Look Left
    elif 5.0 < look_cycle < 6.5: eye_dx = 4 # Look Right
    
    # Head
    # [FIX] Larger Head (Radius 25)
    cv2.circle(frame, (bot_x, bot_y), 25, (50, 50, 50), -1) # Head Background
    cv2.circle(frame, (bot_x, bot_y), 25, (0, 255, 255), 2) # Head Outline
    
    if is_blink:
        # Closed Eyes (Lines)
        cv2.line(frame, (bot_x-12, bot_y-4), (bot_x-4, bot_y-4), (0, 255, 255), 2)
        cv2.line(frame, (bot_x+4, bot_y-4), (bot_x+12, bot_y-4), (0, 255, 255), 2)
    else:
        # Open Eyes (Circles with pupils)
        # Whites
        cv2.circle(frame, (bot_x-8, bot_y-4), 7, (255, 255, 255), -1)
        cv2.circle(frame, (bot_x+8, bot_y-4), 7, (255, 255, 255), -1)
        # Pupils (Moving)
        cv2.circle(frame, (bot_x-8+eye_dx, bot_y-4), 3, (0, 0, 0), -1)
        cv2.circle(frame, (bot_x+8+eye_dx, bot_y-4), 3, (0, 0, 0), -1)

    # Smile (Curve)
    cv2.ellipse(frame, (bot_x, bot_y+5), (11, 7), 0, 20, 160, (0, 255, 255), 2)
    
    # Dynamic Typing Text
    advice = physio_metrics['insight_text']
    # Simple typing effect based on time
    typing_speed = 0.05
    # Cycle every few seconds
    cycle_duration = len(advice) * typing_speed + 3.0
    char_count = int((time.time() % cycle_duration) / typing_speed)
    current_text = advice[:char_count]
    
    # Draw Text Bubble Background
    # [FIX] Larger Font for Bot Text
    bot_font_scale = 0.6
    bot_thickness = 2
    (tw, th), _ = cv2.getTextSize(current_text, cv2.FONT_HERSHEY_SIMPLEX, bot_font_scale, bot_thickness)
    if tw > 0:
        # [FIX] Move text ABOVE bot head to avoid graph overlap
        bx = bot_x - tw // 2
        by = bot_y - 45
        # Ensure within screen
        if bx < 10: bx = 10
        if by < 10: by = 10
        
        # Bubble Box
        cv2.rectangle(frame, (bx - 5, by - th - 5), (bx + tw + 5, by + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (bx - 5, by - th - 5), (bx + tw + 5, by + 5), (0, 255, 255), 1)
        
        # Text
        cv2.putText(frame, current_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, bot_font_scale, (255, 255, 255), bot_thickness)

# Load Om Image
OM_IMG = None
if os.path.exists("om_glow.png"):
    OM_IMG = cv2.imread("om_glow.png", cv2.IMREAD_UNCHANGED)
    # Ensure it has alpha channel
    if OM_IMG is not None and OM_IMG.shape[2] != 4:
        # If no alpha, assume black is transparent (fallback)
        tmp = cv2.cvtColor(OM_IMG, cv2.COLOR_BGR2BGRA)
        OM_IMG = tmp
else:
    print("[WARN] om_glow.png not found. Run generate_om.py first.")

class OmParticleSystem:
    def __init__(self):
        self.particles = []
        
    def update(self, w, h, energy_level):
        # Spawn new particles if energy is high
        # [FIX] Only 4-5 particles, larger area, spreading
        if energy_level > 0.9 and len(self.particles) < 5:
            if random.random() < 0.1: # Slower spawn rate for fewer particles
                spawn_x = w // 2 + random.randint(-150, 150) # Larger area
                # Velocity spreads outwards from center
                vx = (spawn_x - w // 2) * 0.015 
                
                self.particles.append({
                    'x': spawn_x,
                    'y': h // 2 + random.randint(-50, 100),
                    'vy': -0.5 - random.random() * 1.0, # Float up slower
                    'vx': vx,
                    'size': random.randint(40, 70), # Larger for image
                    'alpha': 1.0
                })
                
        # Update existing
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['alpha'] -= 0.015 # Fade out
            
        # Remove dead
        self.particles = [p for p in self.particles if p['alpha'] > 0]

    def draw(self, frame):
        for p in self.particles:
            overlay = frame.copy()
            # Always use procedural drawing for reliability (No boxes!)
            self.draw_om_shape(overlay, int(p['x']), int(p['y']), p['size'])
            cv2.addWeighted(overlay, p['alpha'], frame, 1 - p['alpha'], 0, frame)

    def draw_om_shape(self, img, x, y, s):
        # Procedural Golden Glowing Om (Reliable & Beautiful)
        thickness = max(1, int(s / 12))
        
        # Define relative points (0.0 to 1.0)
        # Top Curve (Upper 3)
        pts_top = np.array([
            [x - s*0.2, y - s*0.1],
            [x + s*0.1, y - s*0.3],
            [x + s*0.3, y - s*0.1],
            [x + s*0.1, y + s*0.05]
        ], np.int32)
        
        # Bottom Curve (Lower 3)
        pts_bot = np.array([
            [x + s*0.1, y + s*0.05],
            [x + s*0.3, y + s*0.2],
            [x + s*0.1, y + s*0.4],
            [x - s*0.2, y + s*0.3]
        ], np.int32)
        
        # Tail
        pts_tail = np.array([
            [x + s*0.1, y + s*0.05],
            [x + s*0.4, y + s*0.05],
            [x + s*0.5, y + s*0.2]
        ], np.int32)
        
        # Crescent
        pts_cres = np.array([
            [x + s*0.3, y - s*0.35],
            [x + s*0.4, y - s*0.45],
            [x + s*0.5, y - s*0.35]
        ], np.int32)
        
        # Colors
        glow_color = (0, 165, 255) # Orange (BGR)
        gold_color = (0, 215, 255) # Gold (BGR)
        white_color = (255, 255, 255)
        
        # Layer 1: Outer Glow (Thick Orange)
        t1 = thickness + 6
        for pts in [pts_top, pts_bot, pts_tail, pts_cres]:
            cv2.polylines(img, [pts], False, glow_color, t1, cv2.LINE_AA)
        cv2.circle(img, (int(x + s*0.4), int(y - s*0.5)), max(2, thickness)+4, glow_color, -1)

        # Layer 2: Gold Body (Medium)
        t2 = thickness + 2
        for pts in [pts_top, pts_bot, pts_tail, pts_cres]:
            cv2.polylines(img, [pts], False, gold_color, t2, cv2.LINE_AA)
        cv2.circle(img, (int(x + s*0.4), int(y - s*0.5)), max(2, thickness)+1, gold_color, -1)
        
        # Layer 3: White Core (Thin)
        t3 = max(1, thickness - 1)
        for pts in [pts_top, pts_bot, pts_tail, pts_cres]:
            cv2.polylines(img, [pts], False, white_color, t3, cv2.LINE_AA)
        cv2.circle(img, (int(x + s*0.4), int(y - s*0.5)), max(1, thickness-1), white_color, -1)

def overlay_image_alpha(img, img_overlay, x, y, alpha_mult=1.0):
    """Overlays an RGBA image onto a BGR image with alpha blending."""
    h, w, _ = img.shape
    oh, ow, _ = img_overlay.shape
    
    # Crop if out of bounds
    if x < 0: 
        ow += x
        img_overlay = img_overlay[:, -x:]
        x = 0
    if y < 0:
        oh += y
        img_overlay = img_overlay[-y:, :]
        y = 0
    if x + ow > w:
        ow = w - x
        img_overlay = img_overlay[:, :ow]
    if y + oh > h:
        oh = h - y
        img_overlay = img_overlay[:oh, :]
        
    if ow <= 0 or oh <= 0: return

    # ROI
    roi = img[y:y+oh, x:x+ow]
    
    # Alpha channel
    alpha = img_overlay[:, :, 3] / 255.0
    alpha = alpha * alpha_mult # Apply particle alpha
    
    # Colors
    color = img_overlay[:, :, :3]
    
    # Blend
    for c in range(0, 3):
        roi[:, :, c] = (alpha * color[:, :, c] + (1.0 - alpha) * roi[:, :, c])
        
    img[y:y+oh, x:x+ow] = roi

om_particles = OmParticleSystem()

def draw_om_effect(frame, energy_level):
    """
    Updates and draws the Om Particle System.
    """
    h, w, _ = frame.shape
    om_particles.update(w, h, energy_level)
    om_particles.draw(frame)


class MeditationTracker:
    def __init__(self):
        self.stage = "Dharana (Concentration)"
        self.concentration_level = 0.0
        self.start_time = 0
        self.in_dhyana = False
        
    def update(self, eye_open, breath_stable, body_still, gaze_label="Center"):
        # Logic: 
        # 1. Eyes Closed -> Jump to 100%
        # 2. Eyes Open + Center Gaze -> Max 50%
        # 3. Eyes Open + Looking Away -> Drop to 0%
        
        if eye_open < EYE_CLOSED_THRESHOLD: # Eyes closed (Using Ratio now)
            if not self.in_dhyana:
                self.in_dhyana = True
                self.start_time = time.time()
                self.stage = "Dhyana (Meditation)"
            
            # Fast increase to 100%
            self.concentration_level = min(100.0, self.concentration_level + 1.0) # Faster gain (was 0.5)
            
            if self.concentration_level > 90 and breath_stable:
                self.stage = "Samadhi (Absorption)"
            elif self.concentration_level > 50:
                self.stage = "Dhyana (Meditation)"
            else:
                self.stage = "Dhyana (Entering...)"
                
        else: # Eyes Open
            self.in_dhyana = False
            
            if gaze_label == "Center":
                # Looking at camera -> Increase to 90% (Max for eyes open)
                # [FIX] Cap at 90% if eyes open. Must close eyes for 100%.
                if self.concentration_level < 90.0:
                    self.concentration_level = min(90.0, self.concentration_level + 0.5) 
                
                # [FIX] Accurate Staging
                if self.concentration_level > 80:
                    self.stage = "Dharana (Deep Focus)"
                elif self.concentration_level > 20:
                    self.stage = "Dharana (Focus)"
                else:
                    self.stage = "Focusing..."
            else:
                # Looking away -> Distracted
                self.concentration_level = max(0.0, self.concentration_level - 0.5) # Slower drop (was 1.0)
                self.stage = "Distracted"
            
        return self.stage, self.concentration_level

# 🔊 Bilingual (Hindi + English) voice summary
def speak_summary(chakra_energies, total_gyan_count, alignment_count, duration_min):
    strongest_idx = int(np.argmax(chakra_energies))
    weakest_idx = int(np.argmin(chakra_energies))

    strongest = CHAKRA_NAMES[strongest_idx]
    weakest = CHAKRA_NAMES[weakest_idx]

    msg = (
        f"Namaste Aditya. Aapka Yoga session ka chhota sa summary ready hai. "
        f"Total session time approximately {duration_min:.1f} minutes raha. "
        f"Is dauraan sabse zyaada active aur powerful chakra tha {strongest}. "
        f"Jo chakra thoda weak side par tha, woh tha {weakest}. "
        f"Aapne Gyaan Mudra total {total_gyan_count} baar kiya. "
        f"Alignment Mode {alignment_count} times activate hua, jab aapki saanse "
        f"aur body dono ekdum shaant state mein aa gayi. "
        f"Keep breathing deeply, dil halka rakhiye, aur apni energy balanced rakhiye. "
        f"Hari Om."
    )

    tts.say(msg)
    tts.runAndWait()


def create_summary_image(chakra_energies, duration_min, total_gyan_count, alignment_count):
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (15, 15, 15)
    cv2.putText(img, "AI ChakraFlow Summary", (170, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    y = 150
    for i, energy in enumerate(chakra_energies):
        cv2.putText(img, f"{CHAKRA_NAMES[i]}: {int(energy*100)}%",
                    (80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CHAKRA_COLORS[i], 2)
        y += 45
    cv2.putText(img, f"Session Time: {duration_min:.1f} min", (80, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
    cv2.putText(img, f"Gyan Mudra: {total_gyan_count}", (80, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
    cv2.putText(img, f"Alignment Mode: {alignment_count}", (80, 520),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 150), 2)
    cv2.imwrite("summary_output.png", img)
    cv2.imwrite("summary_output.png", img)
    return img


def draw_meditation_info_panel(frame, is_meditating, is_eyes_closed, energy_level):
    h, w, _ = frame.shape
    # [FIX] Relocated inside Bio-Analytics Panel (Beside BPM/Oxygen)
    # Panel starts at x=140, y=20
    # BPM/Oxygen take up left side (~100px)
    # We place this to the right: x = 140 + 210 = 350
    # Moved DOWN to y=100 to avoid Header overlap
    x = 350
    y = 100
    
    # Compact Text
    med_col = (0, 255, 255) 
    cv2.putText(frame, f"Meditation: {'ON' if is_meditating else 'OFF'}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, med_col, 1)
    cv2.putText(frame, f"Eyes: {'CLOSED' if is_eyes_closed else 'OPEN'}", (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, med_col, 1)
    cv2.putText(frame, f"Energy: {int(energy_level * 100)}%", (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, med_col, 1)



def show_chakra_bar_graph(chakra_energies):
    graph = np.zeros((500, 800, 3), dtype=np.uint8)
    for i, energy in enumerate(chakra_energies):
        bar_h = int(energy * 400)
        x1 = 80 + i * 100
        y1 = 450 - bar_h
        x2 = x1 + 70
        y2 = 450
        cv2.rectangle(graph, (x1, y1), (x2, y2), CHAKRA_COLORS[i], -1)
        cv2.putText(graph, CHAKRA_NAMES[i].split()[0], (x1, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Chakra Energy Bar Graph", graph)
    cv2.imwrite("chakra_graph.png", graph)


def draw_indian_flag(frame, x, y, width, t):
    """
    Draws a waving, glowing Indian flag with text.
    """
    height = int(width * 0.6)
    
    # Colors (BGR)
    saffron = (0, 153, 255) 
    white = (255, 255, 255)
    green = (19, 136, 19)
    blue = (100, 0, 0) # Navy Blue
    
    # Waving Effect using vertical slices
    seg_w = 2
    for i in range(0, width, seg_w):
        # Sine wave offset
        shift = math.sin(t * 4 + i * 0.1) * 4
        
        # Draw Slices
        # Saffron
        cv2.rectangle(frame, (x+i, int(y+shift)), (x+i+seg_w, int(y+height/3+shift)), saffron, -1)
        # White
        cv2.rectangle(frame, (x+i, int(y+height/3+shift)), (x+i+seg_w, int(y+2*height/3+shift)), white, -1)
        # Green
        cv2.rectangle(frame, (x+i, int(y+2*height/3+shift)), (x+i+seg_w, int(y+height+shift)), green, -1)
        
        # Glow (Add weighted) - Subtle
        if i % 4 == 0:
            overlay = frame.copy()
            cv2.circle(overlay, (x+i, int(y+height/2+shift)), 5, (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)

    # Ashoka Chakra (Center)
    # Approximate center of the wave
    center_shift = math.sin(t * 4 + width/2 * 0.1) * 4
    cx, cy = int(x + width/2), int(y + height/2 + center_shift)
    r = int(height * 0.14)
    cv2.circle(frame, (cx, cy), r, blue, 1)
    # Spokes
    for k in range(0, 360, 15):
        rad = math.radians(k)
        x2 = int(cx + r * math.cos(rad))
        y2 = int(cy + r * math.sin(rad))
        cv2.line(frame, (cx, cy), (x2, y2), blue, 1)

    # Text - Enhanced Visibility
    text = "Proudly Made in India"
    # Calculate text size for centering
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tx = x + (width // 2) - (tw // 2) # Center horizontally
    ty = y + height + 25 # [FIX] Moved down further (was +15)
    
    # Triple Stroke for Max Visibility
    # 1. Black Shadow
    cv2.putText(frame, text, (tx, ty), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 4, cv2.LINE_AA)
    # 2. White Outline
    cv2.putText(frame, text, (tx, ty), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    # 3. Gold Core
    cv2.putText(frame, text, (tx, ty), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)


def generate_aura_photo(frame, aura_color, avg_hr, focus_level):
    """
    Generates a souvenir photo with aura glow and stats.
    """
    h, w, _ = frame.shape
    souvenir = frame.copy()
    
    # 1. Apply Aura Glow (Vignette)
    # Create a radial gradient mask
    Y, X = np.ogrid[:h, :w]
    center = (h//2, w//2)
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = 1 - np.clip(dist_from_center / (w * 0.6), 0, 1) # 1 at center, 0 at edges
    
    # Apply color tint to edges (inverse of mask)
    glow_layer = np.full((h, w, 3), aura_color, dtype=np.uint8)
    
    # Blend: Center is original, Edges are tinted
    # We want edges to be glowing with aura_color
    # So alpha for glow_layer should be high at edges (low mask)
    alpha = (1 - mask) * 0.6 # Max 60% tint at edges
    alpha = np.dstack((alpha, alpha, alpha))
    
    souvenir = (souvenir * (1 - alpha) + glow_layer * alpha).astype(np.uint8)
    
    # 2. Add Border
    cv2.rectangle(souvenir, (0, 0), (w, h), aura_color, 20)
    
    # 3. Add Text Stats
    # Bottom Bar
    bar_h = 120
    cv2.rectangle(souvenir, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    
    # Title
    cv2.putText(souvenir, "YOGA AI SOUVENIR", (w//2 - 150, h - 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Date
    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cv2.putText(souvenir, date_str, (w - 250, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
    # Stats
    stats_text = f"Avg HR: {int(avg_hr)} BPM   |   Focus: {int(focus_level * 100)}%"
    cv2.putText(souvenir, stats_text, (50, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, aura_color, 2)
                
    # Save
    filename = f"aura_souvenir_{int(time.time())}.png"
    cv2.imwrite(filename, souvenir)
    print(f"[INFO] Aura Souvenir saved: {filename}")
    
    # Show briefly (optional, but main loop is ending)
    cv2.imshow("Aura Souvenir", souvenir)
    cv2.waitKey(2000) # Show for 2 seconds before closing


def generate_aura_photo(frame, aura_color, avg_hr, focus_level):
    """
    Generates a souvenir photo with aura glow and stats.
    Saves in 'screenshots' folder with timestamp.
    """
    h, w, _ = frame.shape
    souvenir = frame.copy()
    
    # 1. Apply Aura Glow (Vignette)
    # Create a radial gradient mask
    Y, X = np.ogrid[:h, :w]
    center = (h//2, w//2)
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = 1 - np.clip(dist_from_center / (w * 0.6), 0, 1) # 1 at center, 0 at edges
    
    # Apply color tint to edges (inverse of mask)
    glow_layer = np.full((h, w, 3), aura_color, dtype=np.uint8)
    
    # Blend: Center is original, Edges are tinted
    # We want edges to be glowing with aura_color
    # So alpha for glow_layer should be high at edges (low mask)
    alpha = (1 - mask) * 0.6 # Max 60% tint at edges
    alpha = np.dstack((alpha, alpha, alpha))
    
    souvenir = (souvenir * (1 - alpha) + glow_layer * alpha).astype(np.uint8)
    
    # 2. Add Border
    cv2.rectangle(souvenir, (0, 0), (w, h), aura_color, 20)
    
    # 3. Add Text Stats
    # Bottom Bar
    bar_h = 120
    cv2.rectangle(souvenir, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    
    # Title
    cv2.putText(souvenir, "YOGA AI SOUVENIR", (w//2 - 150, h - 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Date
    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cv2.putText(souvenir, date_str, (w - 250, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
    # Stats
    stats_text = f"Avg HR: {int(avg_hr)} BPM   |   Focus: {int(focus_level * 100)}%"
    cv2.putText(souvenir, stats_text, (50, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, aura_color, 2)
                
    # Save to Screenshots Folder
    screenshot_dir = "screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    
    # Better filename with readable timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(screenshot_dir, f"yoga_session_{timestamp}.png")
    cv2.imwrite(filename, souvenir)
    print(f"[INFO] 📸 Screenshot saved: {filename}")
    
    return filename  # Return filename for confirmation


def show_final_report(session_start, chakra_energies, total_gyan_count, alignment_count):
    end_time = time.time()
    duration_min = (end_time - session_start) / 60.0
    strongest_idx = int(np.argmax(chakra_energies))
    weakest_idx = int(np.argmin(chakra_energies))
    strongest = CHAKRA_NAMES[strongest_idx]
    weakest = CHAKRA_NAMES[weakest_idx]
    calmness_score = int((chakra_energies[3] + chakra_energies[6]) / 2 * 100)

    print("\n=========================")
    print("     FINAL YOGA REPORT")
    print("=========================\n")
    print(f"[Session] Duration: {duration_min:.2f} minutes")
    print(f"[Strongest] Chakra: {strongest}")
    print(f"[Weakest] Chakra: {weakest}")
    print(f"\n[Gyan Mudra] Activations: {total_gyan_count} times")
    print(f"[Alignment Mode] Entered: {alignment_count} times")
    print("\n[Energy Levels] Final Chakra:")
    for i, energy in enumerate(chakra_energies):
        print(f"   {CHAKRA_NAMES[i]}: {int(energy*100)}%")
    print(f"\n[Breath] Calmness Score: {calmness_score}/100")
    print("\n[Namaste] Thank you for practicing with AI ChakraFlow.")
    print("   Keep breathing. Stay mindful. Namaste.\n")


# ======================== MAIN ===========================

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    # Background music
    if os.path.exists(MUSIC_PATH):
        try:
            pygame.mixer.music.load(MUSIC_PATH)
            pygame.mixer.music.set_volume(0.8)
            pygame.mixer.music.play(-1)
            print("[INFO] Playing background music: Adiyogi")
        except Exception as e:
            print("[WARN] Could not play background music:", e)
    else:
        print("[WARN] Music file not found:", MUSIC_PATH)

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5, # Lowered for faster/easier detection
        min_tracking_confidence=0.5
    )
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # Enabled for Gaze Tracking
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0  # lightweight model to reduce lag
    )

    chakra_energies = [0.4] * 7
    last_chakra_index = None
    last_activation_time = time.time()
    breathing = BreathingTracker()

    session_start = time.time()
    eye_closed_frames = 0
    alignment_mode = False
    alignment_start_time = 0.0
    alignment_progress = 0.0
    gyan_active = False
    total_gyan_count = 0
    alignment_count = 0
    alignment_count = 0
    last_gyan_state = False
    
    # Smart Yoga Mode State
    yoga_mode_active = False
    med_level = 0.0
    namaste_hold_start = 0
    namaste_triggered = False
    namaste_grace_frames = 0 # Grace period for flickering detection
    
    # Awakening Sequence State
    was_eyes_closed = False
    eyes_closed_start = 0
    awakening_active = False
    awakening_active = False
    awakening_start = 0
    
    # Screenshot Gesture State
    screenshot_timer = 0
    screenshot_countdown_start = 0
    screenshot_triggered = False
    
    # Animation Time Tracking
    anim_time = 0.0
    last_frame_time = time.time()

    # AI explainer state
    ai_text = "Breathe easy. Hold a gesture to get a short tip."
    last_ai_chakra = None
    last_ai_time = 0
    last_activation_time = 0 # [FIX] Initialize for sticky energy logic

    posture_analyzer = PostureAnalyzer()
    analytics = AnalyticsTracker()
    meditation_tracker = MeditationTracker()
    hr_monitor = HeartRateMonitor() # Initialize Heart Rate Monitor

    # Voice recognizer (heavy) can be disabled if laggy
    ENABLE_VOICE = False
    if ENABLE_VOICE:
        r = sr.Recognizer()
        mic = sr.Microphone()
    last_voice_check = 0

    # Frame throttling for pose to reduce lag
    pose_every_n = 2
    frame_count = 0
    last_pose_landmarks = None

    # [NEW] Gamification System
    # current_level = 0 # Replaced by XP system
    max_level_session = 0
    level_unlock_time = 0
    gift_active = False
    
    # [NEW] XP System (20 Levels)
    total_xp = 0.0
    current_level = 1
    XP_PER_LEVEL = 150 # Approx 5 seconds per level at base rate (30fps)
    MAX_LEVEL = 20

    print("[INFO] AI ChakraFlow FULL started. Press 'q' to quit.")

    cv2.namedWindow("AI ChakraFlow — Full Experience", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("AI ChakraFlow — Full Experience", mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # [FIX] Update Heart Rate Monitor (Read Serial Data)
        hr_monitor.update()

        hand_res = hands.process(rgb)
        face_res = face_mesh.process(rgb)
        frame_count += 1
        if frame_count % pose_every_n == 0:
            pose_res = pose.process(rgb)
            last_pose_landmarks = pose_res.pose_landmarks if pose_res else None
        else:
            pose_res = None

        center_x = w // 2
        top_y = int(h * 0.25)
        bottom_y = int(h * 0.85)

        active_chakra_idx = None
        aura_color = (255, 255, 255)
        mood_label = "Scanning..." # Changed from "..."
        posture_score = 0.0
        posture_label = "No body"
        is_eyes_closed = False
        eye_open = 1.0
        mouth_open = 0.0
        
        # [FIX] Calculate Avg Energy EARLY to avoid NameError
        avg_energy = sum(chakra_energies) / len(chakra_energies)
        mood_label = "Scanning..." # Changed from "..."
        posture_score = 0.0
        posture_label = "No body"
        is_eyes_closed = False
        eye_open = 1.0
        mouth_open = 0.0
        gaze_label = "Center" # Default value
        gaze_x = 0.0 # Default value

        if face_res.multi_face_landmarks:
            face_landmarks = face_res.multi_face_landmarks[0]
            aura_color, mood_label, eye_open, mouth_open, gaze_label, gaze_x = analyze_face(face_landmarks, w, h)
            
            nose = face_landmarks.landmark[1]
            nose_y = nose.y
            breathing.update(nose_y)
            center_x = int(nose.x * w)
            
            # [NEW] Third Eye Interface (Siddhi Mode)
            # Conditions: Energy 100%, Meditation ON, Concentration 100%, Eyes CLOSED
            # Note: med_level is updated above in yoga_mode_active block. 
            # If yoga_mode is NOT active, we need to ensure we have these values or skip.
            # Assuming Yoga Mode is active for this advanced feature.
            
            is_siddhi_ready = False
            if yoga_mode_active:
                is_siddhi_ready = (avg_energy >= 0.99 and 
                                   meditation_tracker.in_dhyana and 
                                   med_level >= 99.0 and 
                                   is_eyes_closed)
            
            if is_siddhi_ready:
                # Use Head Yaw for Control (Nose X)
                # center_x is nose.x * w
                # Normalize to -1.0 to 1.0
                # Center is w/2
                # Deviation = center_x - w/2
                # Range approx +/- 150px for head turn? Let's try 0.2 * w
                head_yaw = (center_x - w/2) / (w * 0.2) 
                # Clamp
                head_yaw = max(-1.0, min(1.0, head_yaw))
                
                # Draw Beam
                third_eye.update_and_draw(frame, face_res.multi_face_landmarks[0], head_yaw, chakra_energies)
                
                # Visual Indicator for Siddhi Mode
                cv2.putText(frame, "SIDDHI MODE ACTIVE", (w//2 - 150, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 215, 0), 2)

            # --- Awakening Trigger Logic ---
            is_eyes_closed = (eye_open < EYE_CLOSED_THRESHOLD)
            
            if is_eyes_closed:
                if not was_eyes_closed:
                    eyes_closed_start = time.time()
                was_eyes_closed = True
                eye_closed_frames += 1
                try:
                    draw_text_with_bg(frame, "Meditation Detector: Eyes Closed", center_x - 120, center_y - 50, color=(255, 255, 0))
                except Exception:
                    pass
            else:
                if was_eyes_closed:
                    duration_closed = time.time() - eyes_closed_start
                    if duration_closed > 4.0:
                        awakening_active = True
                        awakening_start = time.time()
                        print("[INFO] Chakra Awakening Sequence Started!")
                was_eyes_closed = False
                eye_closed_frames = 0

            if eye_closed_frames > EYE_CLOSED_FRAMES_REQUIRED and not alignment_mode:
                alignment_mode = True
                alignment_count += 1
                alignment_start_time = time.time()
                alignment_progress = 0.0
                print("[INFO] Alignment Mode activated.")
        else:
            breathing.update(0.5)
            eye_closed_frames = 0
            was_eyes_closed = False

        # ALIGNMENT (Updated to allow rising)
        if alignment_mode:
            alignment_progress = min(1.0, alignment_progress + 0.01)
            # Instead of clamping, we just ensure a minimum baseline that rises
            # This allows the meditation boost (later in code) to add on top!
            target_base = 1.0 * alignment_progress # Target 100%
            for i in range(len(chakra_energies)):
                # Only pull up if below target, don't pull down if already high
                # Use max to ensure we never drag it down
                chakra_energies[i] = max(chakra_energies[i], target_base)
            
            aura_color = (0, 215, 255)
            if time.time() - alignment_start_time > 8:
                alignment_mode = False
                print("[INFO] Alignment Mode ended.")

        # POSE (throttled)
        pose_landmarks = pose_res.pose_landmarks if pose_res and pose_res.pose_landmarks else last_pose_landmarks
        if pose_landmarks:
            posture_score, posture_label = posture_analyzer.assess(pose_landmarks)
            analytics.record_posture(posture_score)
        else:
            posture_score, posture_label = 0.0, "No body"

        # [FIX] Update Meditation Tracker HERE (Before Energy Logic)
        # Ensure eye_open is defined (default 1.0 if no face)
        if 'eye_open' not in locals(): eye_open = 1.0
        if 'gaze_label' not in locals(): gaze_label = "Center"
        
        med_stage, med_level = meditation_tracker.update(eye_open, True, posture_score > 0.6, gaze_label)

        breath_factor = breathing.get_breath_factor()
        
        # Update Heart Rate
        hr_monitor.update()
        current_avg_energy = sum(chakra_energies) / len(chakra_energies)
        draw_heart_rate_panel(frame, hr_monitor, meditation_tracker.stage, posture_score, current_avg_energy, gaze_label)

        # Fallback Mood Logic (If Face not detected)
        if mood_label == "Scanning..." or mood_label == "No face":
            hr_val, _, _, _, _ = hr_monitor.get_data()
            if hr_val > 0:
                if hr_val < 65: mood_label = "Deeply Relaxed"
                elif hr_val < 85: mood_label = "Calm & Balanced"
                elif hr_val < 110: mood_label = "Active / Focused"
                else: mood_label = "High Energy"

        # --- Hand Analysis & Mudra Detection ---
        detected_mudra = None
        detected_mudra_name = None
        gyan_active = False
        
        if hand_res.multi_hand_landmarks:
            # Debug info - Moved to Bottom Left to avoid overlap
            cv2.putText(frame, f"Hands: {len(hand_res.multi_hand_landmarks)}", (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # Check for Namaste (Anjali) first
            if detect_namaste(hand_res):
                detected_mudra_name = "Anjali Mudra"
                detected_mudra = 4 
            else:
                for hand_landmarks in hand_res.multi_hand_landmarks:
                    finger_states = get_finger_states(hand_landmarks, w, h)
                    
                    # Check specific mudras
                    if detect_gyan_mudra(hand_landmarks, frame, w, h):
                        detected_mudra_name = "Gyan Mudra"
                        detected_mudra = 6 
                    elif detect_prana_mudra(hand_landmarks):
                        detected_mudra_name = "Prana Mudra"
                        detected_mudra = 0 
                    elif detect_apana_mudra(hand_landmarks):
                        detected_mudra_name = "Apana Mudra"
                        detected_mudra = 1 
                    elif detect_surya_mudra(hand_landmarks):
                        detected_mudra_name = "Surya Mudra"
                        detected_mudra = 2 
                    elif detect_varun_mudra(hand_landmarks):
                        detected_mudra_name = "Varun Mudra"
                        detected_mudra = 1 
                    
                    if detected_mudra_name:
                        break 

            # Record analytics
            if detected_mudra_name:
                 analytics.record_mudra(detected_mudra_name)
                 if detected_mudra_name == "Gyan Mudra":
                     total_gyan_count += 1 
                     gyan_active = True

        # [NEW] Gaze Detection for Distraction
        gaze_distracted = False
        if face_res and face_res.multi_face_landmarks:
            fl = face_res.multi_face_landmarks[0]
            # Nose Tip: 1, Left Eye Outer: 33, Right Eye Outer: 263
            nose_x = fl.landmark[1].x
            left_eye_x = fl.landmark[33].x
            right_eye_x = fl.landmark[263].x
            
            eye_mid_x = (left_eye_x + right_eye_x) / 2
            gaze_diff = nose_x - eye_mid_x
            
            # Threshold for looking left/right (Distracted)
            if abs(gaze_diff) > 0.04: 
                gaze_distracted = True
                gaze_label = "Distracted"

        # [FIX] STRICT ENERGY LIMITS
        # [FIX] STRICT ENERGY LIMITS (User Request)
        # 1. Default / Bad Posture -> 10%
        energy_limit = 0.10 
        
        # 2. Good Posture -> 30%
        if posture_score > 0.60: 
            energy_limit = 0.30
            
        # 3. Mudra -> 60%
        if detected_mudra_name:
            energy_limit = 0.60
            
        # 4. Eyes Closed -> 100%
        if is_eyes_closed:
            energy_limit = 1.0 
            if 'eyes_closed_start' in locals():
                duration_closed = time.time() - eyes_closed_start
                if duration_closed > 30.0:
                    energy_limit = 1.0 
        
        # [FIX] Distraction Penalty (Overrides everything)
        # "eedhar udhar dekhne pe... energy ghate gii"
        if gaze_distracted:
            energy_limit = 0.10 # Cap at 10% if looking away

        # [FIX] Sticky Energy Limit (1.5s grace period)
        # Prevents energy limit dropping instantly if detection flickers
        if detected_mudra_name or (time.time() - last_activation_time < 1.5):
             if energy_limit < 0.60 and not gaze_distracted: # Don't hold if distracted
                 energy_limit = 0.60

        if detected_mudra is not None and not gaze_distracted:
            analytics.record_chakra(detected_mudra)
            active_chakra_idx = detected_mudra
            last_activation_time = time.time()
            for i in range(len(chakra_energies)):
                if i == detected_mudra:
                    # Active Mudra Boost - INSTANT
                    base_increase = 0.20 
                    if is_eyes_closed or med_level > 40:
                        base_increase = 0.25 
                    
                    # Apply Limit
                    chakra_energies[i] = min(energy_limit, chakra_energies[i] + base_increase) 
                else:
                    # Holistic Boost
                    target_other = energy_limit * 0.85 
                    if chakra_energies[i] < target_other:
                        chakra_energies[i] += 0.05 
                    else:
                        # Normal decay handled by Global Enforcer
                        pass
                        
        # [FIX] Relaxed Condition: Eyes Closed OR Good Posture
        elif pose_landmarks and (posture_score > 0.5 or is_eyes_closed) and not gaze_distracted:
            # Passive Body Boost
            for i in range(len(chakra_energies)):
                if chakra_energies[i] < energy_limit:
                    chakra_energies[i] = min(energy_limit, chakra_energies[i] + 0.03) 
                else:
                    pass 

        elif pose_landmarks and not gaze_distracted:
            # [FIX] Just Sitting (Body Detected) -> Rise to 30%
            target_sitting = 0.30
            for i in range(len(chakra_energies)):
                if chakra_energies[i] < target_sitting:
                    chakra_energies[i] += 0.01 
                else:
                    pass

        else:
            # No Body Detected -> Decay
            pass 
            
        # [FIX] GLOBAL RAPID DECAY ENFORCER
        # "energy tezi se girni chahiye"
        # If current energy > limit, drop FAST (-0.05)
        for i in range(len(chakra_energies)):
            if chakra_energies[i] > energy_limit:
                chakra_energies[i] = max(energy_limit, chakra_energies[i] - 0.05) # Rapid Drop
            elif chakra_energies[i] < 0:
                chakra_energies[i] = 0.0 

        # Draw Sidebar
        draw_mudra_sidebar(frame, detected_mudra_name)

        # Time Delta Calculation
        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        
        # Golden Aura Logic
        is_yoga_active = (posture_score > 0.1) or gyan_active
        # Removed conflicting assignment: yoga_mode_active = is_yoga_active 

        # [NEW] XP & Leveling System (20 Levels)
        xp_gain = 0.0
        warning_msg = ""
        
        if posture_score > 0.4:
            xp_gain += 1.0 # Base XP for Posture
            if detected_mudra_name:
                xp_gain += 1.0 # Bonus for Mudra
            if is_eyes_closed:
                xp_gain += 2.0 # Bonus for Eyes Closed
        
        # [NEW] Level Gating Logic
        # Levels 4-10: Mudra Mandatory
        if current_level >= 3 and current_level < 10:
             if not detected_mudra_name:
                 xp_gain = 0.0
                 warning_msg = "MUDRA REQUIRED TO PROGRESS!"
        
        # Levels 11-20: Eyes Closed Mandatory
        if current_level >= 10:
             if not is_eyes_closed:
                 xp_gain = 0.0
                 warning_msg = "CLOSE EYES TO PROGRESS!"

        # Apply XP
        if current_level < MAX_LEVEL:
            total_xp += xp_gain
            
            # Check Level Up
            # Level = (Total XP / XP_PER_LEVEL) + 1
            calculated_level = int(total_xp / XP_PER_LEVEL) + 1
            calculated_level = min(MAX_LEVEL, calculated_level)
            
            if calculated_level > current_level:
                current_level = calculated_level
                speak_threaded(f"Level {current_level} Reached!")
                
        # Draw Level Progress UI (Top Center) - PREMIUM VERSION
        # Bar Dimensions - Adjusted to fit between Left Panel and Right Sidebar
        bar_w = 220 # [FIX] Reduced width (was 280)
        bar_h = 15  # [FIX] Reduced height (was 30) for slimmer look
        
        # [FIX] Right-Aligned Position (Close to Arrow/Sidebar)
        # Right Sidebar starts at w-280. We place bar just left of it.
        # bar_x = (Sidebar Start) - (Bar Width) - (Padding)
        bar_x = (w - 280) - bar_w - 20
        bar_center_x = bar_x + bar_w // 2
        
        # [FIX] Moved up slightly per user request
        bar_y = 100 # Was 120
        
        # Draw Level Text with GLOW Effect
        level_text = f"LEVEL {current_level}"
        
        # Dynamic Centering for Text
        (lw, lh), _ = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 6)
        text_x = bar_center_x - lw // 2
        
        # Black shadow for depth
        cv2.putText(frame, level_text, (text_x, bar_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        # Gold glow
        cv2.putText(frame, level_text, (text_x, bar_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 4)
        # White core
        cv2.putText(frame, level_text, (text_x, bar_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Draw Warning Message if Gated
        if warning_msg:
             # [FIX] "Fade in Motion" & "Very Bright"
             # 1. Calculate Pulse (Alpha between 0.6 and 1.0) - [FIX] Increased min alpha
             pulse = (math.sin(anim_time * 8) + 1) / 2 
             alpha = 0.6 + 0.4 * pulse
             
             # 2. Create Overlay for Transparency
             overlay_warn = frame.copy()
             
             # Calculate text size
             (tw, th), _ = cv2.getTextSize(warning_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2) # [FIX] Font 0.55
             # Center relative to the BAR
             tx = bar_center_x - tw // 2
             ty = bar_y + 35 # Adjusted for slightly larger font
             
             # Background Box (Black with Red Border)
             pad = 6 # [FIX] Slightly more padding
             cv2.rectangle(overlay_warn, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad), (0, 0, 0), -1)
             cv2.rectangle(overlay_warn, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad), (0, 0, 255), 1)
             
             # Text (High Visibility: Triple Stroke)
             # 1. Black Outer Shadow (Contrast)
             cv2.putText(overlay_warn, warning_msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 6)
             # 2. White Outline (Brightness)
             cv2.putText(overlay_warn, warning_msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3) 
             # 3. Red Core (Color)
             cv2.putText(overlay_warn, warning_msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
             
             # 3. Apply Fade Animation
             cv2.addWeighted(overlay_warn, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw Progress Bar with PREMIUM styling
        # Outer glow
        for i in range(5):
            alpha = 0.15 - (i * 0.03)
            offset = 5 - i
            overlay_glow = frame.copy()
            cv2.rectangle(overlay_glow, (bar_x - offset, bar_y - offset), (bar_x + bar_w + offset, bar_y + bar_h + offset), (0, 255, 255), -1)
            cv2.addWeighted(overlay_glow, alpha, frame, 1 - alpha, 0, frame)
        
        # Dark background with border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (20, 20, 25), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 255, 255), 2)
        
        # Calculate Progress %
        # XP for current level start = (current_level - 1) * XP_PER_LEVEL
        # XP for next level = current_level * XP_PER_LEVEL
        xp_start = (current_level - 1) * XP_PER_LEVEL
        xp_next = current_level * XP_PER_LEVEL
        
        if current_level < MAX_LEVEL:
            progress = (total_xp - xp_start) / (xp_next - xp_start)
            progress = max(0.0, min(1.0, progress))
            fill_w = int(bar_w * progress)
            
            # Draw gradient fill (darker to lighter cyan)
            for i in range(fill_w):
                ratio = i / max(1, fill_w)
                # Gradient from dark cyan to bright cyan
                color_val = int(150 + (105 * ratio))
                cv2.line(frame, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_h), (0, color_val, 255), 1)
            
            # Inner glow on progress
            overlay_prog = frame.copy()
            cv2.rectangle(overlay_prog, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (100, 255, 255), -1)
            cv2.addWeighted(overlay_prog, 0.3, frame, 0.7, 0, frame)
        else:
            # Max Level - Full Gold Bar with premium glow
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 215, 255), -1)
            (mw, mh), _ = cv2.getTextSize("MAX LEVEL", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, "MAX LEVEL", (bar_center_x - mw // 2, bar_y + bar_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

        # Keep Old Visual Rewards (Neck Medals) based on Milestones
        # Bronze: Lvl 1-5, Silver: Lvl 6-10, Gold: Lvl 11-15, Trophy: Lvl 16-20
        visual_tier = 0
        if current_level >= 1: visual_tier = 1
        if current_level >= 6: visual_tier = 2
        if current_level >= 11: visual_tier = 3
        if current_level >= 16: visual_tier = 4

        # Draw Visual Rewards (Medals on Neck)
        if pose_landmarks:
            # Calculate Neck Position (Midpoint of Shoulders 11 & 12)
            left_shoulder = pose_landmarks.landmark[11]
            right_shoulder = pose_landmarks.landmark[12]
            neck_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
            neck_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
            
            # Draw Medals based on Visual Tier
            if visual_tier >= 1:
                # Bronze Medal
                cv2.circle(frame, (neck_x, neck_y + 40), 15, (50, 100, 150), -1) # Bronze Color
                cv2.circle(frame, (neck_x, neck_y + 40), 15, (255, 255, 255), 1)
            if visual_tier >= 2:
                # Silver Medal (Slightly lower/overlapping)
                cv2.circle(frame, (neck_x + 10, neck_y + 45), 15, (192, 192, 192), -1) # Silver
                cv2.circle(frame, (neck_x + 10, neck_y + 45), 15, (255, 255, 255), 1)
            if visual_tier >= 3:
                # Gold Medal (Center, on top)
                cv2.circle(frame, (neck_x, neck_y + 50), 18, (0, 215, 255), -1) # Gold
                cv2.circle(frame, (neck_x, neck_y + 50), 18, (255, 255, 255), 2)


        # Draw Progression Titles (Top Center - Under Bar)
        if visual_tier == 2:
             # Tier 2: Level 6-10
             title = "YOG GURU"
             (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 5)
             tx = bar_center_x - tw // 2
             cv2.putText(frame, title, (tx, bar_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
             cv2.putText(frame, title, (tx, bar_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        elif visual_tier == 3:
             # Tier 3: Level 11-15
             title = "TRUE YOGI"
             (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 5)
             tx = bar_center_x - tw // 2
             cv2.putText(frame, title, (tx, bar_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
             cv2.putText(frame, title, (tx, bar_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        elif visual_tier == 4:
             # Tier 4: Level 16-20
             title = "MASTER YOGI"
             (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 5)
             tx = bar_center_x - tw // 2
             cv2.putText(frame, title, (tx, bar_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
             cv2.putText(frame, title, (tx, bar_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
             # Trophy Icon 🏆
             tx, ty = bar_center_x, bar_y + 110
             # Cup
             cv2.ellipse(frame, (tx, ty), (30, 30), 0, 0, 180, (0, 215, 255), -1)
             cv2.line(frame, (tx, ty+30), (tx, ty+50), (0, 215, 255), 4) # Stem
             cv2.rectangle(frame, (tx-20, ty+50), (tx+20, ty+60), (0, 215, 255), -1) # Base

        # Aura Color Update based on Yoga State
        # Golden if: Yoga Mode (Manual) OR Yoga Active (Auto) OR Eyes Closed OR Meditation
        aura_color = (255, 0, 0) # Blue (BGR)
        if yoga_mode_active or is_yoga_active or is_eyes_closed or alignment_mode:
            aura_color = (0, 215, 255) # Gold
        
        # Dynamic Speed - SLOWER & CALMER
        speed_multiplier = 1.0
        if is_yoga_active:
            speed_multiplier = 2.0 # Reduced from 4.0
            
        if alignment_mode:
             speed_multiplier = 3.0 # Reduced from 12.0 for smoother feel 
            
        anim_time += dt * speed_multiplier

        # Draw Meditation Info Panel
        # [FIX] Restored and Relocated beside BPM/Oxygen
        avg_energy = sum(chakra_energies) / len(chakra_energies)
        draw_meditation_info_panel(frame, meditation_tracker.in_dhyana, is_eyes_closed, avg_energy)

        # Draw Sidebar

        # Draw Universe Background - REMOVED per user request
        # draw_universe(frame, anim_time)

        # Draw Background Aura
        # center_x is already defined in main loop
        center_y_aura = int(h * 0.55)
        aura_radius = int(min(w, h) * 0.4)
        
        if is_yoga_active and not alignment_mode:
             # Override if yoga is active (keep Gold)
             aura_color = (0, 215, 255) 
        elif not alignment_mode:
            # Dynamic Color based on Energy
            if avg_energy <= 0.05:
                aura_color = (255, 0, 0) # [FIX] Blue (was White)
            elif avg_energy < 0.40:
                aura_color = (255, 0, 0) # Blue
            else:
                aura_color = (0, 215, 255) # Gold (High Energy)

        overlay_bg = frame.copy()
        cv2.circle(overlay_bg, (center_x, center_y_aura), aura_radius, aura_color, -1)
        cv2.addWeighted(overlay_bg, 0.15, frame, 0.85, 0, frame)
        
        if is_yoga_active and not alignment_mode:
             head_y = int(nose.y * h) if 'nose' in locals() else center_y_aura - 100
             head_x = int(nose.x * w) if 'nose' in locals() else center_x
             head_y = int(nose.y * h) if 'nose' in locals() else center_y_aura - 100
             head_x = int(nose.x * w) if 'nose' in locals() else center_x
             # [FIX] Replaced Astral Projection with Divine OM Effect
             draw_om_effect(frame, avg_energy)
             
        if gyan_active and not alignment_mode:
            # draw_gyan_sparkles(frame, center_x, center_y_aura, aura_radius) # REMOVED per user request
            pass
        
        # Breath factor
        breath_factor = breathing.get_breath_factor()
        
        # Draw Chakras
        if not yoga_mode_active:
            draw_chakras(frame, center_x, top_y, bottom_y, detected_mudra, chakra_energies, aura_color, breath_factor, anim_time)
        else:
            # Yoga Mode UI
            # Draw Chakras (so they are visible during Awakening/Meditation)
            draw_chakras(frame, center_x, top_y, bottom_y,
                        active_chakra_idx if active_chakra_idx is not None else -1,
                        chakra_energies, aura_color, breath_factor, anim_time)
            
        # Draw Chakra Meter (ALWAYS VISIBLE)
        draw_chakra_meter(frame, chakra_energies)

        # [MOVED] Meditation Tracker update moved to top of loop
        pass

        if yoga_mode_active:
            # Yoga Mode UI - Narrower & Deeper
            # Align with Meditation Info Panel (x = w - 520)
            stats_x = w - 520
            # [FIX] Start slightly higher to allow for more depth downwards
            stats_y = h - 280 
            
            draw_text_with_bg(frame, "YOGA MODE: ACTIVE", stats_x, stats_y, font_scale=0.7, color=(0, 255, 0))
            
            # Vertical Stack for "Depth"
            # 1. Breath
            draw_text_with_bg(frame, f"Breath: {breathing.breath_phase:.2f}", stats_x, stats_y + 35, font_scale=0.5, color=(200, 255, 200))
            
            # 2. Posture
            draw_text_with_bg(frame, f"Posture: {posture_label}", stats_x, stats_y + 65, font_scale=0.5, color=(200, 255, 200))
            
            # 3. Stage
            stage_color = (255, 200, 100)
            if "Dhyana" in med_stage: stage_color = (100, 255, 255)
            if "Samadhi" in med_stage: stage_color = (255, 100, 255)
            draw_text_with_bg(frame, f"Stage: {med_stage}", stats_x, stats_y + 95, font_scale=0.5, color=stage_color)
            
            # 4. Gaze
            draw_text_with_bg(frame, f"Gaze: {gaze_label}", stats_x, stats_y + 125, font_scale=0.5, color=stage_color)
            
            # 5. Concentration
            draw_text_with_bg(frame, f"Concentration: {med_level:.1f}%", stats_x, stats_y + 155, font_scale=0.5, color=(255, 255, 255))
            
            # If in Dhyana/Samadhi, boost all chakras significantly
            if "Dhyana" in med_stage or "Samadhi" in med_stage:
                 for i in range(len(chakra_energies)):
                    # Faster increase for visibility ("Jump high")
                    chakra_energies[i] = min(1.0, chakra_energies[i] + 0.15)

        # Draw Smart Tracking
        draw_smart_tracking(frame, hand_res, face_res, yoga_mode=yoga_mode_active)

        # [REMOVED] Controls Overlay - User requested removal


        # [NEW] Kumbhaka (Breath Retention) Update & Draw
        # 1. Detect Nose Touch (Nadi Shodhana / Pranayama Gesture)
        is_touching_nose = False
        if face_res.multi_face_landmarks and hand_res.multi_hand_landmarks:
            try:
                # Get Nose Tip (Index 1 or 4)
                face_lm = face_res.multi_face_landmarks[0]
                nose_tip = face_lm.landmark[4] # Tip of nose
                nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)
                
                # Check all hands
                for hand_lm in hand_res.multi_hand_landmarks:
                    # Check Index Tip (8) and Thumb Tip (4)
                    for tip_id in [4, 8]:
                        tip = hand_lm.landmark[tip_id]
                        tx, ty = int(tip.x * w), int(tip.y * h)
                        
                        # Distance check (Euclidean)
                        dist = ((nx - tx)**2 + (ny - ty)**2)**0.5
                        if dist < 40: # Threshold in pixels
                            is_touching_nose = True
                            # Optional: Draw a visual indicator
                            cv2.circle(frame, (nx, ny), 10, (0, 255, 0), -1)
                            break
                    if is_touching_nose: break
            except:
                pass

        # Update with current HR and Nose Touch status
        prana_val, is_holding_breath = kumbhaka_tracker.update(hr_monitor.heart_rate, is_touching_nose)
        draw_kumbhaka_bar(frame, prana_val, is_holding_breath)

        if not is_touching_nose:
             # [REMOVED] Moved to draw_status_panel
             pass

        # [MOVED] Calculate Avg Energy early for use in effects
        avg_energy = sum(chakra_energies) / len(chakra_energies)

        # [NEW] Elemental Mastery Effects
        # Pass hand landmarks list and face landmarks (first face)
        face_lm_single = face_res.multi_face_landmarks[0] if face_res.multi_face_landmarks else None
        elemental_effects.update_and_draw(frame, detected_mudra_name, hand_res.multi_hand_landmarks, face_lm_single)

        # [NEW] Divine OM Effect
        # Only if Energy > 90%
        if avg_energy > 0.9:
             draw_om_effect(frame, avg_energy)

        # Namaste Detection for Screenshot (Replaces Mode Toggle)
        # [FIX] Robust Detection with Grace Period & Visual Feedback
        is_namaste = detect_namaste(hand_res)
        
        if is_namaste:
            namaste_grace_frames = 15 # Reset grace period (approx 0.5s at 30fps)
            if namaste_hold_start == 0:
                namaste_hold_start = time.time()
            
            # Calculate Progress
            hold_duration = time.time() - namaste_hold_start
            req_duration = 1.0 # 1 second hold
            progress = min(1.0, hold_duration / req_duration)
            
            # Draw Progress Bar
            if not namaste_triggered:
                # Draw "Hold for Screenshot" text
                cv2.putText(frame, "HOLD FOR SCREENSHOT...", (center_x - 180, center_y_aura + 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw Progress Bar
                bar_w = 200
                bar_h = 10
                bar_x = center_x - bar_w // 2
                bar_y = center_y_aura + 170
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 255, 0), -1)

            if hold_duration > req_duration:
                if not namaste_triggered:
                    # [FIX] Take Screenshot
                    avg_energy = sum(chakra_energies) / len(chakra_energies)
                    filename = generate_aura_photo(frame, aura_color, hr_monitor.heart_rate, avg_energy)
                    
                    namaste_triggered = True
                    print(f"[INFO] 🙏 Namaste Screenshot Captured! Saved to: {filename}")
                    
                    # Visual Flash Effect
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), -1)
                    cv2.addWeighted(flash, 0.5, frame, 0.5, 0, frame)
                    cv2.putText(frame, "NAMASTE - SCREENSHOT SAVED!", (center_x - 250, center_y_aura), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.imshow("AI ChakraFlow — Full Experience", frame)
                    cv2.waitKey(300) # Brief flash
        else:
            # Grace Period Logic
            if namaste_grace_frames > 0:
                namaste_grace_frames -= 1
                # Keep holding start time valid, just waiting
            else:
                namaste_hold_start = 0
                namaste_triggered = False

        # [NEW] Gesture Screenshot (Peace Sign)
        is_peace = False
        if hand_res.multi_hand_landmarks:
            for hand_lm in hand_res.multi_hand_landmarks:
                if detect_peace(hand_lm):
                    is_peace = True
                    break
        
        if is_peace:
            if screenshot_timer == 0:
                screenshot_timer = time.time()
            
            hold_duration = time.time() - screenshot_timer
            
            if hold_duration > 1.0 and screenshot_countdown_start == 0:
                # Start Countdown
                screenshot_countdown_start = time.time()
                
        else:
            if screenshot_countdown_start == 0:
                screenshot_timer = 0
                
        # Handle Countdown & Capture
        if screenshot_countdown_start > 0:
            elapsed = time.time() - screenshot_countdown_start
            remaining = 3.0 - elapsed
            
            if remaining > 0:
                # Draw Countdown
                count_text = str(int(remaining) + 1)
                cv2.putText(frame, count_text, (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
                cv2.putText(frame, "Say Cheese!", (w//2 - 100, h//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                # Capture!
                avg_energy = sum(chakra_energies) / len(chakra_energies)
                generate_aura_photo(frame, aura_color, hr_monitor.heart_rate, avg_energy)
                
                # Reset
                screenshot_countdown_start = 0
                screenshot_timer = 0
                cv2.putText(frame, "Captured!", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.waitKey(500) # Brief pause to show "Captured!"

        # COACH TEXT & STATUS PANEL
        # Refactored to use draw_status_panel for non-overlapping text in the center
        
        status = {}
        
        # 1. Title: Detected Mudra or Default
        if detected_mudra_name:
            status['title'] = f"Detected: {detected_mudra_name}"
        else:
            status['title'] = "AI ChakraFlow"
            
        # 2. Subtitle: Contextual Info
        if gyan_active and not alignment_mode:
            status['subtitle'] = "Crown Chakra Awakening"
        elif alignment_mode:
            status['subtitle'] = "Alignment Mode Active"
        elif yoga_mode_active:
            status['subtitle'] = "Yoga Mode Active"
        else:
            status['subtitle'] = "Meditation & Mudra Engine"
            
        # 3. Hint: Coach Message or Instructions
        idle_time = time.time() - last_activation_time
        # [FIX] Removed "Touch nose" and "Press q" from here (Moved to Sidebar)
        
        if idle_time > 10 and not alignment_mode and not gyan_active:
            status['hint'] = "Hint: Try Gyan Mudra for Crown"
        else:
            # Use Smart Coach Message if available
            coach_msg = generate_smart_coach_message(chakra_energies, mood_label, alignment_mode, gyan_active)
            # Truncate if too long for the circle
            if len(coach_msg) > 40: coach_msg = coach_msg[:37] + "..."
            status['hint'] = coach_msg
            
        # 4. Extra: Quit instruction - REMOVED
        # status['extra'] = "Press 'q' to quit"
        
        # Draw the Status Panel in the Center Circle
        # center_x and center_y_aura are defined earlier
        draw_status_panel(frame, center_x, center_y_aura, aura_radius, status)


        elapsed_min = (time.time() - session_start) / 60.0
        top_text = f"AI ChakraFlow  |  Session: {elapsed_min:.1f} min  |  Mood: {mood_label}  |  Posture: {posture_label}"
        
        # [FIX] Premium Top Bar (Smaller, Centered, Gold Accent)
        # Calculate size for centering
        (tw, th), _ = cv2.getTextSize(top_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx = w // 2 - tw // 2
        ty = 15 # [FIX] Moved to very top (was 30) to avoid overlap
        
        # Background with Border
        overlay = frame.copy()
        pad = 8 # Slightly reduced padding
        cv2.rectangle(overlay, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad), (10, 10, 10), -1) 
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Gold Border
        cv2.rectangle(frame, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad), (0, 215, 255), 1)
        
        # Text (Gold)
        cv2.putText(frame, top_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)

        # [NEW] Indian Flag (Top Left Corner - No Overlap)
        # Chakra Meter moved down to y=130 to accommodate this
        draw_indian_flag(frame, 20, 15, 40, anim_time)

        # [NEW] Check Hover for Speaking Graphs
        check_hover_and_speak(w, h)

        cv2.imshow("AI ChakraFlow — Full Experience", frame)

        # Voice trigger disabled to avoid lag; set ENABLE_VOICE=True to re-enable.
        if False:
            if time.time() - last_voice_check > 4:
                last_voice_check = time.time()
                try:
                    with mic as source:
                        r.adjust_for_ambient_noise(source, duration=0.2)
                        audio = r.listen(source, timeout=0.6, phrase_time_limit=1.2)
                    try:
                        cmd = r.recognize_google(audio, language="hi-IN").lower()
                        if "hari om" in cmd or ("hari" in cmd and "om" in cmd):
                            duration_min_now = (time.time() - session_start) / 60.0
                            show_chakra_bar_graph(chakra_energies)
                            summary_img = create_summary_image(chakra_energies, duration_min_now,
                                                               total_gyan_count, alignment_count)
                            cv2.imshow("Summary Image", summary_img)
                            speak_summary(chakra_energies, total_gyan_count,
                                          alignment_count, duration_min_now)
                    except Exception:
                        pass
                except Exception:
                    pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # [FIX] Removed Screenshot on Quit
            break
        elif key == ord('r'):
            print("[INFO] Manual Reconnect Requested...")
            hr_monitor.connect()
        elif key == ord('s'):
            # [NEW] Manual Screenshot with 'S' key
            avg_energy = sum(chakra_energies) / len(chakra_energies)
            filename = generate_aura_photo(frame, aura_color, hr_monitor.heart_rate, avg_energy)
            print(f"[INFO] 📸 Manual Screenshot Captured! Saved to: {filename}")
            # Visual Flash Effect
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), -1)
            cv2.addWeighted(flash, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, "SCREENSHOT SAVED!", (center_x - 180, center_y_aura), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("AI ChakraFlow — Full Experience", frame)
            cv2.waitKey(300)  # Brief flash

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

    show_final_report(session_start, chakra_energies, total_gyan_count, alignment_count)
    summary = analytics.summary()
    print("\n--- Analytics ---")
    print("Mudra counts:", summary["mudras"])
    print("Avg posture score:", f"{summary['avg_posture']:.2f}")
    print("Posture alerts (score<0.5):", summary["posture_alerts"])
    print("Time per chakra (s):", [round(t, 1) for t in summary["chakra_time"]])
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
