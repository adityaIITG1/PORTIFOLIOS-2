#!/usr/bin/env python
"""Diagnostic script to test which module is hanging"""

print("[START] Beginning initialization test...")

print("[TEST 1] Importing basic modules...")
import cv2
import mediapipe as mp
import numpy as np
import time
import math
print("[OK] Basic modules imported")

print("[TEST 2] Importing pygame...")
import pygame
print("[OK] Pygame imported")

print("[TEST 3] Initializing pygame.mixer...")
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print("[OK] Pygame mixer initialized")
except Exception as e:
    print(f"[ERROR] Pygame mixer failed: {e}")

print("[TEST 4] Importing pyttsx3...")
import pyttsx3
print("[OK] pyttsx3 imported")

print("[TEST 5] Initializing TTS engine...")
try:
    tts = pyttsx3.init()
    print("[OK] TTS initialized")
    
    print("[TEST 6] Setting TTS properties...")
    voices = tts.getProperty('voices')
    print(f"[OK] Found {len(voices)} voices")
    tts.setProperty('rate', 150)
    tts.setProperty('volume', 1.0)
    print("[OK] TTS configured")
except Exception as e:
    print(f"[ERROR] TTS failed: {e}")

print("[COMPLETE] All tests passed!")
