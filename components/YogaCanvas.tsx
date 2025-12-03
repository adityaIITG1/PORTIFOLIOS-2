"use client";

import { useEffect, useRef, useState } from "react";
import { useVisionModels } from "@/hooks/useVisionModels";
import { useVoiceAssistant } from "@/hooks/useVoiceAssistant";
import { useArduino } from "@/hooks/useArduino";
import {
    classifyGesture,
    detectNamaste,
} from "@/utils/gesture-recognition";
import { analyzeFace } from "@/utils/face-logic";
import {
    drawUniverse,
    drawChakras,
    drawSmartTracking,
} from "@/utils/drawing";
import { generateSmartCoachMessage } from "@/utils/smart-coach";
import html2canvas from "html2canvas";

import TopBar from "./TopBar";
import RightSidebar from "./RightSidebar";
import LeftSidebar from "./LeftSidebar";
import BottomOverlay from "./BottomOverlay";
import BioAnalyticsPanel from "./BioAnalyticsPanel";

export default function YogaCanvas() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const universeRef = useRef<HTMLCanvasElement>(null);
    const { handLandmarker, faceLandmarker, isLoading, error: aiError } = useVisionModels();
    const { isListening, isSpeaking, toggleListening } = useVoiceAssistant();
    const { arduinoData, connectArduino, arduinoError } = useArduino();

    // Ref to track isSpeaking without triggering re-renders in the animation loop
    const isSpeakingRef = useRef(isSpeaking);
    useEffect(() => {
        isSpeakingRef.current = isSpeaking;
    }, [isSpeaking]);

    const [gesture, setGesture] = useState<string | null>(null);
    const [feedback, setFeedback] = useState("Tip: Focus on breath. Root is strong...");
    const [logs, setLogs] = useState<string[]>([]);
    const [isPlaying, setIsPlaying] = useState(false);

    // Animation state
    const requestRef = useRef<number>(0);
    const startTimeRef = useRef<number>(0);

    // Refs for Animation Loop State
    const energiesRef = useRef<number[]>([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    const activeIndexRef = useRef(0);
    const gestureRef = useRef<string | null>(null);
    const auraIntensityRef = useRef(0.0);
    const eyesClosedTimeRef = useRef(0);
    const eyesOpenTimeRef = useRef(0);
    const isMeditationRef = useRef(false);

    // XP & Level State
    const xpRef = useRef(0.0);
    const levelRef = useRef(3);
    const warningMsgRef = useRef<string | null>(null);
    const pranaRef = useRef(0); // Kumbhaka
    const namasteHoldTimeRef = useRef(0); // Screenshot Trigger
    const lastScreenshotTimeRef = useRef(0);

    // Third Eye & Particles
    const thirdEyeRef = useRef({ dwellTime: 0, target: null as string | null });
    const particlesRef = useRef<Array<{ x: number, y: number, vx: number, vy: number, life: number, type: string, color: string }>>([]);

    // UI State for Level & Stats
    const [xp, setXp] = useState(0);
    const [level, setLevel] = useState(3);
    const [levelProgress, setLevelProgress] = useState(0);
    const [warningMsg, setWarningMsg] = useState<string | null>(null);
    const [mood, setMood] = useState("Relaxed");
    const [posture, setPosture] = useState("Good");
    const [alignmentMode, setAlignmentMode] = useState("Standard");

    // Debounce Refs
    const pendingGestureRef = useRef<string | null>(null);
    const pendingGestureStartTimeRef = useRef(0);
    const lastSpeechTimeRef = useRef(0);
    const lastSpeechTextRef = useRef("");

    // React State for UI updates (Sidebar)
    const [uiEnergies, setUiEnergies] = useState<number[]>([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    const [sessionTime, setSessionTime] = useState("0.0 min");
    const [screenshotFlash, setScreenshotFlash] = useState(false);

    const addLog = (msg: string) => setLogs(prev => [...prev.slice(-4), msg]);

    const audioRef = useRef<HTMLAudioElement | null>(null);

    const speak = (text: string) => {
        // Prevent overlapping with Intro Speech
        if (isSpeakingRef.current) return;

        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            const voices = window.speechSynthesis.getVoices();
            // Try to find an Indian female voice
            const voice = voices.find(v =>
                (v.name.includes("India") || v.name.includes("Hindi") || v.name.includes("Heera")) &&
                v.name.includes("Female")
            ) || voices.find(v => v.name.includes("Google हिन्दी")) || voices.find(v => v.name.includes("Female"));

            if (voice) utterance.voice = voice;
            utterance.rate = 1.1; // Energetic (slightly faster)
            utterance.pitch = 1.2; // Sweet (higher pitch)
            window.speechSynthesis.speak(utterance);
        }
    };

    const hasLoadedRef = useRef(false);
    const hasErrorRef = useRef(false);

    useEffect(() => {
        if (isLoading && !hasLoadedRef.current) {
            // addLog("AI: Loading models...");
        }
        if (handLandmarker && faceLandmarker && !hasLoadedRef.current) {
            setTimeout(() => addLog("AI: Models loaded successfully"), 0);
            hasLoadedRef.current = true;
        }
        if (aiError && !hasErrorRef.current) {
            setTimeout(() => addLog(`AI Error: ${aiError}`), 0);
            hasErrorRef.current = true;
        }
        if (arduinoError) {
            setTimeout(() => addLog(`Sensor Error: ${arduinoError}`), 0);
        }
    }, [isLoading, handLandmarker, faceLandmarker, aiError, arduinoError]);

    useEffect(() => {
        audioRef.current = new Audio("/adiyogi.mp3");
        audioRef.current.loop = true;
        audioRef.current.volume = 0.6;
        audioRef.current.preload = "auto";

        const playAudio = async () => {
            try {
                await audioRef.current?.play();
                setIsPlaying(true);
                addLog("Audio: Auto-playing");
            } catch (e) {
                console.warn("Autoplay blocked", e);
                addLog("Audio: Autoplay blocked. Click Start.");
                setIsPlaying(false);
            }
        };
        playAudio();

        const startCamera = async () => {
            if (videoRef.current) {
                addLog("Camera: Requesting access...");
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            width: 1280,
                            height: 720,
                        },
                    });
                    addLog("Camera: Access granted");
                    videoRef.current.srcObject = stream;
                    videoRef.current.onloadeddata = () => addLog("Camera: Data loaded");
                    videoRef.current.play();
                    addLog("Camera: Playing stream");
                } catch (err: unknown) {
                    console.error("Error accessing webcam:", err);
                    const errorMessage = err instanceof Error ? err.message : String(err);
                    addLog(`Camera Error: ${errorMessage}`);
                    setFeedback("Camera access denied.");
                }
            }
        };

        startCamera();

        return () => {
            audioRef.current?.pause();
            audioRef.current = null;
        };
    }, []);

    const toggleAudio = () => {
        if (!audioRef.current) return;
        if (isPlaying) {
            audioRef.current.pause();
            setIsPlaying(false);
            addLog("Audio: Paused");
        } else {
            audioRef.current.play().then(() => {
                setIsPlaying(true);
                addLog("Audio: Playing");
            }).catch(e => addLog(`Audio Error: ${(e as Error).message}`));
        }
    };

    const takeScreenshot = async () => {
        const now = Date.now();
        if (now - lastScreenshotTimeRef.current < 2000) return; // Debounce 2s
        lastScreenshotTimeRef.current = now;

        // Visual Flash
        setScreenshotFlash(true);
        setTimeout(() => setScreenshotFlash(false), 300);

        speak("Screenshot captured.");
        addLog("📸 Screenshot Saved!");

        if (document.body) {
            try {
                const canvas = await html2canvas(document.body, {
                    useCORS: true,
                    ignoreElements: (element) => element.tagName === 'VIDEO', // Avoid video taint issues if any
                    backgroundColor: '#000000',
                });

                // Manually draw the video frame onto the canvas since html2canvas might miss it
                if (videoRef.current && canvasRef.current) {
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                        // Draw video frame (mirrored)
                        ctx.save();
                        ctx.scale(-1, 1);
                        ctx.translate(-canvas.width, 0);
                        // We need to draw the video at the bottom layer, but html2canvas already drew the UI.
                        // Ideally we'd draw video first, but html2canvas captures the DOM.
                        // A simple workaround for this "parity" task: Just capture the UI or accept the video might be black if tainted.
                        // Better approach: Draw the current video frame from our canvasRef (which has the video drawn on it!)
                        // Our canvasRef has the video drawn on it in the animate loop!
                        // "ctx.drawImage(video, 0, 0, width, height);"
                        // So canvasRef actually HAS the video frame!
                        // So we just need to make sure html2canvas captures canvasRef correctly.
                    }
                }

                const link = document.createElement('a');
                const timestamp = new Date().toISOString().replace(/[-:.]/g, "").slice(0, 15);
                link.download = `yoga_session_${timestamp}.png`;
                link.href = canvas.toDataURL("image/png");
                link.click();
            } catch (err) {
                console.error("Screenshot failed:", err);
                addLog("Screenshot Failed");
            }
        }
    };

    // 'S' Key Listener
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key.toLowerCase() === 's') {
                takeScreenshot();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    useEffect(() => {
        if (
            !handLandmarker ||
            !faceLandmarker ||
            !videoRef.current ||
            !canvasRef.current
        ) return;

        startTimeRef.current = Date.now();

        // Load Mudra Images
        const brainImg = new Image();
        brainImg.src = "/brain_glow.png";
        const sunImg = new Image();
        sunImg.src = "/sun_glow.png";
        const pranaImg = new Image();
        pranaImg.src = "/prana_glow.png";
        const apanaImg = new Image();
        apanaImg.src = "/apana_glow.png";
        const varunImg = new Image();
        varunImg.src = "/varun_glow.png";

        const animate = () => {
            if (
                !canvasRef.current ||
                !videoRef.current ||
                videoRef.current.readyState < 2
            ) {
                requestRef.current = requestAnimationFrame(animate);
                return;
            }

            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d");
            const video = videoRef.current;

            if (!ctx) return;

            if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }

            const width = canvas.width;
            const height = canvas.height;
            const t = (Date.now() - startTimeRef.current) / 1000;

            // 1. Draw Video
            ctx.save();
            ctx.scale(-1, 1);
            ctx.translate(-width, 0);
            ctx.drawImage(video, 0, 0, width, height);
            ctx.restore();

            // 2. AI Detection
            const now = Date.now();
            const handResults = handLandmarker.detectForVideo(video, now);
            const faceResults = faceLandmarker.detectForVideo(video, now);

            let currentGesture = null;
            let isEyesClosed = false;

            // Hand Logic
            if (handResults.landmarks) {
                for (const landmarks of handResults.landmarks) {
                    drawSmartTracking(ctx, landmarks, width, height);

                    const g = classifyGesture(landmarks);
                    if (g) {
                        currentGesture = g;

                        // [NEW] Draw Mudra Visuals (Brain/Sun/Prana/Apana/Varun)
                        // Calculate Palm Center (Approx between Wrist 0 and Middle MCP 9)
                        const wrist = landmarks[0];
                        const middleMcp = landmarks[9];

                        const cx = (wrist.x + middleMcp.x) / 2 * width;
                        const cy = (wrist.y + middleMcp.y) / 2 * height;
                        const size = 100; // Size of the icon

                        ctx.save();
                        ctx.globalAlpha = 0.9;

                        if (g === "Gyan Mudra" && brainImg.complete) {
                            ctx.drawImage(brainImg, cx - size / 2, cy - size / 2, size, size);
                            ctx.shadowColor = "cyan";
                            ctx.shadowBlur = 20;
                        } else if (g === "Surya Mudra" && sunImg.complete) {
                            ctx.drawImage(sunImg, cx - size / 2, cy - size / 2, size, size);
                            ctx.shadowColor = "orange";
                            ctx.shadowBlur = 20;
                        } else if (g === "Prana Mudra" && pranaImg.complete) {
                            ctx.drawImage(pranaImg, cx - size / 2, cy - size / 2, size, size);
                            ctx.shadowColor = "#00ff00"; // Green
                            ctx.shadowBlur = 20;

                            // Spawn Nature Particles
                            if (Math.random() < 0.2) {
                                particlesRef.current.push({
                                    x: cx, y: cy,
                                    vx: (Math.random() - 0.5) * 2,
                                    vy: -1 - Math.random(), // Float up
                                    life: 1.0,
                                    type: "nature",
                                    color: "rgba(50, 205, 50, 0.8)"
                                });
                            }
                        } else if (g === "Apana Mudra" && apanaImg.complete) {
                            ctx.drawImage(apanaImg, cx - size / 2, cy - size / 2, size, size);
                            ctx.shadowColor = "#8b0000"; // Dark Red
                            ctx.shadowBlur = 20;
                        } else if (g === "Varun Mudra" && varunImg.complete) {
                            ctx.drawImage(varunImg, cx - size / 2, cy - size / 2, size, size);
                            ctx.shadowColor = "#00bfff"; // Deep Sky Blue
                            ctx.shadowBlur = 20;

                            // Spawn Water Particles
                            if (Math.random() < 0.3) {
                                particlesRef.current.push({
                                    x: cx, y: cy,
                                    vx: (Math.random() - 0.5) * 2,
                                    vy: 2 + Math.random() * 2, // Fall down
                                    life: 1.0,
                                    type: "water",
                                    color: "rgba(0, 191, 255, 0.8)"
                                });
                            }
                        }

                        ctx.restore();
                    }
                }

                if (handResults.landmarks.length >= 2) {
                    if (detectNamaste(handResults.landmarks)) {
                        currentGesture = "Namaste / Anjali Mudra";

                        // Screenshot Logic
                        namasteHoldTimeRef.current += 16; // ~16ms per frame
                        if (namasteHoldTimeRef.current > 1000) { // 1 second hold
                            takeScreenshot();
                            namasteHoldTimeRef.current = 0; // Reset
                        }
                    } else {
                        namasteHoldTimeRef.current = 0;
                    }
                } else {
                    namasteHoldTimeRef.current = 0;
                }
            }

            // --- ELEMENTAL PARTICLES UPDATE & DRAW ---
            if (particlesRef.current.length > 0) {
                ctx.save();
                particlesRef.current.forEach((p, i) => {
                    p.x += p.vx;
                    p.y += p.vy;
                    p.life -= 0.02;

                    if (p.type === "nature") {
                        // Leaf movement
                        p.x += Math.sin(Date.now() / 200 + i) * 1;
                    }

                    ctx.fillStyle = p.color;
                    ctx.globalAlpha = p.life;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.type === "water" ? 3 : 4, 0, 2 * Math.PI);
                    ctx.fill();
                });
                // Remove dead particles
                particlesRef.current = particlesRef.current.filter(p => p.life > 0);
                ctx.restore();
            }

            // Face Logic
            if (faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
                const face = faceResults.faceLandmarks[0];
                const analysis = analyzeFace(face);
                isEyesClosed = analysis.isEyesClosed;

                // --- THIRD EYE INTERFACE ---
                const forehead = face[10];
                const fx = forehead.x * width;
                const fy = forehead.y * height;
                const gazeX = analysis.gazeX || 0;

                // Calculate Target
                let targetX = width / 2 + gazeX * (width * 0.8);
                let targetY = height * 0.6;
                let targetLabel = null;

                if (gazeX < -0.3) targetLabel = "Right"; // Screen Left
                else if (gazeX > 0.3) targetLabel = "Left"; // Screen Right

                thirdEyeRef.current.target = targetLabel;

                // Draw Beam & Reticle
                const beamColor = "rgba(255, 0, 255, 0.4)"; // Purple
                const reticleColor = targetLabel ? "rgba(0, 255, 255, 0.8)" : "rgba(255, 255, 255, 0.3)";

                ctx.save();
                ctx.beginPath();
                ctx.moveTo(fx, fy);
                ctx.lineTo(targetX, targetY);
                ctx.strokeStyle = beamColor;
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 15]); // Dashed beam
                ctx.stroke();
                ctx.setLineDash([]);

                // Reticle
                ctx.beginPath();
                ctx.arc(targetX, targetY, 15, 0, 2 * Math.PI);
                ctx.strokeStyle = reticleColor;
                ctx.lineWidth = 2;
                ctx.stroke();

                if (targetLabel) {
                    ctx.fillStyle = "cyan";
                    ctx.font = "12px monospace";
                    ctx.fillText(targetLabel === "Left" ? "FOCUS LEFT" : "FOCUS RIGHT", targetX - 40, targetY + 30);

                    // Locked Indicator
                    ctx.beginPath();
                    ctx.arc(targetX, targetY, 5, 0, 2 * Math.PI);
                    ctx.fillStyle = "white";
                    ctx.fill();
                }
                ctx.restore();
            }

            // Meditation Logic (Stabilized)
            if (isEyesClosed) {
                eyesOpenTimeRef.current = 0; // Reset open timer
                if (eyesClosedTimeRef.current === 0) eyesClosedTimeRef.current = now;

                // Trigger meditation after 1 second of eyes closed
                if (now - eyesClosedTimeRef.current > 1000) {
                    isMeditationRef.current = true;
                }
            } else {
                // Eyes are Open (or lost)
                eyesClosedTimeRef.current = 0; // Reset closed timer

                if (isMeditationRef.current) {
                    // If currently meditating, use a SAFER safety buffer (2000ms)
                    // This prevents "flickering" off due to camera noise
                    if (eyesOpenTimeRef.current === 0) eyesOpenTimeRef.current = now;

                    if (now - eyesOpenTimeRef.current > 2000) {
                        isMeditationRef.current = false;

                        // Only announce "Yoga Stopped" if NO gesture is active
                        if (!currentGesture) {
                            const msg = "Yoga band ho gaya hai. Meditation stopped.";
                            setFeedback(msg);
                            speak(msg);
                        } else {
                            setFeedback("Meditation ended. Maintaining Yoga pose.");
                        }
                    }
                } else {
                    isMeditationRef.current = false;
                }
            }

            // Gesture State Update with Hysteresis (Debounce)
            if (currentGesture) {
                if (pendingGestureRef.current !== currentGesture) {
                    // New potential gesture detected, start timer
                    pendingGestureRef.current = currentGesture;
                    pendingGestureStartTimeRef.current = now;
                } else {
                    // Same pending gesture, check duration
                    if (now - pendingGestureStartTimeRef.current > 500) { // 500ms stability required
                        if (gestureRef.current !== currentGesture) {
                            // Confirmed new gesture
                            gestureRef.current = currentGesture;
                            setGesture(currentGesture);

                            const isGyan = currentGesture === "Gyan Mudra";
                            const msg = generateSmartCoachMessage(energiesRef.current, "Calm", isMeditationRef.current, isGyan);

                            // Prevent repeating the same message too soon (10s)
                            if (msg !== lastSpeechTextRef.current || now - lastSpeechTimeRef.current > 10000) {
                                setFeedback(msg);
                                speak(msg);
                                lastSpeechTextRef.current = msg;
                                lastSpeechTimeRef.current = now;
                            }
                        }
                    }
                }
            } else {
                // No gesture detected, reset pending if it was something
                pendingGestureRef.current = null;
                pendingGestureStartTimeRef.current = 0;
            }

            if (isMeditationRef.current && gestureRef.current !== "Meditation") {
                // Trigger speech for meditation start
                gestureRef.current = "Meditation";
                setGesture("Meditation");
                const msg = "Deep meditation detected. Your energy is rising rapidly.";

                if (msg !== lastSpeechTextRef.current || now - lastSpeechTimeRef.current > 10000) {
                    setFeedback(msg);
                    speak(msg);
                    lastSpeechTextRef.current = msg;
                    lastSpeechTimeRef.current = now;
                }
            }

            // --- XP & LEVEL LOGIC ---
            let xpGain = 0.0;
            let warning = null;
            const currentLevel = levelRef.current;
            const XP_PER_LEVEL = 150;
            const MAX_LEVEL = 20;

            // Base XP for Posture (Proxy using face detection for now)
            if (faceResults.faceLandmarks.length > 0) {
                xpGain += 1.0;
            }

            // Bonus XP for Mudra
            if (currentGesture) {
                xpGain += 1.0;
            }

            // Bonus XP for Eyes Closed
            if (isMeditationRef.current) {
                xpGain += 2.0;
            }

            // Level Gating Logic
            if (currentLevel >= 3 && currentLevel < 10) {
                if (!currentGesture) {
                    xpGain = 0;
                    warning = "MUDRA REQUIRED TO PROGRESS!";
                }
            } else if (currentLevel >= 10) {
                if (!isMeditationRef.current) {
                    xpGain = 0;
                    warning = "CLOSE EYES TO PROGRESS!";
                }
            }

            // Apply XP
            if (currentLevel < MAX_LEVEL) {
                xpRef.current += xpGain;

                // Check Level Up
                const calculatedLevel = Math.floor(xpRef.current / XP_PER_LEVEL) + 1;
                const cappedLevel = Math.min(MAX_LEVEL, calculatedLevel);

                if (cappedLevel > levelRef.current) {
                    levelRef.current = cappedLevel;
                    const msg = `Congratulations! You reached Level ${cappedLevel}.`;
                    setFeedback(msg);
                    speak(msg);
                }
            }
            warningMsgRef.current = warning;

            // 3. Logic: Aura & Energy
            const isYogaMode = !!currentGesture || isMeditationRef.current;

            // Aura Dynamics
            if (isYogaMode) {
                auraIntensityRef.current = Math.min(1.0, auraIntensityRef.current + 0.08); // Faster Rise
            } else {
                auraIntensityRef.current = Math.max(0.0, auraIntensityRef.current - 0.08); // Faster Fade
            }

            // Energy Dynamics (Strict Caps & Rapid Decay)
            const energies = energiesRef.current;
            let allBalanced = true;

            // Determine Max Cap based on State
            let energyCap = 0.1; // Default: Distracted / No Focus
            let decayRate = 0.05; // Rapid Decay by default

            // [NEW] Energy Peak & Applause Logic
            let riseRate = 0.005; // Default slow rise

            if (isMeditationRef.current && gestureRef.current) {
                // "Deep Meditation" State
                energyCap = 1.0;
                riseRate = 0.05; // Extremely fast rise (Peak)

                // Trigger Applause if not already triggered recently
                if (energiesRef.current[0] > 0.95) {
                    const now = Date.now();
                    if (now - lastSpeechTimeRef.current > 15000) { // 15s cooldown
                        const msg = "Excellent! Deep Meditation Achieved.";
                        setFeedback(msg);
                        speak(msg);
                        lastSpeechTimeRef.current = now;
                        lastSpeechTextRef.current = msg;
                    }
                }
            } else if (isMeditationRef.current) {
                energyCap = 1.0; // 100% - Eyes Closed (Meditation)
                decayRate = 0.0; // No decay, rising
                riseRate = 0.02; // Fast rise
            } else if (currentGesture) {
                energyCap = 0.6; // 60% - Mudra Active
                decayRate = 0.01; // Slow decay if fluctuating
                riseRate = 0.01; // Moderate rise
            } else if (faceResults.faceLandmarks.length > 0) {
                energyCap = 0.3; // 30% - Face Detected (Good Posture Proxy)
                decayRate = 0.02; // Moderate decay
                riseRate = 0.005; // Slow rise
            }

            // Apply Energy Logic
            for (let i = 0; i < 7; i++) {
                if (energies[i] < energyCap) {
                    // Specific Chakra Boost for Mudras
                    let currentRise = riseRate;
                    if (currentGesture === "Gyan Mudra" && (i === 0 || i === 6)) {
                        currentRise = 0.01; // Boost Root & Crown
                    }

                    energies[i] = Math.min(energyCap, energies[i] + currentRise);
                } else if (energies[i] > energyCap) {
                    // Decay Logic (Rapid Drop if cap lowered)
                    energies[i] = Math.max(energyCap, energies[i] - decayRate);
                }

                if (energies[i] < 0.95) allBalanced = false;
            }

            // Check for Full Balance Event
            if (allBalanced && gestureRef.current !== "Balanced") {
                gestureRef.current = "Balanced";
                const msg = "All Chakras are perfectly balanced. You are in harmony.";
                setFeedback(msg);
                speak(msg);
            }

            // Sync UI occasionally (every 10 frames)
            if (Math.floor(t * 30) % 10 === 0) {
                setUiEnergies([...energies]);

                // Update Session Time
                const elapsedMin = (Date.now() - startTimeRef.current) / 60000;
                setSessionTime(`${elapsedMin.toFixed(1)} min`);

                // Update Mood & Posture
                if (isMeditationRef.current) {
                    setMood("Deep Peace");
                    setPosture("Lotus");
                } else if (currentGesture) {
                    setMood("Focused");
                    setPosture("Asana");
                } else if (faceResults.faceLandmarks.length > 0) {
                    setMood("Calm");
                    setPosture("Good");
                } else {
                    setMood("Distracted");
                    setPosture("Poor");
                }
            }

            // 4. Render UI Overlays
            // Level Progress Bar
            const progress = (xpRef.current % XP_PER_LEVEL) / XP_PER_LEVEL * 100;
            setLevelProgress(progress);
            setLevel(levelRef.current);
            setWarningMsg(warning);

            // Aura
            if (auraIntensityRef.current > 0.01) {
                ctx.save();
                const cx = width / 2;
                const cy = height / 2;
                const maxRadius = Math.max(width, height) * 0.8;
                const gradient = ctx.createRadialGradient(cx, cy, 100, cx, cy, maxRadius);
                gradient.addColorStop(0, `rgba(255, 215, 0, ${auraIntensityRef.current * 0.3})`); // Gold center
                gradient.addColorStop(0.5, `rgba(255, 140, 0, ${auraIntensityRef.current * 0.1})`); // Orange mid
                gradient.addColorStop(1, "rgba(0, 0, 0, 0)");

                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, width, height);
                ctx.restore();
            }

            // Speed factor depends on yoga mode
            const speedFactor = isYogaMode ? 2.0 : 0.5;

            drawUniverse(ctx, width, height, t, speedFactor);
            const breathFactor = 1.0 + 0.1 * Math.sin(t * 0.8);

            drawChakras(
                ctx,
                width * 0.5,
                height * 0.2,
                height * 0.8,
                activeIndexRef.current,
                energies,
                breathFactor,
                t
            );

            // --- KUMBHAKA (Breath Retention) LOGIC ---
            let isTouchingNose = false;
            if (faceResults.faceLandmarks.length > 0 && handResults.landmarks.length > 0) {
                const noseTip = faceResults.faceLandmarks[0][4]; // Nose Tip

                for (const hand of handResults.landmarks) {
                    const indexTip = hand[8]; // Index Finger Tip
                    const thumbTip = hand[4]; // Thumb Tip

                    // Check distance to Index or Thumb
                    const distIndex = Math.hypot(noseTip.x - indexTip.x, noseTip.y - indexTip.y);
                    const distThumb = Math.hypot(noseTip.x - thumbTip.x, noseTip.y - thumbTip.y);

                    if (distIndex < 0.05 || distThumb < 0.05) { // Threshold
                        isTouchingNose = true;
                        break;
                    }
                }
            }

            // Update Prana Value
            if (isTouchingNose) {
                pranaRef.current = Math.min(100, pranaRef.current + 0.5);
            } else {
                pranaRef.current = Math.max(0, pranaRef.current - 1.0);
            }

            // Draw Kumbhaka Bar
            if (pranaRef.current > 0) {
                const barW = 300;
                const barH = 20;
                const barX = canvas.width / 2 - barW / 2;
                const barY = canvas.height - 150;

                // Label
                ctx.fillStyle = "rgba(0, 255, 255, 1)";
                ctx.font = "bold 16px monospace";
                ctx.textAlign = "center";
                ctx.fillText(isTouchingNose ? "KUMBHAKA ACTIVE: HOLD BREATH" : "PRANA DISSIPATING...", canvas.width / 2, barY - 10);

                // Bar Background
                ctx.fillStyle = "rgba(0, 50, 50, 0.8)";
                ctx.fillRect(barX, barY, barW, barH);
                ctx.strokeStyle = "rgba(0, 255, 255, 0.5)";
                ctx.strokeRect(barX, barY, barW, barH);

                // Bar Fill
                ctx.fillStyle = isTouchingNose ? "rgba(0, 255, 255, 1)" : "rgba(0, 150, 150, 0.8)";
                ctx.fillRect(barX, barY, (pranaRef.current / 100) * barW, barH);

                // Glow
                if (isTouchingNose) {
                    ctx.shadowColor = "rgba(0, 255, 255, 0.8)";
                    ctx.shadowBlur = 15;
                    ctx.fillRect(barX, barY, (pranaRef.current / 100) * barW, barH);
                    ctx.shadowBlur = 0;
                }
            }

            requestRef.current = requestAnimationFrame(animate);
        };

        requestRef.current = requestAnimationFrame(animate);

        return () => {
            if (requestRef.current) {
                cancelAnimationFrame(requestRef.current);
            }
        };
    }, [handLandmarker, faceLandmarker]);

    return (
        <div className="relative w-full h-screen bg-black overflow-hidden font-sans select-none">
            {isLoading && (
                <div className="absolute z-50 top-0 left-0 w-full h-full bg-black/90 flex flex-col items-center justify-center gap-4">
                    <div className="w-16 h-16 border-4 border-green-500 border-t-transparent rounded-full animate-spin"></div>
                    <div className="text-green-400 text-xl font-light tracking-widest animate-pulse">
                        INITIALIZING AI MODELS...
                    </div>
                </div>
            )}

            {/* Background: Universe/Nebula */}
            <canvas
                ref={universeRef}
                className="absolute top-0 left-0 w-full h-full object-cover opacity-60"
            />

            {/* Video Feed */}
            <video
                ref={videoRef}
                className="absolute top-0 left-0 w-full h-full object-cover opacity-90 mix-blend-screen"
                autoPlay
                playsInline
                muted
            />

            {/* Overlays */}
            <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none z-10"
            />

            {/* Top Bar (Session Info) */}
            <TopBar
                sessionTime={sessionTime}
                mood={mood}
                posture={posture}
                alignmentMode={alignmentMode}
            />

            {/* Level Progress Bar (Top Right Center) */}
            <div className="absolute top-24 right-80 w-80 z-20 pointer-events-none flex flex-col items-end">
                <div className="flex justify-between items-end mb-1 w-full">
                    <span className="text-yellow-400 font-bold text-3xl tracking-widest drop-shadow-[0_0_10px_rgba(234,179,8,0.8)] animate-pulse">LEVEL {level}</span>
                    <span className="text-yellow-200/80 font-mono text-sm tracking-widest">{Math.floor(levelProgress)}%</span>
                </div>
                <div className="h-8 w-full bg-black/80 border-2 border-yellow-500 rounded-lg overflow-hidden shadow-[0_0_25px_rgba(234,179,8,0.6)] relative backdrop-blur-sm">
                    {/* Background Glow */}
                    <div className="absolute inset-0 bg-yellow-900/20"></div>

                    {/* Progress Fill */}
                    <div
                        className="h-full bg-gradient-to-r from-yellow-700 via-yellow-500 to-yellow-200 relative overflow-hidden transition-all duration-300 ease-out"
                        style={{ width: `${levelProgress}%`, boxShadow: '0 0 20px rgba(234, 179, 8, 0.8)' }}
                    >
                        {/* Shine Effect */}
                        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-white/40 to-transparent opacity-50"></div>
                        <div className="absolute top-0 -left-full w-full h-full bg-gradient-to-r from-transparent via-white/40 to-transparent animate-[shimmer_2s_infinite]"></div>
                    </div>

                    {/* Scanline effect */}
                    <div className="absolute inset-0 bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.4)_50%)] bg-[length:4px_4px] pointer-events-none opacity-70"></div>

                    {/* Glass Reflection */}
                    <div className="absolute top-0 left-0 w-full h-1/2 bg-gradient-to-b from-white/10 to-transparent pointer-events-none"></div>
                </div>

                {/* Mudra Required Warning */}
                {warningMsg && (
                    <div className="mt-2 bg-red-600/90 border-2 border-red-500 text-white px-4 py-1 font-bold text-sm tracking-widest uppercase shadow-[0_0_20px_rgba(220,38,38,0.6)] animate-bounce">
                        {warningMsg}
                    </div>
                )}

                {/* Level Titles */}
                {level >= 6 && level <= 10 && (
                    <div className="mt-2 text-yellow-300 font-bold text-lg tracking-[0.2em] drop-shadow-[0_0_10px_rgba(255,215,0,0.8)] animate-pulse">
                        YOG GURU
                    </div>
                )}
                {level >= 11 && level <= 15 && (
                    <div className="mt-2 text-cyan-300 font-bold text-lg tracking-[0.2em] drop-shadow-[0_0_10px_rgba(0,255,255,0.8)] animate-pulse">
                        TRUE YOGI
                    </div>
                )}
                {level >= 16 && (
                    <div className="mt-2 text-purple-300 font-bold text-xl tracking-[0.2em] drop-shadow-[0_0_15px_rgba(255,0,255,0.8)] animate-pulse">
                        MASTER YOGI 🏆
                    </div>
                )}
            </div>

            {arduinoData.isConnected && (
                <div className="absolute top-4 left-20 z-20 animate-slide-in-left">
                    <BioAnalyticsPanel
                        heartRate={arduinoData.heartRate}
                        spo2={arduinoData.spo2}
                        beatDetected={arduinoData.beatDetected}
                        energyLevel={uiEnergies[3]} // Heart Chakra Energy
                        stressLevel={1.0 - (uiEnergies[6] || 0.5)} // Inverse of Crown
                        focusScore={uiEnergies[5] || 0.5} // Third Eye
                        isConnected={arduinoData.isConnected}
                        hrvIndex={arduinoData.hrvIndex}
                        doshas={arduinoData.doshas}
                        insightText={arduinoData.insightText}
                        finding={arduinoData.finding}
                    />
                </div>
            )}

            {/* Right Sidebar (Mudras & Guide) */}
            <RightSidebar activeGesture={gesture} />

            {/* Bottom Overlay (Feedback) */}
            <BottomOverlay
                feedback={feedback}
                gesture={gesture}
                logs={logs}
            />

            {/* Center Title & Hint Overlay */}
            <div className="absolute bottom-12 left-1/2 -translate-x-1/2 text-center z-20 pointer-events-none">
                <h1 className="text-yellow-400 font-bold text-3xl uppercase tracking-widest drop-shadow-[0_4px_8px_rgba(0,0,0,0.8)] mb-1">
                    AI ChakraFlow
                </h1>
                <div className="text-yellow-200 text-sm font-mono tracking-wider uppercase mb-4">
                    Meditation & Mudra Engine
                </div>
                <div className="bg-black/60 backdrop-blur-md border border-yellow-500/50 px-6 py-2 rounded-full inline-block shadow-[0_0_20px_rgba(234,179,8,0.3)]">
                    <span className="text-yellow-300 font-bold">Hint: </span>
                    <span className="text-white">Try Gyan Mudra for Crown Chakra</span>
                </div>
            </div>

            {!isPlaying && (
                <div className="absolute z-50 top-0 left-0 w-full h-full bg-black/80 backdrop-blur-sm flex items-center justify-center">
                    <button
                        onClick={toggleAudio}
                        className="
                                group relative px-10 py-5 bg-transparent overflow-hidden rounded-full
                                border border-green-500/50 text-white shadow-[0_0_40px_rgba(34,197,94,0.2)]
                                transition-all duration-500 hover:shadow-[0_0_60px_rgba(34,197,94,0.4)] hover:border-green-400
                            "
                    >
                        <div className="absolute inset-0 w-full h-full bg-green-600/20 group-hover:bg-green-600/30 transition-all duration-500"></div>
                        <span className="relative text-2xl font-light tracking-widest flex items-center gap-4">
                            <span>START JOURNEY</span>
                            <span className="text-3xl">🕉️</span>
                        </span>
                    </button>
                </div>
            )}

            {/* Controls (Voice & Connect) */}
            <div className="absolute bottom-8 right-8 flex flex-col gap-4 z-50">
                {/* Connect Sensor Button */}
                {!arduinoData.isConnected && (
                    <button
                        onClick={connectArduino}
                        className="bg-black/60 backdrop-blur-md border border-green-500/50 text-green-400 px-4 py-3 rounded-full font-bold hover:bg-green-500/20 transition-all flex items-center gap-2 shadow-[0_0_15px_rgba(34,197,94,0.3)]"
                    >
                        <span>🔌</span>
                        Connect Sensor
                    </button>
                )}

                {/* Voice Toggle */}
                <button
                    onClick={toggleListening}
                    className={`
                            w-16 h-16 rounded-full flex items-center justify-center backdrop-blur-md border-2 transition-all duration-500 shadow-[0_0_30px_rgba(0,0,0,0.5)]
                            ${isListening
                            ? "bg-green-500/20 border-green-400 text-green-400 animate-pulse shadow-[0_0_30px_rgba(74,222,128,0.4)]"
                            : "bg-white/10 border-white/20 text-white/60 hover:bg-white/20 hover:scale-105"
                        }
                        `}
                >
                    {isListening ? (
                        <div className="flex gap-1 items-center">
                            <div className="w-1 h-4 bg-green-400 animate-[bounce_1s_infinite]"></div>
                            <div className="w-1 h-6 bg-green-400 animate-[bounce_1.2s_infinite]"></div>
                            <div className="w-1 h-4 bg-green-400 animate-[bounce_1s_infinite]"></div>
                        </div>
                    ) : (
                        <span className="text-2xl">🎙️</span>
                    )}
                </button>
            </div>

            {/* Error Toast */}
            {arduinoError && (
                <div className="absolute top-24 left-1/2 -translate-x-1/2 bg-red-500/90 text-white px-6 py-3 rounded-lg shadow-xl z-50 animate-bounce">
                    ⚠️ {arduinoError}
                </div>
            )}

            {/* Draw Namaste Progress Bar (Screenshot) */}
            {namasteHoldTimeRef.current > 100 && (
                <div className="absolute bottom-24 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 z-50">
                    <div className="text-white font-bold text-xs tracking-widest animate-pulse">HOLD FOR SCREENSHOT...</div>
                    <div className="w-48 h-2 bg-black/50 rounded-full overflow-hidden border border-white/20">
                        <div
                            className="h-full bg-green-500 transition-all duration-75 ease-linear"
                            style={{ width: `${Math.min(100, (namasteHoldTimeRef.current / 1000) * 100)}%` }}
                        ></div>
                    </div>
                </div>
            )}

            {/* Screenshot Flash Overlay */}
            {screenshotFlash && (
                <div className="absolute inset-0 bg-white z-[100] animate-flash pointer-events-none"></div>
            )}
            {screenshotFlash && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-black/80 text-green-400 px-8 py-4 rounded-xl border border-green-500 shadow-2xl z-[101] flex flex-col items-center animate-bounce">
                    <span className="text-4xl">📸</span>
                    <span className="text-xl font-bold tracking-widest mt-2">SCREENSHOT SAVED!</span>
                </div>
            )}
        </div>
    );
}
