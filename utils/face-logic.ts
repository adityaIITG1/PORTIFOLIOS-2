import { NormalizedLandmark } from "@mediapipe/tasks-vision";

// Helper to calculate Euclidean distance
function dist(p1: NormalizedLandmark, p2: NormalizedLandmark): number {
    return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
}

// Calculate Eye Aspect Ratio (EAR)
function getEyeAspectRatio(eyeLandmarks: NormalizedLandmark[]): number {
    // Vertical distances
    const A = dist(eyeLandmarks[1], eyeLandmarks[5]);
    const B = dist(eyeLandmarks[2], eyeLandmarks[4]);
    // Horizontal distance
    const C = dist(eyeLandmarks[0], eyeLandmarks[3]);

    return (A + B) / (2.0 * C);
}

// Analyze face for eyes closed
// Returns { eyeOpen: number, isEyesClosed: boolean }
export function analyzeFace(landmarks: NormalizedLandmark[]) {
    // MediaPipe Face Mesh Indices
    // Left Eye: 33, 160, 158, 133, 153, 144
    // Right Eye: 362, 385, 387, 263, 373, 380

    const leftEyeIndices = [33, 160, 158, 133, 153, 144];
    const rightEyeIndices = [362, 385, 387, 263, 373, 380];

    const leftEye = leftEyeIndices.map(i => landmarks[i]);
    const rightEye = rightEyeIndices.map(i => landmarks[i]);

    const leftEAR = getEyeAspectRatio(leftEye);
    const rightEAR = getEyeAspectRatio(rightEye);

    const avgEAR = (leftEAR + rightEAR) / 2.0;

    // Threshold for closed eyes (tuned for webcam distance)
    const EAR_THRESHOLD = 0.30;

    const isEyesClosed = avgEAR < EAR_THRESHOLD;

    // --- Gaze Tracking Logic (Ported from main2.py) ---
    let gazeX = 0.0;
    // Check if iris landmarks exist (Standard Face Mesh has 478 landmarks, 468-477 are iris)
    if (landmarks.length > 470) {
        // Right Eye (User's Right, Screen Left)
        // Inner: 362, Outer: 263
        const r_inner = landmarks[362];
        const r_outer = landmarks[263];
        const r_iris = landmarks[473]; // Right Iris Center

        const h_dist = dist(r_inner, r_outer);

        if (h_dist > 0) {
            const eye_center_x = (r_inner.x + r_outer.x) / 2;
            // Normalize: deviation / (half_width)
            // Factor 4.0 to make it more sensitive (from main2.py)
            gazeX = (r_iris.x - eye_center_x) / (h_dist / 2) * 4.0;
        }
    }

    return { avgEAR, isEyesClosed, gazeX };
}
