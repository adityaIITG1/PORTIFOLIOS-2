import os
import requests

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"

def get_ai_explanation(scripture_entry, pose_state, mudra_state, breath_state, session_context, enable_api=True):
    # INTEGRATED API KEY HERE
    api_key = "AIzaSyAHazke5EhUlp2bqfKszpi3mKw3fTYxiSw"
    
    if not enable_api or not api_key:
        return _fallback(scripture_entry)
        
    prompt = _build_prompt(scripture_entry, pose_state, mudra_state, breath_state, session_context)
    
    try:
        resp = requests.post(
            GEMINI_ENDPOINT,
            params={"key": api_key},
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            },
            timeout=8
        )
        
        if resp.status_code != 200:
            # Optional: Print error to console for debugging
            # print(f"Error {resp.status_code}: {resp.text}")
            return _fallback(scripture_entry)
            
        data = resp.json()
        candidates = data.get("candidates", [])
        
        if not candidates:
            return _fallback(scripture_entry)
            
        return candidates[0]["content"]["parts"][0]["text"]
        
    except Exception as e:
        # Optional: Print exception for debugging
        # print(f"Exception occurred: {e}")
        return _fallback(scripture_entry)


def _fallback(entry):
    return f"{entry['hinglish']}: {entry['meaning']} Keep breath soft, mind steady. No medical or spiritual guarantees—just gentle guidance."


def _build_prompt(entry, pose_state, mudra_state, breath_state, session_context):
    return f"""
You are a wise yoga teacher. Explain the scripture in 4-6 sentences.
Include a relevant short quote or wisdom from the **Vedas** or **Ramayana** to inspire the user.
Avoid medical claims, be inclusive, non-fanatical, non-political.

Scripture:
- Source: {entry['source']}
- Sanskrit: {entry['sanskrit']}
- Hinglish: {entry['hinglish']}
- Meaning: {entry['meaning']}

Current state:
- Pose: {pose_state.get('pose')}
- Mudra: {mudra_state.get('mudra')}
- Breath smoothness: {breath_state.get('smoothness'):.2f}, rate: {breath_state.get('rate'):.1f} bpm
- Pranayama count: {breath_state.get('pranayama_count')}

User: beginner home practitioner, tone: calm and friendly.
"""