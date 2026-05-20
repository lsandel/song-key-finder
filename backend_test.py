import time
import requests
import numpy as np
import sounddevice as sd
import librosa

# --- CONFIGURATION ---
DURATION = 8         
SAMPLE_RATE = 22050  
OLLAMA_URL = "http://localhost:11434/api/generate"

def record_audio(duration, sr):
    print(f"\n🔴 LISTENING: Play the song near your mic now ({duration}s)...")
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("🟢 RECORDING COMPLETE. Analyzing frequencies...")
    return np.squeeze(audio_data)

def extract_clean_notes(audio_data, sr):
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=2048, hop_length=512)
    mean_chroma = np.mean(chroma, axis=1)
    
    # 35% Noise gate to isolate primary foundational frequencies
    max_val = np.max(mean_chroma) if np.max(mean_chroma) > 0 else 1
    mean_chroma = np.where(mean_chroma < (max_val * 0.35), 0, mean_chroma)
    mean_chroma = (mean_chroma / max_val) * 100
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return {note_names[i]: int(mean_chroma[i]) for i in range(12) if mean_chroma[i] > 0}

def run_multi_llm_pipeline(note_profile):
    """Passes data through Llama3 first, then refines the theory via Mistral."""
    
    # -------------------------------------------------------------
    # STAGE 1: LLM #1 (Llama 3) - The Chord Extractor
    # -------------------------------------------------------------
    print("🧠 Stage 1: Llama 3 is translating raw notes into chords...")
    stage1_prompt = f"""
    You are an AI audio musicologist. Analyze these note frequency intensities (0-100):
    {note_profile}
    
    Identify which 2 to 4 primary musical chords are actively being formed by these notes.
    List only the chord names clearly. Do not determine the key signature yet.
    """
    
    payload_1 = {"model": "llama3", "prompt": stage1_prompt, "stream": False, "options": {"temperature": 0.1}}
    
    try:
        response_1 = requests.post(OLLAMA_URL, json=payload_1)
        detected_chords = response_1.json().get("response", "").strip()
        print(f"\n🎼 Llama 3 Detected Chords:\n{detected_chords}\n")
    except Exception as e:
        return f"Stage 1 Error: {e}"

    # -------------------------------------------------------------
    # STAGE 2: LLM #2 (Mistral) - The Key & Guitar Expert
    # -------------------------------------------------------------
    print("🎸 Stage 2: Mistral is calculating the song key and jam map...")
    stage2_prompt = f"""
    You are an expert guitar coach. A song analyzer just listened to a track and detected these chords:
    {detected_chords}
    
    Based on this chord progression, determine the single overall musical key signature.
    
    Format your response exactly like this:
    ### 🔑 REFINED KEY: [Detected Key]
    * **Jam Scale:** [Best scale/mode to play over these chords]
    * **Root Fret:** [Fret number on the Low E string]
    * **Theory Check:** [1 sentence explaining why these chords lock into this specific key]
    """
    
    payload_2 = {"model": "mistral", "prompt": stage2_prompt, "stream": False, "options": {"temperature": 0.2}}
    
    try:
        response_2 = requests.post(OLLAMA_URL, json=payload_2)
        final_verdict = response_2.json().get("response", "").strip()
        return final_verdict
    except Exception as e:
        return f"Stage 2 Error: {e}"

if __name__ == "__main__":
    print("=== MULTI-LLM EXPERT BACKEND ===")
    audio = record_audio(DURATION, SAMPLE_RATE)
    notes = extract_clean_notes(audio, SAMPLE_RATE)
    
    print(f"Cleaned note profiles: {notes}")
    print("-----------------------------------------")
    
    verdict = run_multi_llm_pipeline(notes)
    print("\n=== 🎯 FINAL REFINED RESULTS ===")
    print(verdict)