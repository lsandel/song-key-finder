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
    print("🟢 RECORDING COMPLETE. Analyzing instantly...")
    return np.squeeze(audio_data)

def extract_clean_notes(audio_data, sr):
    # Short-Time Fourier Transform
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=2048, hop_length=512)
    mean_chroma = np.mean(chroma, axis=1)
    
    # Increase the noise floor gate to 40% to filter out acoustic room clutter
    max_val = np.max(mean_chroma) if np.max(mean_chroma) > 0 else 1
    mean_chroma = np.where(mean_chroma < (max_val * 0.40), 0, mean_chroma)
    
    mean_chroma = (mean_chroma / max_val) * 100
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return {note_names[i]: int(mean_chroma[i]) for i in range(12) if mean_chroma[i] > 0}

def query_local_llm(note_profile):
    # We alter the prompt to force the LLM to skip conversational fluff entirely
    prompt = f"""
    You are a strict, no-nonsense guitar scale calculator. 
    Analyze these musical note intensities (0-100) captured from a song: {note_profile}
    
    Identify the most prominent notes, deduce the musical key, and output the answer immediately.
    Do NOT write an introduction. Do NOT write "Step 1" or "Step 2". Get straight to the results.
    
    Format your response EXACTLY like this:
    ### 🔑 KEY: [Detected Key, e.g., G Major]
    * **Jam Scale:** [e.g., G Major Pentatonic / E Minor Pentatonic]
    * **Root Fret:** [Fret number on the Low E string to start playing]
    * **Brief Reason:** [1 short sentence explaining why these notes fit that scale]
    """
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Keep it purely mathematical/logical
            # "num_predict" is completely REMOVED so it will never cut off mid-sentence
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json().get("response", "Error reading response.")
    except Exception as e:
        return f"ERROR: Could not communicate with local Ollama server. {e}"

if __name__ == "__main__":
    print("=== TUNED GUITAR JAM BACKEND ===")
    audio = record_audio(DURATION, SAMPLE_RATE)
    notes = extract_clean_notes(audio, SAMPLE_RATE)
    
    print(f"Cleaned note profiles (0-100): {notes}")
    verdict = query_local_llm(notes)
    print("\n=== JAM RESULTS ===")
    print(verdict)