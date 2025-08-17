# sonify.py
import numpy as np
import soundfile as sf
from io import BytesIO
from sklearn.cluster import KMeans
from PIL import Image

def get_dominant_colors(image_pil, n_colors=5):
    img = image_pil.convert("RGB").resize((200,200))
    arr = np.array(img).reshape(-1,3)/255.0
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(arr)
    centers = kmeans.cluster_centers_
    # return colors as RGB floats 0..1
    return (centers * 255).astype(int).tolist()

def rgb_to_freq(rgb):
    """
    Map RGB to a frequency. Simple mapping: hue -> freq range
    """
    r,g,b = rgb
    # convert to hue roughly
    mx = max(r,g,b); mn = min(r,g,b)
    if mx==mn:
        hue = 0
    else:
        if mx==r: hue = (60*((g-b)/(mx-mn)) + 360) % 360
        elif mx==g: hue = (60*((b-r)/(mx-mn)) + 120) % 360
        else: hue = (60*((r-g)/(mx-mn)) + 240) % 360
    # map hue 0..360 to 220..880 Hz
    freq = 220 + (hue/360.0)*(880-220)
    # brightness -> amplitude
    brightness = (r+g+b)/(3*255)
    amp = 0.2 + 0.8*brightness
    return float(freq), float(amp)

def synthesize_tone_sequence(freq_amp_pairs, duration_per_tone=0.7, sr=22050):
    total_dur = duration_per_tone * len(freq_amp_pairs)
    t = np.linspace(0, total_dur, int(sr*total_dur), endpoint=False)
    signal = np.zeros_like(t)
    for i,(f,a) in enumerate(freq_amp_pairs):
        start = int(i*duration_per_tone*sr)
        end = int((i+1)*duration_per_tone*sr)
        tt = t[start:end] - i*duration_per_tone
        signal[start:end] += a * np.sin(2*np.pi*f*tt) * np.exp(-3*tt)  # little envelope
    # normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    # write to BytesIO as WAV using soundfile
    buffer = BytesIO()
    sf.write(buffer, signal, sr, format='WAV')
    buffer.seek(0)
    return buffer.getvalue()
