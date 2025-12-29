import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from typing import List, Tuple
import soundfile as sf
import torch
import torchaudio

DEBUG_ACTIVATED = True

def DEBUG(msg, *strs):
    if DEBUG_ACTIVATED:
        print(f"[DEBUG] {msg} {' '.join(map(str,strs))}")

def read_labels_splited(text_path: str, spliter=" ") -> List[str]:
    with open(text_path, "r") as f:
        return f.read().strip().split(spliter)

def normalize_int16(y: np.ndarray) -> np.ndarray:
    return y.astype(np.float32) / (1 << 15)  # 32768

def apply_voice_filter(y: np.ndarray, sr: int, 
                       lowcut: float = 80.0, 
                       highcut: float = 6200.0) -> np.ndarray:
    """
    Aplica filtro passa-banda otimizado para voz humana.
    - Remove frequências abaixo de 80 Hz (ruído de fundo, rumble)
    - Remove frequências acima de 6200 Hz (sibilâncias, ruído de alta freq)
    - Preserva a faixa vocal fundamental e harmônicas principais
    """
    if y.ndim == 2:
        y = y.mean(axis=1)
    
    nyquist = sr / 2.0
    
    # Verifica se as frequências são válidas
    if highcut >= nyquist:
        highcut = nyquist * 0.95
        DEBUG(f"Ajustando highcut para {highcut:.0f} Hz (Nyquist: {nyquist:.0f} Hz)")
    
    # Filtro passa-banda Butterworth (4a ordem = suave, sem ringing)
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    y_filtered = filtfilt(b, a, y)
    
    # Normaliza pós-filtragem
    max_val = np.abs(y_filtered).max()
    if max_val > 0:
        y_filtered = y_filtered / max_val
    
    return y_filtered

def apply_preemphasis(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Aplica pré-ênfase para realçar componentes de alta frequência da voz.
    Fórmula: y[n] = x[n] - coef * x[n-1]
    """
    return np.append(y[0], y[1:] - coef * y[:-1])

def detect_segments_enhanced(y: np.ndarray, sr: int,
                            frame_len: float = 0.02,
                            thresh_mul: float = 2.5,
                            min_word_len: float = 0.08,
                            min_silence: float = 0.15,
                            pad: float = 0.03) -> List[Tuple[float, float]]:
    """
    Detecção melhorada de segmentos com:
    - Threshold adaptativo baseado em percentis
    - Mínimo de silêncio entre palavras
    - Remoção de segmentos muito curtos
    - Padding mais inteligente
    """
    if y.ndim == 2:
        y = y.mean(axis=1)
    N = len(y)

    # Calcula energia RMS por frame
    frame_n = max(1, int(frame_len * sr))
    sq = y.astype(np.float32) ** 2
    kernel = np.ones(frame_n) / frame_n
    energy = np.convolve(sq, kernel, mode="same")
    rms = np.sqrt(energy + 1e-16)

    # Threshold adaptativo: usa percentil 15 como piso de ruído
    noise_floor = np.percentile(rms, 15)
    speech_level = np.percentile(rms, 85)
    thresh = noise_floor + (speech_level - noise_floor) * 0.25
    thresh = max(thresh, noise_floor * thresh_mul, 1e-6)

    voiced = rms > thresh

    # Detecta transições
    diff = np.diff(voiced.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if voiced[0]:
        starts = np.concatenate(([0], starts))
    if voiced[-1]:
        ends = np.concatenate((ends, [N]))

    # Filtra segmentos curtos e aplica padding
    segments = []
    for s, e in zip(starts, ends):
        seg_dur = (e - s) / sr
        if seg_dur >= min_word_len:
            s2 = max(0, s - int(pad * sr))
            e2 = min(N, e + int(pad * sr))
            segments.append((s2 / sr, e2 / sr))

    # Mescla segmentos próximos (gap < min_silence)
    merged = []
    if segments:
        cur_s, cur_e = segments[0]
        for s, e in segments[1:]:
            gap = s - cur_e
            if gap < min_silence:
                cur_e = e  # Estende o segmento atual
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

    return merged

# Carrega modelo Silero VAD
silero_model, silero_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)
(get_speech_timestamps,
 _, read_audio,
 _, _) = silero_utils


def detect_segments_vad_neural(y: np.ndarray, sr: int,
                               threshold: float = 0.5,
                               min_speech_duration_ms: int = 50,
                               min_silence_duration_ms: int = 100) -> List[Tuple[float, float]]:
    """
    Detecta segmentos de fala via Silero VAD com parâmetros ajustáveis.
    """
    if y.ndim == 2:
        y = y.mean(axis=1)

    wav = torch.from_numpy(y.astype(np.float32))

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        sr = 16000

    wav = wav.unsqueeze(0)

    timestamps = get_speech_timestamps(
        wav, 
        silero_model, 
        sampling_rate=sr,
        return_seconds=True,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms
    )

    segments = [(t['start'], t['end']) for t in timestamps]
    return segments

def match_segments_to_labels(segments: List[Tuple[float, float]],
                             labels: List[str],
                             sr: int,
                             y: np.ndarray,
                             max_gap: float = 0.05) -> List[Tuple[str, float, float]]:
    """
    Versão melhorada: prioriza fusão de gaps pequenos primeiro,
    depois divide segmentos longos de forma mais inteligente.
    """
    segs = segments.copy()

    # Se temos mais segmentos que labels, mescla os mais próximos
    while len(segs) > len(labels) and len(segs) > 1:
        gaps = [(i, segs[i+1][0] - segs[i][1]) for i in range(len(segs)-1)]
        idx, gap_size = min(gaps, key=lambda x: x[1])
        
        # Só mescla se o gap for razoável
        if gap_size > max_gap and len(segs) == len(labels):
            break
            
        s0, e0 = segs[idx]
        s1, e1 = segs[idx+1]
        segs[idx] = (s0, e1)
        del segs[idx+1]

    # Se temos menos segmentos, divide proporcionalmente
    while len(segs) < len(labels):
        lengths = [(i, segs[i][1] - segs[i][0]) for i in range(len(segs))]
        idx, ln = max(lengths, key=lambda x: x[1])
        
        if ln < 0.06:  # Segmento muito curto para dividir
            last_end = segs[-1][1]
            new_seg = (last_end, min(last_end + 0.05, len(y)/sr))
            segs.append(new_seg)
            continue
            
        s, e = segs[idx]
        # Divide no ponto de menor energia (mais natural)
        mid_i = int(s * sr + ln * sr / 2)
        window = int(0.01 * sr)  # janela de 10ms
        start_i = max(int(s * sr), mid_i - window)
        end_i = min(int(e * sr), mid_i + window)
        
        if end_i > start_i:
            segment_audio = y[start_i:end_i] if y.ndim == 1 else y[start_i:end_i].mean(axis=1)
            energy = segment_audio ** 2
            min_energy_idx = start_i + np.argmin(energy)
            mid = min_energy_idx / sr
        else:
            mid = s + (e - s) / 2.0
            
        segs[idx] = (s, mid)
        segs.insert(idx+1, (mid, e))

    # Garante número exato
    if len(segs) > len(labels):
        segs = segs[:len(labels)]
    elif len(segs) < len(labels):
        last = segs[-1] if segs else (0.0, 0.0)
        while len(segs) < len(labels):
            s = last[1]
            e = min(s + 0.05, len(y)/sr)
            segs.append((s, e))
            last = segs[-1]

    paired = [(labels[i], segs[i][0], segs[i][1]) for i in range(len(labels))]
    return paired

def extract_segment_from_buffer(y: np.ndarray,
                                sr: int,
                                start_sec: float,
                                end_sec: float) -> np.ndarray:
    """Extrai segmento do áudio original."""
    start_i = int(start_sec * sr)
    end_i   = int(end_sec * sr)

    start_i = max(0, min(start_i, len(y)))
    end_i   = max(0, min(end_i, len(y)))

    return y[start_i:end_i]

def main(audio_path: str, text_path: str, method: str = "neural"):
    DEBUG("Passed Paths:", audio_path, text_path)
    labels_splited = read_labels_splited(text_path)
    DEBUG("Labels:", labels_splited)
    
    # Lê áudio original
    sr, y_original = wavfile.read(audio_path)
    DEBUG("SAMPLE RATE:", sr)
    DEBUG("TOTAL TIME:", f"{len(y_original)/sr:.3f}s")

    y_normalized = normalize_int16(y_original)
    
    # Aplica filtros para DETECÇÃO apenas
    DEBUG("Aplicando filtro passa-banda para voz (80-6200 Hz)...")
    y_filtered = apply_voice_filter(y_normalized, sr, lowcut=80, highcut=6200)
    
    DEBUG("Aplicando pré-ênfase...")
    y_processed = apply_preemphasis(y_filtered, coef=0.97)
    
    # Detecta segmentos no áudio PROCESSADO
    DEBUG(f"Detectando segmentos (método: {method})...")
    
    if method == "neural":
        segments = detect_segments_vad_neural(
            y_processed, sr,
            threshold=0.4,
            min_speech_duration_ms=80,
            min_silence_duration_ms=150
        )
    elif method == "enhanced":
        segments = detect_segments_enhanced(
            y_processed, sr,
            frame_len=0.02,
            thresh_mul=2.5,
            min_word_len=0.08,
            min_silence=0.15,
            pad=0.03
        )
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    DEBUG("DETECTED SEGMENTS (count):", len(segments))
    for i, (s, e) in enumerate(segments[:15]):
        DEBUG(f"seg[{i}]:", f"{s:.3f}s - {e:.3f}s (dur: {e-s:.3f}s)")

    # Faz o matching
    paired = match_segments_to_labels(segments, labels_splited, sr, y_normalized)
    
    # Extrai do áudio ORIGINAL usando os timestamps detectados
    DEBUG("\nExtraindo segmentos do áudio original...")
    for label, s, e in paired:
        seg = extract_segment_from_buffer(y_original, sr, s, e)
        sf.write(f"{label}.wav", seg, sr)
        print(f"{label} {s:.3f} {e:.3f}")

    return paired

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("Uso: python script.py <audio_path> <label_path> [method]")
        print("method: 'neural' (padrão) ou 'enhanced'")
        sys.exit(1)
    
    audio_path, label_path = args[0], args[1]
    method = args[2] if len(args) > 2 else "neural"
    main(audio_path, label_path, method)