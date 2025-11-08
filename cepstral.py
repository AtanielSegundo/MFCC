import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import os

try:
	import torch
	import torchaudio.transforms as T
except Exception as e:
    print('[TORCH IMPORT] ',e)

ms = lambda t : t * 1e-3
f_to_qf = lambda f : 2595 * np.log10(1 + f/700)
qf_to_f = lambda qf : 700 * (10**(qf/2595) - 1)
BWcr = lambda f : 25 + 75*(1+ 1.4*(f/1000)**2)**(0.69) #(PICONE 1993, RABINER 1993)

class Signal:
	def __init__(self,S:np.ndarray,Fs,duration_s):
		self.fs = Fs
		self.duration_s = duration_s
		self.samples = S
	def __call__(self):
		return self.samples

def plot_discrete_signal(X,Y,xlabel=None,ylabel=None,
						 title=None,
						 max_points=None):
	if(len(X) != len(Y)):
		print(f"SHAPE X:{X.shape} ~= Y:{Y.shape}")
		return
	limit = max_points or len(X)
	markerline, stemlines, baseline = plt.stem(X[:limit], Y[:limit])
	baseline.set_visible(False)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	
def gen_signal(Fs:float,duration_s:float,
			   Fcarrier=2000.0,Fmod=5.0
			   ) -> np.ndarray:
	N = Fs*duration_s
	n = np.arange(N) / Fs
	return (1 + 0.5 * np.sin(2 * np.pi * Fmod * n)) * np.sin(2 * np.pi * Fcarrier * n)

def signal_windowing(S:Signal,
                     window_dt: float, hop_dt: float,
					 window_fn = np.hamming
					 ) -> np.ndarray:
    """
    Splits a 1D signal into overlapping frames

    Args:
        X (np.ndarray): Input signal (1D).
        Fs (float): Sampling rate in Hz.
        signal_duration (float): Duration of the signal in seconds.
        window_dt (float): Window length in seconds.
        hop_dt (float): Hop size (frame shift) in seconds.

    Returns:
        np.ndarray: 2D array of shape (n_frames, n_samples_per_frame)
                    containing the windowed signal.
    """
    signal_duration = S.duration_s
    Fs = S.fs
    X = S.samples

    # --- Compute parameters ---
    n_frames = int(1 + np.floor((signal_duration - window_dt) / hop_dt))
    n_window_samples = int(np.floor(window_dt * Fs))
    n_hop_samples    = int(np.floor(hop_dt * Fs))

    # --- Check signal length ---
    expected_length = (n_frames - 1) * n_hop_samples + n_window_samples
    if len(X) < expected_length:
        raise ValueError("Input signal X is too short for the given parameters.")

    # --- Create windowed view using strides ---
    shape = (n_frames, n_window_samples)
    strides = (X.strides[0] * n_hop_samples, X.strides[0])
    windowed_signal = as_strided(X, shape=shape, strides=strides).copy()

	# --- Apply window function ---
    w = window_fn(n_window_samples)
    windowed_signal = windowed_signal * w

    return windowed_signal

def bins_to_freqs(N,Fs):
	inv = Fs / N 
	return np.arange(0,(N//2 + 1)) * inv

def mel_filterbank(Fs, N_fft, N_filters, f_min=0.0, f_max=None):
    if f_max is None:
        f_max = Fs / 2.0

    mel_min, mel_max = f_to_qf(f_min), f_to_qf(f_max)
    mel_points = np.linspace(mel_min, mel_max, N_filters + 2)

    hz_points = qf_to_f(mel_points)
    # (N_fft + 1) is used to ensure correctly Nyquist map 
	# in the last frequency
    bins = np.floor((N_fft + 1) * hz_points / Fs).astype(int)

    n_bins = N_fft // 2 + 1
    M = np.zeros((N_filters, n_bins))

    for m in range(1, N_filters + 1):
        left, center, right = bins[m-1], bins[m], bins[m+1]
        if center == left:
            left_slope = np.zeros(0)
        else:
            denom = center - left
            for k in range(left, center):
                if 0 <= k < n_bins:
                    M[m-1, k] = (k - left) / denom
        if right == center:
            pass
        else:
            denom = right - center
            for k in range(center, right):
                if 0 <= k < n_bins:
                    M[m-1, k] = (right - k) / denom

    return M  # shape: (N_filters, N_fft//2 + 1)

def plot_mfcc_spectogram(mfcc_m_k,hop_dt,cmap='viridis',
                         title=None,xlabel=None,ylabel=None
						 ):
	time_axis = np.arange(mfcc_m_k.shape[0]) * hop_dt
	mfcc_m_k_no0 = mfcc_m_k[:,1:]
	plt.figure(figsize=(12, 6))
	plt.imshow(mfcc_m_k_no0.T, aspect='auto', origin='lower', cmap=cmap,
           	   extent=[time_axis[0], time_axis[-1],0,mfcc_m_k.shape[1]-1])
	plt.yticks(np.arange(0, mfcc_m_k_no0.shape[1]))
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.colorbar()
	plt.title(title)
	
def get_mfcc_fftpack(signal:Signal,
                     window_dt,hop_dt,
					 N_fft,N_filters,N_mels,
                     bank_min_f,bank_max_f
					 ):
	
	s_m_n = signal_windowing(signal,window_dt,hop_dt)
	S_m_k = np.fft.rfft(s_m_n,axis=1,n=N_fft)[:,:N_fft//2 + 1]
	print(f"windowed signal shape: ",s_m_n.shape)
	print(f"Fourier signal: ",S_m_k.shape)
	
	freq_bins = bins_to_freqs(N_fft,signal.fs)

	plt.cla()
	plot_discrete_signal(freq_bins,np.abs(np.sum(S_m_k,axis=0)))
	plt.savefig(f"output/signal_frequency.png")
    
	f_i_k = mel_filterbank(signal.fs,N_fft,N_filters,bank_min_f,bank_max_f)
    
	plt.cla()
	[plt.plot(freq_bins,f_i_k[i,:]) for i in range(N_filters)]
	plt.xlim((bank_min_f,bank_max_f*1.2))
	plt.savefig(f"output/filter_bank.png")
     
	E_m_i = np.dot(np.abs(S_m_k)**2, f_i_k.T)
	E_m_i_log = np.log(np.maximum(E_m_i, 1e-10))
     
	assert N_mels <= N_filters
	mfcc_m_k = dct(E_m_i_log, type=2, axis=1, norm='ortho')
	print(f"N Cepstral is: {N_mels} | mfcc_m_k shape is: {mfcc_m_k.shape}")
    
	return mfcc_m_k[:,:N_mels]

def get_mfcc_torchaudio(signal:Signal,
                     window_dt,hop_dt,
					 N_fft,N_filters,N_mels,
                     bank_min_f,bank_max_f,
					 ):
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  signal_tensor = torch.from_numpy(signal()).float().to(device)
    
  mfcc_transform = T.MFCC(
		sample_rate=signal.fs,
		n_mfcc=N_mels,
		log_mels=True,
		melkwargs={
			'n_fft': N_fft,
			'hop_length': int(signal.fs*hop_dt),
			'win_length': int(signal.fs*window_dt),
			'window_fn': torch.hamming_window,
			'n_mels': N_filters,
			'f_min': bank_min_f,
			'f_max': bank_max_f,
			'power': 2.0,
			'norm': None,
		}
		
	)
    
  torchaudio_mfcc = mfcc_transform(signal_tensor)
  torchaudio_mfcc_np = torchaudio_mfcc.cpu().numpy()
      
  return torchaudio_mfcc_np.T

def _dct_type2_ortho(x, axis=1):
    """
    DCT-II (type=2) with 'ortho' normalization implemented with numpy.
    x: array with shape (..., N, ...) where the transform is along `axis`.
    Returns array with same shape.
    """
    x = np.asarray(x, dtype=float)
    if axis != 1:
        x = np.moveaxis(x, axis, 1)
        moved = True
    else:
        moved = False

    if x.ndim != 2:
        front_shape = x.shape[0]
        newshape = (x.shape[0], x.shape[1])
    
    n_frames, N = x.shape

    n = np.arange(N)
    k = np.arange(N)[:, None]  # shape (N,1)
    # kernel: cos(pi * (2n + 1) * k / (2N)), shape (N, N) with rows = k, cols = n
    phi = np.cos(np.pi * (2 * n + 1) * k / (2.0 * N))  # shape (N, N)

    alpha = np.sqrt(2.0 / N) * np.ones(N)
    alpha[0] = np.sqrt(1.0 / N)

    res = x.dot(phi.T) * alpha  # shape (n_frames, N)

    if moved:
        res = np.moveaxis(res, 1, axis)
    return res


def get_mfcc(signal:Signal,
             window_dt,hop_dt,
             N_fft,N_filters,N_mels,
             bank_min_f,bank_max_f):
    """
    Numpy-only MFCC:
    - windowing (signal_windowing)
    - rFFT -> power spectrum
    - mel filterbank multiplication -> filter energies
    - log energy (floor at 1e-10)
    - DCT-II (ortho) over filter dimension
    Returns: mfcc_m_k with shape (n_frames, N_mels)
    """
    s_m_n = signal_windowing(signal, window_dt, hop_dt)

    S_m_k = np.fft.rfft(s_m_n, n=N_fft, axis=1)[:, : (N_fft // 2 + 1)]

    power_m_k = np.abs(S_m_k) ** 2

    f_i_k = mel_filterbank(signal.fs, N_fft, N_filters, bank_min_f, bank_max_f)

    E_m_i = np.dot(power_m_k, f_i_k.T)

    E_m_i_log = np.log(np.maximum(E_m_i, 1e-10))

    mfcc_all = _dct_type2_ortho(E_m_i_log, axis=1)

    assert N_mels <= N_filters, "N_mels must be <= N_filters"
    return mfcc_all[:, :N_mels]

import time

def main():
	Fs = 22050
	duration_s = 1.0
	N_fft = 1024
	N_filters = 20
	N_mels = 16
	bank_min_f = 0
	bank_max_f = 4600
	window_dt = ms(25)
	hop_dt = ms(10)

	s_n = gen_signal(Fs, duration_s)
	plot_discrete_signal(np.arange(Fs * duration_s), s_n, max_points=500)
	plt.savefig("output/generated_signal.png")

	signal = Signal(s_n, Fs, duration_s)

	# ---------------- FFTPACK ----------------
	start = time.perf_counter()
	fftpack_mfcc_m_k = get_mfcc_fftpack(signal, window_dt, hop_dt,
										N_fft, N_filters, N_mels,
										bank_min_f, bank_max_f)
	t_fftpack = time.perf_counter() - start
	print(f"[BENCHMARK] get_mfcc_fftpack: {t_fftpack:.6f} s")

	plt.cla()
	plot_mfcc_spectogram(fftpack_mfcc_m_k, hop_dt,
						 title='fftpack MFCC',
						 xlabel='Time (s)',
						 ylabel='Coefficient')
	plt.savefig("output/fftpack_mfcc.png")

	# ---------------- TORCHAUDIO ----------------
	torch.set_default_device('cpu' if not torch.cuda.is_available() else 'cuda'); print("torch.cuda.is_available(): ",torch.cuda.is_available())
	start = time.perf_counter()
	torch_mfcc_m_k = get_mfcc_torchaudio(signal, window_dt, hop_dt,
										 N_fft, N_filters, N_mels,
										 bank_min_f, bank_max_f)
	t_torch = time.perf_counter() - start
	print(f"[BENCHMARK] get_mfcc_torchaudio: {t_torch:.6f} s")

	plt.cla()
	plot_mfcc_spectogram(torch_mfcc_m_k, hop_dt,
						 title='torch MFCC',
						 xlabel='Time (s)',
						 ylabel='Coefficient')
	plt.savefig("output/torch_mfcc.png")

	# ---------------- NUMPY-ONLY ----------------
	start = time.perf_counter()
	mfcc_m_k = get_mfcc(signal, window_dt, hop_dt,
						N_fft, N_filters, N_mels,
						bank_min_f, bank_max_f)
	t_numpy = time.perf_counter() - start
	print(f"[BENCHMARK] get_mfcc (numpy-only): {t_numpy:.6f} s")

	plt.cla()
	plot_mfcc_spectogram(mfcc_m_k, hop_dt,
						 title='numpy only MFCC',
						 xlabel='Time (s)',
						 ylabel='Coefficient')
	plt.savefig("output/numpy_only_mfcc.png")

	# ---------------- Summary ----------------
	print("\n=== MFCC Benchmark Summary ===")
	print(f"FFTpack     : {t_fftpack:.6f} s")
	print(f"Torchaudio  : {t_torch:.6f} s")
	print(f"Numpy-only  : {t_numpy:.6f} s")

if __name__ == "__main__":
	os.makedirs("output", exist_ok=True)
	main()