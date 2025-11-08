import argparse
import queue
import sys
import time
import threading

import numpy as np
import sounddevice as sd
import soundcard as sc
import matplotlib.pyplot as plt

from cepstral import Signal, get_mfcc, ms


# ---------------- Ring buffer ----------------
class RingBuffer:
    def __init__(self, size):
        self.size = int(size)
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.write_ptr = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        data = data.ravel().astype(np.float32)
        n = data.shape[0]
        if n == 0:
            return
        with self.lock:
            end = self.write_ptr + n
            if end <= self.size:
                self.buf[self.write_ptr:end] = data
            else:
                first = self.size - self.write_ptr
                self.buf[self.write_ptr:] = data[:first]
                self.buf[: end % self.size] = data[first:]
            self.write_ptr = end % self.size
            self.total_written += n

    def get_last(self, length):
        length = int(length)
        if length <= 0:
            return np.zeros(0, dtype=np.float32)
        with self.lock:
            if self.total_written < self.size:
                available = min(self.write_ptr, length)
                start = self.write_ptr - available
                return self.buf[start : self.write_ptr].copy()
            if length >= self.size:
                return np.concatenate((self.buf[self.write_ptr :], self.buf[: self.write_ptr])).copy()
            start = (self.write_ptr - length) % self.size
            if start < self.write_ptr:
                return self.buf[start : self.write_ptr].copy()
            else:
                return np.concatenate((self.buf[start:], self.buf[: self.write_ptr])).copy()


def get_device_for_input(source: str, device_name=None):
    """
    Returns an input device depending on the source:
      - 'mic': sounddevice input (regular mic)
      - 'loopback': soundcard loopback device (system output capture)
    """
    if source == "mic":
        return device_name

    if source == "loopback":
        mics = sc.all_microphones(include_loopback=True)
        loopbacks = [m for m in mics if m.isloopback]
        if not loopbacks:
            raise RuntimeError(
                "Nenhum dispositivo de loopback encontrado. "
                "Ative 'Mixagem estéreo' ou use um dispositivo compatível."
            )
        print(f"Usando loopback: {loopbacks[0].name}")
        return loopbacks[0]

    raise ValueError(f"Unknown input source: {source}")


# ---------------- Real-time plotting ----------------
def run_realtime_cepstral(
    device=None,
    samplerate=22050,
    display_seconds=1.0,
    window_ms=25.0,
    hop_ms=10.0,
    N_fft=1024,
    N_filters=20,
    N_mels=16,
    bank_min_f=0,
    bank_max_f=None,
    input_source="mic",
    vmin=-100,
    vmax=100
):
    if bank_max_f is None:
        bank_max_f = samplerate / 2.0

    window_dt = window_ms / 1000.0
    hop_dt = hop_ms / 1000.0
    hop_size = int(round(hop_dt * samplerate))
    win_size = int(round(window_dt * samplerate))
    display_size = int(round(display_seconds * samplerate))

    print(f"samplerate={samplerate} win={win_size} hop={hop_size} display={display_size}")
    print(f"Input source: {input_source}")

    q = queue.Queue(maxsize=50)
    rb = RingBuffer(max(display_size * 2, win_size * 4))

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Real-time Cepstral (MFCC) [{input_source}]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cepstral coefficient (k)")
    im, cbar = None, None

    # Resolve device (sounddevice or soundcard)
    input_dev = get_device_for_input(input_source, device)

    def process_audio_block(block):
        mono = block.mean(axis=1).astype(np.float32)
        rb.write(mono)

    print("Stream started. Press Ctrl+C to stop.")

    if input_source == "mic":
        # --- Sounddevice mic stream ---
        def audio_callback(indata, frames, time_info, status):
            if status:
                print("[sounddevice status]", status)
            try:
                q.put_nowait(indata.copy())
            except queue.Full:
                q.get_nowait()
                q.put_nowait(indata.copy())

        stream = sd.InputStream(
            device=input_dev,
            channels=1,
            samplerate=samplerate,
            blocksize=hop_size,
            callback=audio_callback,
        )
        stream.start()

    else:
        # --- Soundcard loopback ---
        def soundcard_thread():
            with input_dev.recorder(samplerate=samplerate) as mic:
                while True:
                    data = mic.record(numframes=hop_size)
                    q.put(data)
        threading.Thread(target=soundcard_thread, daemon=True).start()

    # --- Main processing loop ---
    try:
        last_update = 0.0
        plot_interval = hop_dt

        while True:
            try:
                while True:
                    block = q.get_nowait()
                    process_audio_block(block)
            except queue.Empty:
                pass

            sig = rb.get_last(display_size)
            if sig.size < win_size:
                time.sleep(0.005)
                continue

            sig_obj = Signal(sig, samplerate, len(sig) / samplerate)
            mfcc = get_mfcc(sig_obj, window_dt, hop_dt, N_fft, N_filters, N_mels, bank_min_f, bank_max_f)
            mfcc_plot = mfcc.T
            n_frames = mfcc.shape[0]
            extent = [0.0, n_frames * hop_dt, 0, N_mels]

            now = time.perf_counter()
            if now - last_update >= plot_interval * 0.9:
                if im is None:
                    im = ax.imshow(
						mfcc_plot,
						aspect="auto",
						origin="lower",
						cmap="viridis",
						extent=extent,
						vmin=vmin,
						vmax=vmax,
					)
                    cbar = fig.colorbar(im, ax=ax)
                    ax.set_ylim(0, N_mels)
                else:
                    im.set_data(mfcc_plot)
                    im.set_extent(extent)
                fig.canvas.flush_events()
                plt.pause(0.001)
                last_update = now

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if input_source == "mic":
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        plt.ioff()
        plt.close(fig)
        print("Stream closed. Goodbye.")

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Real-time cepstral (MFCC) viewer")
    p.add_argument("--device", default=None, help="Input device (sounddevice). Default: system default")
    p.add_argument("--samplerate", type=int, default=48000, help="Sampling rate (Hz)")
    p.add_argument("--display-seconds", type=float, default=1.0, help="How many seconds of audio to show")
    p.add_argument("--window-ms", type=float, default=25.0, help="Window length (ms)")
    p.add_argument("--hop-ms", type=float, default=10.0, help="Hop length (ms)")
    p.add_argument("--vmin", type=float, default=-25.0, help="Colormap max value")
    p.add_argument("--vmax", type=float, default=25.0, help="Colormap min value")
    p.add_argument("--n-fft", type=int, default=1024, help="FFT size")
    p.add_argument("--n-filters", type=int, default=20, help="Number of mel filters")
    p.add_argument("--n-mels", type=int, default=16, help="Number of cepstral coefficients to display")
    p.add_argument("--bank-max-f", type=float, default=None, help="Max freq for mel bank (Hz). Default: samplerate/2.")
    p.add_argument("--input-source", choices=["mic", "loopback"], default="mic", help="Audio source: mic or loopback")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_realtime_cepstral(
        device=args.device,
        samplerate=args.samplerate,
        display_seconds=args.display_seconds,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        N_fft=args.n_fft,
        N_filters=args.n_filters,
        N_mels=args.n_mels,
        bank_max_f=args.bank_max_f,
        input_source=args.input_source,
        vmin=args.vmin,
        vmax=args.vmax,
    )
