#!/usr/bin/env python3
import numpy as np
import scipy.signal

def framing(
        x,
        frame_length: int = 512,
        frame_shift: int = 256,
        centered: bool = True,
        padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result
def detect_non_silence(
        x: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 1024,
        frame_shift: int = 512,
        window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.
    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    # power: (C, T)
    power = (framed_w ** 2).mean(axis=-1)
    # mean_power: (C, 1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=np.bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )

def snr_estimate(input_signal, threshold=0.01):
    EPS = 0.00000001
    # speech detection
    speech_idx = detect_non_silence(input_signal, threshold)
    noise_idx = ~speech_idx
    speech_sig = input_signal[speech_idx]
    noise_sig = input_signal[noise_idx]
    # Calc power on non silence region
    if speech_sig.size==0:
        power_s = EPS
    else:
        power_s = (speech_sig ** 2).mean()
    if noise_sig.size==0:  # the noise has not been detected, usually the signal is very short
        power_n = power_s   # we assume the signal is noisy with short speech segment.
    else:
        power_n = (noise_sig ** 2).mean()

    snr = 10 * np.log10(power_s/power_n)
    return snr

if __name__ == "__main__":
    import sys
    import librosa
    #wav_path = sys.stdin
    #snr = sys.stdout
    #list = ["asr_testset_0541 /data/megastore/Projects/DuJing/data/ASR_test/test_xmov/wav/asr_testset_0541.wav",]
    for path in sys.stdin:
        path = path.strip().split()
        name = path[0]
        path = path[1]
        wav, sr = librosa.load(path, sr=16000)
        if sr != 16000:
            wav = librosa.resample(wav, sr, 16000)
        if wav.ndim > 1:
            wav = wav.mean(-1)
        assert wav.ndim == 1
        snr = snr_estimate(wav, 0.2)
        sys.stdout.write(f"{name} {snr}\n")