import numpy as np
import torch


def _parse_audio_input(audio):
    """Parse ComfyUI AUDIO input into (sample_rate, numpy_array) tuple.
    Returns (sr: int, wav: np.ndarray) where wav is float32 in [-1, 1]
    and shape is (channels, samples).
    """
    sr = None
    data = None

    if isinstance(audio, (tuple, list)):
        for item in audio:
            if isinstance(item, (int, np.integer)):
                sr = int(item)
            elif hasattr(item, "shape"):
                data = item
        if sr is None and len(audio) >= 2:
            a, b = audio[:2]
            if isinstance(a, (int, np.integer)) and hasattr(b, "shape"):
                sr, data = int(a), b
            elif isinstance(b, (int, np.integer)) and hasattr(a, "shape"):
                sr, data = int(b), a
    elif isinstance(audio, dict):
        sr = audio.get("sample_rate")
        if sr is None:
            sr = audio.get("sr")
        data = audio.get("waveform")
        if data is None:
            data = audio.get("samples")
        if data is None:
            data = audio.get("data")

    if sr is None or data is None:
        raise ValueError(f"Invalid AUDIO input. Expected (sample_rate, numpy_array) tuple, got {type(audio).__name__}")

    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    wav = np.asarray(data)

    if wav.ndim == 1:
        wav = wav[None, :]  # (1, N)
    elif wav.ndim == 2:
        ch_dim = 0 if wav.shape[0] <= 8 and wav.shape[0] <= wav.shape[1] else 1 if wav.shape[1] <= 8 else 0
        if ch_dim == 1:
            wav = np.transpose(wav, (1, 0))
    elif wav.ndim >= 3:
        sizes = list(wav.shape)
        sample_axis = int(np.argmax(sizes))
        axes = [i for i in range(wav.ndim) if i != sample_axis] + [sample_axis]
        wav = np.transpose(wav, axes)
        c = int(np.prod(wav.shape[:-1]))
        wav = np.reshape(wav, (c, wav.shape[-1]))

    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        denom = float(max(abs(info.min), abs(info.max))) or 32767.0
        wav = wav.astype(np.float32) / denom
    else:
        wav = np.clip(wav.astype(np.float32), -1.0, 1.0)

    return int(sr), wav


class ZYK_IndexTTS2VolumeAdjust:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enabled": ("BOOLEAN", {"default": True,
                                        "tooltip": "Disable to bypass volume adjustment (audio passes through unchanged)."}),
                "gain_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -20.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Volume adjustment in decibels (-20 dB = 0.1x, +20 dB = 10x)."
                }),
                "peak_limit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable peak limiting to prevent clipping distortion when boosting volume."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "zyk-Audio/IndexTTS"

    def process(self, audio, enabled, gain_db, peak_limit):
        """Adjust audio volume."""
        if not enabled:
            return (audio,)

        sr, wav = _parse_audio_input(audio)

        # Apply gain: dB -> linear
        gain_linear = 10.0 ** (float(gain_db) / 20.0)
        wav = wav * gain_linear

        # Peak limiting to prevent clipping
        if peak_limit:
            peak = float(np.max(np.abs(wav)))
            if peak > 0.98:
                # Scale down so peak hits 0.98 (slightly below 0 dBFS)
                wav = wav * (0.98 / peak)
            elif peak > 1.0:
                # Clip hard at 1.0 (shouldn't happen if peak_limit is on, but safety)
                wav = np.clip(wav, -1.0, 1.0)
        else:
            # Hard clip at 1.0 to prevent extreme values downstream
            wav = np.clip(wav, -1.0, 1.0)

        # Return in standard ComfyUI AUDIO dict format
        waveform = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)  # (1, C, T)
        return ({"waveform": waveform, "sample_rate": sr},)
