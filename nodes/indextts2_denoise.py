import os
import numpy as np
import torch

# ─── Backend detection ────────────────────────────────────────────────────────
# We try deepfilternet first (better quality). If not installed, fall back to
# noisereduce (pure Python, no compilation needed).

_DENOISE_USE_DF = False
_DENOISE_USE_NR = False

try:
    import noisereduce as nr
    _DENOISE_USE_NR = True
except ImportError:
    pass

try:
    from df.enhance import init_df, enhance
    import df.config as df_config
    from df.model import ModelParams
    _DENOISE_USE_DF = True
except ImportError:
    pass

if not _DENOISE_USE_DF and not _DENOISE_USE_NR:
    print(
        "[IndexTTS2 Denoise] Neither deepfilternet nor noisereduce is installed.\n"
        "  Install noisereduce (recommended for easy setup): pip install noisereduce\n"
        "  Or install deepfilternet (better quality, needs Rust): pip install deepfilternet"
    )


# ─── DeepFilterNet backend (kept for reference) ──────────────────────────────
#
# If a cp313 wheel for deepfilterlib becomes available in the future, you can
# re-enable this backend. The original code has been preserved below as a
# reference for the implementation pattern.
#
# Original download helpers (ModelScope / GitHub / caching) have been removed
# from the active code but are kept as comments for future restoration.
#
# To restore deepfilternet support:
#   1. Uncomment the _download_model_* and _get_or_download_model functions
#   2. Restore _DENOISE_SR = 48000
#   3. Restore _DENOISE_CACHE dict
#   4. Restore _load_model() method using init_df()
#   5. Restore the deepfilternet processing path in process()
#
"""
_DENOISE_SR = 48000  # DeepFilterNet operates at 48 kHz
_DENOISE_CACHE: dict = {}

_MODELSCOPE_MAP = {
    "DeepFilterNet": "pengzhendong/DeepFilterNet",
    "DeepFilterNet3": "fal/DeepFilterNet3",
}

def _get_ext_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _get_denoise_models_dir():
    return os.path.join(_get_ext_root(), "checkpoints", "audio_denoise")

def _ensure_model_dir(model_name: str) -> str:
    model_dir = os.path.join(_get_denoise_models_dir(), model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def _is_model_valid(model_dir: str) -> bool:
    config_path = os.path.join(model_dir, "config.ini")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    if os.path.isfile(config_path) and os.path.isdir(checkpoint_dir):
        return True
    return os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0

def _download_model_from_ms(model_name: str, target_dir: str) -> bool:
    ms_id = _MODELSCOPE_MAP.get(model_name)
    if not ms_id:
        return False
    try:
        from modelscope import snapshot_download
        print(f"[IndexTTS2 Denoise] Downloading {model_name} from ModelScope ({ms_id})...")
        downloaded = snapshot_download(ms_id)
        if os.path.isdir(downloaded):
            for item in os.listdir(downloaded):
                src = os.path.join(downloaded, item)
                dst = os.path.join(target_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            print(f"[IndexTTS2 Denoise] ModelScope download complete: {model_name}")
            return True
    except Exception as e:
        print(f"[IndexTTS2 Denoise] ModelScope download failed: {e}")
    return False

def _download_model_from_github(model_name: str, target_dir: str) -> bool:
    try:
        from df.enhance import maybe_download_model
    except ImportError:
        return False
    try:
        downloaded_dir = maybe_download_model(model_name)
        if os.path.isdir(downloaded_dir) and downloaded_dir != target_dir:
            for item in os.listdir(downloaded_dir):
                src = os.path.join(downloaded_dir, item)
                dst = os.path.join(target_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            return True
    except Exception:
        pass
    return False

def _get_or_download_model(model_name: str) -> str:
    model_dir = _ensure_model_dir(model_name)
    if _is_model_valid(model_dir):
        return model_dir
    if _download_model_from_ms(model_name, model_dir):
        return model_dir
    if _download_model_from_github(model_name, model_dir):
        return model_dir
    return model_dir
"""


# ─── Audio parsing ────────────────────────────────────────────────────────────

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
        wav = wav[None, :]
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


# ─── Global model cache for denoise ───────────────────────────────────────────

_DENOISE_CACHE: dict = {}
_DENOISE_SR = 48000  # DeepFilterNet operates at 48 kHz; noisereduce works at any SR


class ZYK_IndexTTS2Denoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable to bypass denoising (audio passes through unchanged).",
                }),
                "model": (["noisereduce", "DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3"], {
                    "default": "noisereduce",
                    "tooltip": "Denoising engine. DeepFilterNet models require deepfilternet package. "
                               "noisereduce works out of the box (pip install noisereduce).",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Noise reduction strength. 0 = no reduction, 1.0 = normal, 2.0 = aggressive.",
                }),
                "post_filter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable post-filter to further suppress residual noise. "
                               "(Only used by DeepFilterNet backend.)",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "zyk-Audio/IndexTTS"

    def __init__(self):
        self._model_instance = None
        self._df_state = None
        self._current_model_name = None

    def _load_model(self, model_name: str):
        """Load (or retrieve from cache) a DeepFilterNet model (or noisereduce)."""
        if model_name == "noisereduce":
            if not _DENOISE_USE_NR:
                raise RuntimeError(
                    "noisereduce is not installed.\n"
                    "  Install with: pip install noisereduce"
                )
            self._current_model_name = "noisereduce"
            return

        cache_key = f"denoise_{model_name}"

        if cache_key in _DENOISE_CACHE:
            self._model_instance, self._df_state = _DENOISE_CACHE[cache_key]
            self._current_model_name = model_name
            return

        if not _DENOISE_USE_DF:
            raise RuntimeError(
                f"Model '{model_name}' requires deepfilternet, but it is not installed.\n"
                f"  Either install deepfilternet (needs Rust): pip install deepfilternet\n"
                f"  Or use the 'noisereduce' model instead.\n"
                f"  On Windows, Rust is needed: https://rustup.rs"
            )

        # Get model directory (downloads if needed)
        model_dir = _get_or_download_model(model_name)

        try:
            import logging
            logging.getLogger("df").setLevel(logging.WARNING)

            print(f"[IndexTTS2 Denoise] Loading model: {model_name} from {model_dir}")
            model, df_state, _, _ = init_df(
                model_base_dir=model_dir,
                post_filter=False,
                log_level="ERROR",
                config_allow_defaults=True,
            )
            model.eval()
            self._model_instance = model
            self._df_state = df_state
            self._current_model_name = model_name

            _DENOISE_CACHE[cache_key] = (model, df_state)
            print(f"[IndexTTS2 Denoise] Model loaded: {model_name}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load denoise model '{model_name}'. "
                f"Make sure 'deepfilternet' is installed (pip install deepfilternet). "
                f"On Windows, the Rust toolchain is also needed: https://rustup.rs\n"
                f"Error: {e}"
            )

    def _denoise_noisereduce(self, wav: np.ndarray, sr: int, strength: float) -> np.ndarray:
        """Apply noise reduction using noisereduce.
        wav: (channels, samples) float32 in [-1, 1]
        Returns denoised array of same shape.
        """
        # noisereduce works per-channel
        result = []
        for ch in range(wav.shape[0]):
            # Map strength to noisereduce parameters
            # prop_decrease: 0.0 = no reduction, 1.0 = full reduction
            prop_decrease = min(1.0, strength)
            # Stationary noise reduction for simplicity (no need for noise clip)
            # We estimate noise from the first 0.5 second
            noise_samples = int(min(sr * 0.5, wav.shape[1] * 0.1))
            if noise_samples < sr * 0.1:
                noise_samples = max(1, wav.shape[1] // 10)
            ch_denoised = nr.reduce_noise(
                y=wav[ch],
                sr=sr,
                prop_decrease=prop_decrease,
                n_std_thresh_stationary=1.5 - float(strength) * 0.5,
                stationary=True,
            )
            result.append(ch_denoised)
        return np.stack(result, axis=0)

    def _denoise_deepfilternet(self, wav: np.ndarray, sr: int,
                               model, df_state, strength: float,
                               post_filter: bool) -> torch.Tensor:
        """Apply noise reduction using deepfilternet.
        Returns torch tensor (channels, samples).
        """
        orig_sr = sr
        audio_t = torch.from_numpy(wav).float()

        # Resample to 48kHz if needed
        if orig_sr != _DENOISE_SR:
            audio_t = torchaudio_resample(audio_t, orig_sr, _DENOISE_SR)

        # Ensure at least 2 channels (DeepFilterNet works best with stereo)
        if audio_t.ndim == 1:
            audio_t = audio_t.unsqueeze(0)
        is_mono = audio_t.shape[0] == 1
        if is_mono:
            audio_t = audio_t.repeat(2, 1)

        # Map strength to attenuation limit (dB)
        if strength <= 0.0:
            return None  # caller handles bypass
        elif strength < 1.0:
            atten_lim = 24.0 * (1.0 - float(strength))
        else:
            atten_lim = None

        try:
            df_config.set("mask_pf", bool(post_filter), bool, ModelParams().section)
        except Exception:
            pass

        with torch.no_grad():
            enhanced = enhance(
                model, df_state, audio_t,
                pad=True, atten_lim_db=atten_lim,
            )

        # For strength > 1.0, apply extra attenuation
        if strength > 1.0:
            extra = min(1.0, float(strength) - 1.0)
            factor = 1.0 - extra * 0.75
            enhanced = factor * enhanced + (1.0 - factor) * torch.clamp(
                enhanced - audio_t * (1.0 - factor) * 0.3, -1.0, 1.0
            )

        enhanced = torch.clamp(enhanced.cpu(), -1.0, 1.0)

        if is_mono:
            enhanced = enhanced[:1, :]

        if orig_sr != _DENOISE_SR:
            enhanced = torchaudio_resample(enhanced, _DENOISE_SR, orig_sr)

        return enhanced

    def process(self, audio, enabled, model, strength, post_filter):
        """Denoise audio."""
        if not enabled:
            return (audio,)

        sr, wav = _parse_audio_input(audio)

        # Load model if needed
        if self._current_model_name != model:
            self._load_model(model)

        denoised = None

        if model == "noisereduce":
            denoised_np = self._denoise_noisereduce(wav, sr, strength)
            denoised = torch.from_numpy(denoised_np.astype(np.float32))
        else:
            denoised = self._denoise_deepfilternet(
                wav, sr, self._model_instance, self._df_state,
                strength, post_filter
            )

        if denoised is None:
            return (audio,)

        # Return in standard ComfyUI AUDIO dict format
        return ({"waveform": denoised.unsqueeze(0), "sample_rate": sr},)


def torchaudio_resample(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample audio tensor using torchaudio."""
    try:
        import torchaudio
        # torchaudio's resample expects (..., samples)
        return torchaudio.functional.resample(audio, orig_sr, target_sr)
    except Exception:
        try:
            import librosa
            # librosa resample expects (samples,) or (channels, samples) as numpy
            audio_np = audio.numpy()
            if audio_np.ndim == 1:
                resampled = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=target_sr)
                return torch.from_numpy(resampled)
            else:
                resampled = np.stack([
                    librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr)
                    for ch in audio_np
                ])
                return torch.from_numpy(resampled)
        except Exception:
            raise RuntimeError(
                "Resampling requires either torchaudio or librosa. "
                f"Cannot resample from {orig_sr}Hz to {target_sr}Hz."
            )
