import os
import shutil
import numpy as np
import torch

# ─── Model download helpers ───────────────────────────────────────────────────

# Mapping from UI model name → ModelScope repo ID
_MODELSCOPE_MAP = {
    "DeepFilterNet": "pengzhendong/DeepFilterNet",
    # DeepFilterNet2 not available on ModelScope as of 2026-06
    "DeepFilterNet3": "fal/DeepFilterNet3",
}

# The deepfilternet package downloads from GitHub by default.
# We use that as primary, and ModelScope as a fallback.


def _get_ext_root():
    """Get the extension root directory (ComfyUI-IndexTTS2)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_denoise_models_dir():
    """Get the directory where denoise models are stored."""
    return os.path.join(_get_ext_root(), "checkpoints", "audio_denoise")


def _ensure_model_dir(model_name: str) -> str:
    """Ensure the local model directory exists.
    Returns the path to the model directory.
    """
    model_dir = os.path.join(_get_denoise_models_dir(), model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def _is_model_valid(model_dir: str) -> bool:
    """Check if a model directory has the required files."""
    config_path = os.path.join(model_dir, "config.ini")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    if os.path.isfile(config_path) and os.path.isdir(checkpoint_dir):
        return True
    # Also accept if there are any files in the dir (partial download recovery)
    return os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0


def _download_model_from_ms(model_name: str, target_dir: str) -> bool:
    """Try to download model from ModelScope.
    Returns True if successful.
    """
    ms_id = _MODELSCOPE_MAP.get(model_name)
    if not ms_id:
        return False
    try:
        from modelscope import snapshot_download
        print(f"[IndexTTS2 Denoise] Downloading {model_name} from ModelScope ({ms_id})...")
        # snapshot_download downloads to a cache dir; we copy to our target dir
        downloaded = snapshot_download(ms_id)
        if os.path.isdir(downloaded):
            # Copy files to our target directory
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
    """Use deepfilternet's built-in download mechanism (from GitHub).
    Returns True if successful.
    """
    try:
        from df.enhance import maybe_download_model
    except ImportError:
        print("[IndexTTS2 Denoise] deepfilternet package not installed. Skipping GitHub download.")
        return False

    try:
        print(f"[IndexTTS2 Denoise] Downloading {model_name} from GitHub (via deepfilternet)...")
        # This downloads via deepfilternet's built-in mechanism
        downloaded_dir = maybe_download_model(model_name)
        if os.path.isdir(downloaded_dir) and downloaded_dir != target_dir:
            # Copy from cache to our local directory
            for item in os.listdir(downloaded_dir):
                src = os.path.join(downloaded_dir, item)
                dst = os.path.join(target_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            print(f"[IndexTTS2 Denoise] GitHub download complete: {model_name}")
            return True
    except Exception as e:
        print(f"[IndexTTS2 Denoise] GitHub download failed: {e}")
    return False


def _get_or_download_model(model_name: str) -> str:
    """Get the model directory, downloading if necessary.
    Returns the path to the model directory.
    """
    model_dir = _ensure_model_dir(model_name)

    # 1. Check if already downloaded locally
    if _is_model_valid(model_dir):
        print(f"[IndexTTS2 Denoise] Using cached model: {model_dir}")
        return model_dir

    # 2. Try ModelScope first (fastest for China users)
    if _download_model_from_ms(model_name, model_dir):
        return model_dir

    # 3. Fallback to GitHub via deepfilternet
    if _download_model_from_github(model_name, model_dir):
        return model_dir

    # 4. If all downloads failed but dir was created, return it anyway
    #    (init_df will give a clear error)
    return model_dir


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
        sr = audio.get("sample_rate") or audio.get("sr")
        data = audio.get("waveform") or audio.get("samples") or audio.get("data")

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
_DENOISE_SR = 48000  # DeepFilterNet operates at 48 kHz


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
                "model": (["DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3"], {
                    "default": "DeepFilterNet2",
                    "tooltip": "Denoising model version. DeepFilterNet2 is the recommended default.",
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
                    "tooltip": "Enable post-filter to further suppress residual noise.",
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
        """Load (or retrieve from cache) a DeepFilterNet model."""
        cache_key = f"denoise_{model_name}"

        if cache_key in _DENOISE_CACHE:
            self._model_instance, self._df_state = _DENOISE_CACHE[cache_key]
            self._current_model_name = model_name
            return

        # Get model directory (downloads if needed)
        model_dir = _get_or_download_model(model_name)

        try:
            from df.enhance import init_df
            import logging

            # Suppress deepfilternet logging
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

            # Cache it
            _DENOISE_CACHE[cache_key] = (model, df_state)
            print(f"[IndexTTS2 Denoise] Model loaded: {model_name}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load denoise model '{model_name}'. "
                f"Make sure 'deepfilternet' is installed (pip install deepfilternet). "
                f"Error: {e}"
            )

    def process(self, audio, enabled, model, strength, post_filter):
        """Denoise audio."""
        if not enabled:
            return (audio,)

        sr, wav = _parse_audio_input(audio)
        orig_sr = sr

        # Load model if needed
        if self._current_model_name != model:
            self._load_model(model)

        from df.enhance import enhance

        # Convert to torch tensor [channels, samples]
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

        # Map strength to attenuation limit (dB).
        # DeepFilterNet's atten_lim_db: 0 = no denoising (original pass-through),
        # None = full model denoising, positive = max suppression in dB.
        if strength <= 0.0:
            return (audio,)
        elif strength < 1.0:
            atten_lim = 24.0 * (1.0 - float(strength))
        else:
            atten_lim = None

        # Apply post-filter via DeepFilterNet's global config.
        # Set before each enhance() call to handle toggling between calls.
        try:
            from df.config import config as df_config
            from df.model import ModelParams
            df_config.set("mask_pf", bool(post_filter), bool, ModelParams().section)
        except Exception:
            pass

        # Apply denoising
        with torch.no_grad():
            enhanced = enhance(
                self._model_instance, self._df_state, audio_t,
                pad=True, atten_lim_db=atten_lim,
            )

        # For strength > 1.0, apply extra attenuation by blending with original.
        # This is a soft-gate approach: mix the denoised output with a further
        # attenuated version to increase suppression depth.
        if strength > 1.0:
            # extra ranges 0.0→1.0 as strength goes 1.0→2.0
            extra = min(1.0, float(strength) - 1.0)
            # scale factor: at extra=0 → factor=1, at extra=1 → factor=0.25
            factor = 1.0 - extra * 0.75
            # Keep original as reference for residual noise suppression
            enhanced = factor * enhanced + (1.0 - factor) * torch.clamp(
                enhanced - audio_t * (1.0 - factor) * 0.3, -1.0, 1.0
            )

        # Clamp to safe range
        enhanced = torch.clamp(enhanced.cpu(), -1.0, 1.0)

        # Down-mix back to mono if input was mono
        if is_mono:
            enhanced = enhanced[:1, :]

        # Resample back to original sample rate
        if orig_sr != _DENOISE_SR:
            enhanced = torchaudio_resample(enhanced, _DENOISE_SR, orig_sr)

        return ((orig_sr, enhanced.numpy().astype(np.float32)),)


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
