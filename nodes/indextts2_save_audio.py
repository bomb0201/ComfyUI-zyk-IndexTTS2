import os
from typing import List
import numpy as np
import folder_paths

class IndexTTS2SaveAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "name": ("STRING", {"default": "tts2", "placeholder": "file name prefix"}),
                "format": ("COMBO", {"options": ["wav", "mp3"], "default": "wav"}),
            },
            "optional": {
                "normalize_peak": ("BOOLEAN", {"default": False, "tooltip": "Normalize peak to ~0.98 before saving."}),
                # WAV
                "wav_pcm": ("COMBO", {"options": ["pcm16", "pcm24", "f32"], "default": "pcm16"}),
                # MP3
                "mp3_bitrate": ("COMBO", {"options": ["128k", "192k", "256k", "320k"], "default": "320k"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "saved_path")
    FUNCTION = "save"
    CATEGORY = "Audio/IndexTTS"

    def _normalize(self, mono: np.ndarray):
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak > 1e-6:
            mono = np.clip(mono * (0.98 / peak), -1.0, 1.0)
        return mono

    def _save_wav(self, path: str, data: np.ndarray, sr: int, pcm: str):
        try:
            import soundfile as sf  # type: ignore
            subtype = {
                "pcm16": "PCM_16",
                "pcm24": "PCM_24",
                "f32": "FLOAT",
            }.get(pcm, "PCM_16")
            sf.write(path, data.T, sr, subtype=subtype, format="WAV")
            return True
        except Exception:
            # Fallback to wave for PCM16 only
            if pcm != "pcm16":
                raise
            import wave, contextlib
            pcm16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
            with contextlib.closing(wave.open(path, "wb")) as wf:
                wf.setnchannels(int(data.shape[0]))
                wf.setsampwidth(2)
                wf.setframerate(int(sr))
                wf.writeframes(pcm16.T.tobytes())
            return True

    def _compose_paths(self, name_prefix: str, batch_count: int) -> List[str]:
        output_dir = folder_paths.get_output_directory()
        # Use Comfy's helper to build prefix and a counter
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            f"audio/{name_prefix}", output_dir
        )
        paths = []
        for b in range(batch_count):
            filename_with_batch = filename.replace("%batch_num%", str(b))
            file = f"{filename_with_batch}_{counter:05}_"
            paths.append(os.path.join(full_output_folder, file))
            counter += 1
        return paths

    def _save_with_av(self, fmt: str, audio, filename_prefix: str, quality: str = "320k") -> List[str]:
        try:
            from comfy_extras import nodes_audio as ce_audio  # type: ignore
        except Exception as e:
            raise RuntimeError(f"PyAV save requires comfy_extras.nodes_audio: {e}")

        if fmt == "mp3":
            saver = ce_audio.SaveAudioMP3()
            ui = saver.save_mp3(audio, filename_prefix=filename_prefix, format="mp3", quality=quality)
        else:
            raise ValueError(f"Unsupported format for AV saver (mp3 only): {fmt}")

        results = ui.get("ui", {}).get("audio", [])
        base = folder_paths.get_output_directory()
        out: List[str] = []
        for item in results:
            sub = item.get("subfolder") or ""
            out.append(os.path.join(base, sub, item.get("filename", "")))
        return out
    
    def save(self, audio, name: str, format: str,
             normalize_peak: bool = False,
             wav_pcm: str = "pcm16",
             mp3_bitrate: str = "320k"):
        # Extract waveform
        import torch
        wav = audio["waveform"]
        sr = int(audio["sample_rate"]) if isinstance(audio.get("sample_rate"), (int, float)) else 22050
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav)
        # Shape: (B, C, N)
        if wav.ndim != 3:
            raise ValueError("AUDIO input must be shaped (B, C, N)")

        # Prepare per-batch data as float32 in [-1,1]
        batch = []
        for b in range(wav.shape[0]):
            np_w = wav[b]
            if np_w.dtype == np.int16:
                np_w = np_w.astype(np.float32) / 32767.0
            elif np_w.dtype != np.float32:
                np_w = np_w.astype(np.float32)
            # Keep original channels; expect 1 or 2 generally
            if normalize_peak:
                if np_w.shape[0] == 1:
                    np_w[0] = self._normalize(np_w[0])
                else:
                    # Normalize jointly to keep relative balance
                    peak = float(np.max(np.abs(np_w))) if np_w.size else 0.0
                    if peak > 1e-6:
                        np_w = np.clip(np_w * (0.98 / peak), -1.0, 1.0)
            batch.append(np_w)

        name_prefix = (name or "tts2").strip() or "tts2"
        paths: List[str] = []

        if format == "wav":
            base_paths = self._compose_paths(name_prefix, len(batch))
            for np_w, base in zip(batch, base_paths):
                out_path = base + ".wav"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                self._save_wav(out_path, np_w, sr, wav_pcm)
                paths.append(out_path)
        elif format == "mp3":
            paths = self._save_with_av("mp3", audio, filename_prefix=f"audio/{name_prefix}", quality=mp3_bitrate)
        else:
            raise ValueError(f"Unsupported format: {format}")

        saved = "\n".join(paths)
        # passthrough audio so the graph can continue if needed
        return (audio, saved)





