from .nodes.indextts2_node import ZYK_IndexTTS2Simple
from .nodes.indextts2_node_advanced import ZYK_IndexTTS2Advanced
from .nodes.indextts2_node_emovec import ZYK_IndexTTS2EmotionVector
from .nodes.indextts2_node_emotext import ZYK_IndexTTS2EmotionFromText
from .nodes.indextts2_save_audio import ZYK_IndexTTS2SaveAudio
from .nodes.indextts2_volume_adjust import ZYK_IndexTTS2VolumeAdjust
from .nodes.indextts2_denoise import ZYK_IndexTTS2Denoise

NODE_CLASS_MAPPINGS = {
    "zyk-IndexTTS2Advanced": ZYK_IndexTTS2Advanced,
    "zyk-IndexTTS2EmotionFromText": ZYK_IndexTTS2EmotionFromText,
    "zyk-IndexTTS2EmotionVector": ZYK_IndexTTS2EmotionVector,
    "zyk-IndexTTS2SaveAudio": ZYK_IndexTTS2SaveAudio,
    "zyk-IndexTTS2Simple": ZYK_IndexTTS2Simple,
    "zyk-IndexTTS2VolumeAdjust": ZYK_IndexTTS2VolumeAdjust,
    "zyk-IndexTTS2Denoise": ZYK_IndexTTS2Denoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zyk-IndexTTS2Advanced": "zyk-IndexTTS2 Advanced",
    "zyk-IndexTTS2EmotionFromText": "zyk-IndexTTS2 Emotion From Text",
    "zyk-IndexTTS2EmotionVector": "zyk-IndexTTS2 Emotion Vector",
    "zyk-IndexTTS2SaveAudio": "zyk-IndexTTS2 Save Audio",
    "zyk-IndexTTS2Simple": "zyk-IndexTTS2 Simple",
    "zyk-IndexTTS2VolumeAdjust": "zyk-IndexTTS2 Volume Adjust",
    "zyk-IndexTTS2Denoise": "zyk-IndexTTS2 Denoise",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

