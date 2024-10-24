from .nodes import TextNode,CosyVoiceNode,CosyVoiceDubbingNode,LoadSRT,PreviewAudio,LoadAudio

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "TextNode": TextNode,
    "CosyVoiceNode": CosyVoiceNode,
    "CosyVoiceDubbingNode":CosyVoiceDubbingNode,
    "LoadSRT":LoadSRT,
    "LoadAudio": LoadAudio,
    "PreviewAudio": PreviewAudio,
}
