from .nodes import TextNode, CosyVoiceNode, LoadSRT, CosyVoiceDubbingNode

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "LoadSRT":LoadSRT,
    "TextNode": TextNode,
    "CosyVoiceNode": CosyVoiceNode,
    "CosyVoiceDubbingNode":CosyVoiceDubbingNode
}
