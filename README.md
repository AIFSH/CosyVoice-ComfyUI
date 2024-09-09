# CosyVoice-ComfyUI
a comfyui custom node for [CosyVoice](https://github.com/FunAudioLLM/CosyVoice),you can find workflow in [workflows](./workflows/)

## new Feature
suport `srt` file to single voice or mutiple voice clone

input
- [tts_srt](./workflows/dubbing/zh_test.srt)
- [prompt_wav](./workflows/dubbing/test.mp3)
- [prompt_srt](./workflows/dubbing/en_test.srt)(optional)

output


## Example
test on 2080ti 11GB torch==2.3.0+cu121 python 3.10.8
use case | tts_text | prompt_text | prompt_wav | instruct_text | output 
----- | ---- | ---- | ---- | ---- | ----
`base tts` | `你好，我是通义生成式语音大模型，请问有什么可以帮您的吗` | | | | <video src="https://github.com/AIFSH/CosyVoice-ComfyUI/assets/149982694/03dcca8b-3ea5-41ac-9cbb-279dffaf6122" />
`3s clone tts` | `收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。` |  `希望你以后能够做的比我还好呦。` | <video src="https://github.com/AIFSH/CosyVoice-ComfyUI/assets/149982694/1055a5e4-676c-486e-abdb-d2b76c08878b" /> |  | <video src="https://github.com/AIFSH/CosyVoice-ComfyUI/assets/149982694/fd2c42df-35c9-4ad8-934d-7b0e9e29e7cf"/>
`cross lingual` | "And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\\'s coming into the family is a reason why sometimes we don\\'t buy the whole thing." | | <video src="https://github.com/AIFSH/CosyVoice-ComfyUI/assets/149982694/751b0ed0-3c49-463e-ab9f-6dd7f30fa74c"/> | | <video src="https://github.com/AIFSH/CosyVoice-ComfyUI/assets/149982694/ab375378-ab1c-411a-99bf-6d445d12cb2f"/>
`instruct` | `在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。` | | | `Theo \\'Crimson\\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.` |  <video src="https://github.com/AIFSH/CosyVoice-ComfyUI/assets/149982694/646ff534-fd50-4c63-ad7e-9402c1457993"/>

## How to use
test on py3.10，2080ti 11gb，torch==2.3.0+cu121

make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
## in ComfyUI/custom_nodes
git clone https://github.com/AIFSH/CosyVoice-ComfyUI.git
cd CosyVoice-ComfyUI
pip install -r requirements.txt
```
weights will be downloaded from modelscope

## Tutorial
- [DEMO](https://www.bilibili.com/video/BV16H4y1w7su)
