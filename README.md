# CosyVoice-ComfyUI
a comfyui custom node for [CosyVoice](https://github.com/FunAudioLLM/CosyVoice),you can find workflow in [workflows](./workflows/)

## Example
test on 2080ti 11GB torch==2.3.0+cu121 python 3.10.8

base tts
- tts_text `你好，我是通义生成式语音大模型，请问有什么可以帮您的吗`
- ouput audio

3s clone tts
- tts_text `收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。`
- prompt_text `希望你以后能够做的比我还好呦。`
- prompt_wav `zero_shot_prompt.wav`

- output audio

cross lingual
- tts_text `<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\\'s coming into the family is a reason why sometimes we don\\'t buy the whole thing.`
- prompt_wav  `cross_lingual_prompt.wav`
- output audio

instruct 
- tts_text `在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。`
- instruct_text `Theo \\'Crimson\\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.`
- output audio

## How to use
```
## in ComfyUI/custom_nodes
git clone https://github.com/AIFSH/CosyVoice-ComfyUI.git
cd CosyVoice-ComfyUI
pip install -r requirements.txt
```
weights will be downloaded from modelscope

## Tutorial
- todo
- QQ群：852228202

## Thanks

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice_For_Windows](https://github.com/v3ucn/CosyVoice_For_Windows)
