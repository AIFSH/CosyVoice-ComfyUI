import torch
import random
import librosa
import zipfile
import torchaudio
import numpy as np
import os,sys
import folder_paths
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
input_dir = folder_paths.get_input_directory()
output_dir = os.path.join(folder_paths.get_output_directory(),"cosyvoice_dubb")
pretrained_models = os.path.join(now_dir,"pretrained_models")

from modelscope import snapshot_download

import audiotsm
import audiosegment
from srt import parse as SrtPare
from audiosegment import AudioSegment

from cosyvoice.cli.cosyvoice import CosyVoice

sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

max_val = 0.8
prompt_sr, target_sr = 16000, 22050
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

class TextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "encode"

    CATEGORY = "AIFSH_CosyVoice"

    def encode(self,text):
        return (text, )


class CosyVoiceNode:
    def __init__(self):
        self.model_dir = None
        self.cosyvoice = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("TEXT",),
                "inference_mode":(inference_mode_list,{
                    "default": "预训练音色"
                }),
                "sft_dropdown":(sft_spk_list,{
                    "default":"中文女"
                }),
                "seed":("INT",{
                    "default": 42
                })
            },
            "optional":{
                "prompt_text":("TEXT",),
                "prompt_wav": ("AUDIO",),
                "instruct_text":("TEXT",),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_CosyVoice"

    def generate(self,tts_text,inference_mode,sft_dropdown,seed,
                 prompt_text=None,prompt_wav=None,instruct_text=None):
        if inference_mode == '自然语言控制':
            model_dir = os.path.join(pretrained_models,"CosyVoice-300M-Instruct")
            snapshot_download(model_id="iic/CosyVoice-300M-Instruct",local_dir=model_dir)
            assert instruct_text is not None, "in 自然语言控制 mode, instruct_text can't be none"
        if inference_mode in ["跨语种复刻",'3s极速复刻']:
            model_dir = os.path.join(pretrained_models,"CosyVoice-300M")
            snapshot_download(model_id="iic/CosyVoice-300M",local_dir=model_dir)
            assert prompt_wav is not None, "in 跨语种复刻 or 3s极速复刻 mode, prompt_wav can't be none"
            if inference_mode == "3s极速复刻":
                assert len(prompt_text) > 0, "prompt文本为空，您是否忘记输入prompt文本？"
        if inference_mode == "预训练音色":
            model_dir = os.path.join(pretrained_models,"CosyVoice-300M-SFT")
            snapshot_download(model_id="iic/CosyVoice-300M-Instruct",local_dir=model_dir)


        if self.model_dir != model_dir:
            self.model_dir = model_dir
            self.cosyvoice = CosyVoice(model_dir)
        
        if prompt_wav:
            waveform = prompt_wav['waveform'].squeeze(0)
            source_sr = prompt_wav['sample_rate']
            speech = waveform.mean(dim=0,keepdim=True)
            if source_sr != prompt_sr:
                speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
        if inference_mode == '预训练音色':
            print('get sft inference request')
            print(self.model_dir)
            set_all_random_seed(seed)
            output = self.cosyvoice.inference_sft(tts_text, sft_dropdown)
        elif inference_mode == '3s极速复刻':
            print('get zero_shot inference request')
            print(self.model_dir)
            prompt_speech_16k = postprocess(speech)
            set_all_random_seed(seed)
            output = self.cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
        elif inference_mode == '跨语种复刻':
            print('get cross_lingual inference request')
            print(self.model_dir)
            prompt_speech_16k = postprocess(speech)
            set_all_random_seed(seed)
            output = self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
        else:
            print('get instruct inference request')
            set_all_random_seed(seed)
            print(self.model_dir)
            output = self.cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text)
        audio = {"waveform": [output['tts_speech']],"sample_rate":target_sr}
        return (audio,)

class CosyVoiceDubbingNode:
    def __init__(self):
        self.cosyvoice = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_srt":("SRT",),
                "language":(["<|zh|>","<|en|>","<|jp|>","<|yue|>","<|ko|>"]),
                "prompt_wav": ("AUDIO",),
                "seed":("INT",{
                    "default": 42
                })
            },
        }
    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_CosyVoice"

    def generate(self,tts_srt,language,prompt_wav,seed):
        model_dir = os.path.join(pretrained_models,"CosyVoice-300M")
        snapshot_download(model_id="iic/CosyVoice-300M",local_dir=model_dir)
        if self.cosyvoice is None:
            self.cosyvoice = CosyVoice(model_dir)
        
        with open(tts_srt, 'r', encoding="utf-8") as file:
            text_file_content = file.read()
        text_subtitles = list(SrtPare(text_file_content))

        waveform = prompt_wav['waveform'].squeeze(0)
        source_sr = prompt_wav['sample_rate']
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
        print(speech.shape)
        speech_numpy = speech.transpose(1,0).numpy()
        print(speech_numpy.shape)
        audio_seg = audiosegment.from_numpy_array(speech_numpy,prompt_sr)
        
        new_audio_seg = AudioSegment.silent(0)
        for i,text_sub in enumerate(text_subtitles):
            start_time = text_sub.start.total_seconds() * 1000
            end_time = text_sub.end.total_seconds() * 1000
            if i == 0:
                new_audio_seg += audio_seg[:start_time]
                
            curr_tts_text = language + text_sub.content
            prompt_wav_seg = audio_seg[start_time:end_time]
            prompt_wav_seg_numpy = prompt_wav_seg.to_numpy_array()
            print(prompt_wav_seg_numpy.shape)
            prompt_speech_16k = postprocess(prompt_wav_seg_numpy)
            set_all_random_seed(seed)
            curr_output = self.cosyvoice.inference_cross_lingual(curr_tts_text, prompt_speech_16k)
            curr_output_numpy = curr_output['tts_speech'].squeeze(0).numpy()

            text_audio = audiosegment.from_numpy_array(curr_output_numpy)
            text_audio_dur_time = text_audio.duration_seconds * 1000

            if i < len(text_subtitles) - 1:
                nxt_start = text_subtitles[i+1].start.total_seconds() * 1000
                dur_time =  nxt_start - start_time
            else:
                org_dur_time = audio_seg.duration_seconds * 1000
                dur_time = org_dur_time - start_time
            
            ratio = text_audio_dur_time / dur_time

            if text_audio_dur_time > dur_time:
                tmp_audio = self.map_vocal(text_audio,ratio,dur_time,f"{i}_refer.wav")
                tmp_audio += AudioSegment.silent(dur_time - tmp_audio.duration_seconds*1000)
            else:
                tmp_audio = text_audio + AudioSegment.silent(dur_time - text_audio_dur_time)
          
            new_audio_seg += tmp_audio
        output_numpy = new_audio_seg.to_numpy_array()
        print(output_numpy.shpe)
        audio = {"waveform": [torch.Tensor(output_numpy)],"sample_rate":target_sr}
        return (audio,)
    
    def map_vocal(self,audio,ratio,dur_time,wav_name):
        os.makedirs(output_dir, exist_ok=True)
        tmp_path = f"{output_dir}/map_{wav_name}"
        audio.export(tmp_path, format="wav")
        
        clone_path = f"{output_dir}/cloned_{wav_name}"
        reader = audiotsm.io.wav.WavReader(tmp_path)
        
        writer = audiotsm.io.wav.WavWriter(clone_path,channels=reader.channels,
                                        samplerate=reader.samplerate)
        wsloa = audiotsm.phasevocoder(channels=reader.channels,speed=ratio)
        wsloa.run(reader=reader,writer=writer)
        audio_extended = AudioSegment.from_file(clone_path)
        return audio_extended[:dur_time]

class LoadSRT:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1] in ["srt", "txt"]]
        return {"required":
                    {"srt": (sorted(files),)},
                }

    CATEGORY = "AIFSH_CosyVoice"

    RETURN_TYPES = ("SRT",)
    FUNCTION = "load_srt"

    def load_srt(self, srt):
        srt_path = folder_paths.get_annotated_filepath(srt)
        return (srt_path,)