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

import ffmpeg
import audiosegment
from srt import parse as SrtPare
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

def speed_change(input_audio, speed, sr):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")


    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio

class TextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "encode"

    CATEGORY = "AIFSH_CosyVoice"

    def encode(self,text):
        return (text, )

from time import time as ttime
class CosyVoiceNode:
    def __init__(self):
        self.model_dir = None
        self.cosyvoice = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("TEXT",),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
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

    def generate(self,tts_text,speed,inference_mode,sft_dropdown,seed,
                 prompt_text=None,prompt_wav=None,instruct_text=None):
        t0 = ttime()
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
            snapshot_download(model_id="iic/CosyVoice-300M-SFT",local_dir=model_dir)


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
        output_list =［］
        for out_dict in output:
            output_numpy = out_dict['tts_speech'].squeeze(0).numpy() * 32768 
            output_numpy = output_numpy.astype(np.int16)
            if speed > 1.0 or speed < 1.0:
                output_numpy = speed_change(output_numpy,speed,target_sr)
            output_list.append(torch.Tensor(output_numpy/32768).unsqueeze(0))
        t1 = ttime()
        print("cost time \t %.3f" % (t1-t0))
        audio = {"waveform": torch.stack(output_list),"sample_rate":target_sr}
        return (audio,)

class CosyVoiceDubbingNode:
    def __init__(self):
        self.cosyvoice = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_srt":("SRT",),
                "prompt_wav": ("AUDIO",),
                "language":(["<|zh|>","<|en|>","<|jp|>","<|yue|>","<|ko|>"],),
                "if_single":("BOOLEAN",{
                    "default": True
                }),
                "seed":("INT",{
                    "default": 42
                })
            },
            "optional":{
                "prompt_srt":("SRT",),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_CosyVoice"

    def generate(self,tts_srt,prompt_wav,language,if_single,seed,prompt_srt=None):
        model_dir = os.path.join(pretrained_models,"CosyVoice-300M")
        snapshot_download(model_id="iic/CosyVoice-300M",local_dir=model_dir)
        set_all_random_seed(seed)
        if self.cosyvoice is None:
            self.cosyvoice = CosyVoice(model_dir)
        
        with open(tts_srt, 'r', encoding="utf-8") as file:
            text_file_content = file.read()
        text_subtitles = list(SrtPare(text_file_content))

        if prompt_srt:
            with open(prompt_srt, 'r', encoding="utf-8") as file:
                prompt_file_content = file.read()
            prompt_subtitles = list(SrtPare(prompt_file_content))

        waveform = prompt_wav['waveform'].squeeze(0)
        source_sr = prompt_wav['sample_rate']
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
        speech_numpy = speech.squeeze(0).numpy() * 32768
        speech_numpy = speech_numpy.astype(np.int16)
        audio_seg = audiosegment.from_numpy_array(speech_numpy,prompt_sr)
        assert audio_seg.duration_seconds > 3, "prompt wav should be > 3s"
        # audio_seg.export(os.path.join(output_dir,"test.mp3"),format="mp3")
        new_audio_seg = audiosegment.silent(0,target_sr)
        for i,text_sub in enumerate(text_subtitles):
            start_time = text_sub.start.total_seconds() * 1000
            end_time = text_sub.end.total_seconds() * 1000
            if i == 0:
                new_audio_seg += audio_seg[:start_time]
            
            if if_single:
                curr_tts_text = language + text_sub.content
            else:
                curr_tts_text = language + text_sub.content[1:]
                speaker_id = text_sub.content[0]
            
            prompt_wav_seg = audio_seg[start_time:end_time]
            if prompt_srt:
                prompt_text_list = [prompt_subtitles[i].content]
            while prompt_wav_seg.duration_seconds < 30:
                for j in range(i+1,len(text_subtitles)):
                    j_start = text_subtitles[j].start.total_seconds() * 1000
                    j_end = text_subtitles[j].end.total_seconds() * 1000
                    if if_single:
                        prompt_wav_seg += (audiosegment.silent(500,frame_rate=prompt_sr) + audio_seg[j_start:j_end])
                        if prompt_srt:
                            prompt_text_list.append(prompt_subtitles[j].content)
                    else:
                        if text_subtitles[j].content[0] == speaker_id:
                            prompt_wav_seg += (audiosegment.silent(500,frame_rate=prompt_sr) + audio_seg[j_start:j_end])
                            if prompt_srt:
                                prompt_text_list.append(prompt_subtitles[j].content)
                for j in range(0,i):
                    j_start = text_subtitles[j].start.total_seconds() * 1000
                    j_end = text_subtitles[j].end.total_seconds() * 1000
                    if if_single:
                        prompt_wav_seg += (audiosegment.silent(500,frame_rate=prompt_sr) + audio_seg[j_start:j_end])
                        if prompt_srt:
                            prompt_text_list.append(prompt_subtitles[j].content)
                    else:
                        if text_subtitles[j].content[0] == speaker_id:
                            prompt_wav_seg += (audiosegment.silent(500,frame_rate=prompt_sr) + audio_seg[j_start:j_end])
                            if prompt_srt:
                                prompt_text_list.append(prompt_subtitles[j].content)

                if prompt_wav_seg.duration_seconds > 3:
                    break
            print(f"prompt_wav {prompt_wav_seg.duration_seconds}s")
            prompt_wav_seg.export(os.path.join(output_dir,f"{i}_prompt.wav"),format="wav")
            prompt_wav_seg_numpy = prompt_wav_seg.to_numpy_array() / 32768
            # print(prompt_wav_seg_numpy.shape)
            prompt_speech_16k = postprocess(torch.Tensor(prompt_wav_seg_numpy).unsqueeze(0))
            if prompt_srt:
                # prompt_text = prompt_subtitles[i].content
                prompt_text = ','.join(prompt_text_list)
                print(f"prompt_text:{prompt_text}")
                curr_output = self.cosyvoice.inference_zero_shot(curr_tts_text,prompt_text,prompt_speech_16k)
            else:
                curr_output = self.cosyvoice.inference_cross_lingual(curr_tts_text, prompt_speech_16k)
            
            curr_output_numpy = curr_output['tts_speech'].squeeze(0).numpy() * 32768
            # print(curr_output_numpy.shape)
            curr_output_numpy = curr_output_numpy.astype(np.int16)
            text_audio = audiosegment.from_numpy_array(curr_output_numpy,target_sr)
            # text_audio.export(os.path.join(output_dir,f"{i}_res.wav"),format="wav")
            text_audio_dur_time = text_audio.duration_seconds * 1000

            if i < len(text_subtitles) - 1:
                nxt_start = text_subtitles[i+1].start.total_seconds() * 1000
                dur_time =  nxt_start - start_time
            else:
                org_dur_time = audio_seg.duration_seconds * 1000
                dur_time = org_dur_time - start_time
            
            ratio = text_audio_dur_time / dur_time

            if text_audio_dur_time > dur_time:
                tmp_numpy = speed_change(curr_output_numpy,ratio,target_sr)
                tmp_audio = audiosegment.from_numpy_array(tmp_numpy,target_sr)
                # tmp_audio = self.map_vocal(text_audio,ratio,dur_time,f"{i}_res.wav")
                tmp_audio += audiosegment.silent(dur_time - tmp_audio.duration_seconds*1000,target_sr)
            else:
                tmp_audio = text_audio + audiosegment.silent(dur_time - text_audio_dur_time,target_sr)
          
            new_audio_seg += tmp_audio

            if i == len(text_subtitles) - 1:
                new_audio_seg += audio_seg[end_time:]

        output_numpy = new_audio_seg.to_numpy_array() / 32768
        # print(output_numpy.shape)
        audio = {"waveform": torch.stack([torch.Tensor(output_numpy).unsqueeze(0)]),"sample_rate":target_sr}
        return (audio,)

    
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