'''
Author: SpenserCai
Date: 2024-09-29 23:28:01
version: 
LastEditors: SpenserCai
LastEditTime: 2024-09-30 01:05:56
Description: file content
'''
import os
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import logging,load_wav
import torch
import torchaudio
import librosa

class Spk2InfoTool(CosyVoiceFrontEnd):
    def __init__(self,model_dir):
        self.max_val = 0.8
        self.prompt_sr, self.target_sr = 16000, 22050
        instruct = False
        print('{}/cosyvoice.yaml'.format(model_dir))
        self.model_dir= model_dir
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        super().__init__(configs['get_tokenizer'],
            configs['feat_extractor'],
            '{}/campplus.onnx'.format(model_dir),
            '{}/speech_tokenizer_v1.onnx'.format(model_dir),
            '{}/spk2info.pt'.format(model_dir),
            instruct,
            configs['allowed_special'])

    def list_avaliable_spks(self):
        spks = list(self.spk2info.keys())
        return spks
    
    def _postprocess(self,speech,top_db=60, hop_length=220, win_length=440):
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > self.max_val:
            speech = speech / speech.abs().max() * self.max_val
        speech = torch.concat([speech, torch.zeros(1, int(self.target_sr * 0.2))], dim=1)
        return speech
    

    def add_spk(self,spk_name,prompt_text,speech):
        prompt_speech_16k = self._postprocess(load_wav(speech, self.prompt_sr))
        # prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        _,_ = self._extract_text_token(prompt_text)
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        speech_feat, _ = self._extract_speech_feat(prompt_speech_22050)
        speech_token, _ = self._extract_speech_token(prompt_speech_16k)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        self.spk2info[spk_name] = {
          "embedding": embedding,
          "speech_feat": speech_feat,
          "speech_token": speech_token
        }

    def save_spk2info(self,save_name='spk2info.pt'):
        torch.save(self.spk2info, os.path.join(self.model_dir, save_name))

        
