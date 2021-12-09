# -*- coding: utf-8 -*-
"""네이버클라우드_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14x5bbKr1ihdfsRfK3JfmyaM3MoFul_lF
"""

ID = 'id.txt'
with open(ID) as f: ID = f.read()

Secret = 'secret.txt'
with open(Secret) as f: Secret = f.read()
# 인증 정보의 Client Secret

import json
import requests
import re

# Module 1 : 음성파일 >> 텍스트 
def print_voice(voice):

  data = open(voice, "rb") # STT를 진행하고자 하는 음성 파일

  Lang = "Kor" # Kor / Jpn / Chn / Eng
  URL = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + Lang
      

  headers = {
      "Content-Type": "application/octet-stream", # Fix
      "X-NCP-APIGW-API-KEY-ID": ID,
      "X-NCP-APIGW-API-KEY": Secret,
  }
  response = requests.post(URL,  data=data, headers=headers)
  rescode = response.status_code

  if(rescode == 200):
    ques = response.text
    question = re.sub('[^ㄱ-ㅎ ㅏ-ㅣ가-힣 ]', '', ques)      # 증상명만 출력
    return question
  else:
      return "Error : " + response.text

# Module 2 : 텍스트 >> 음성 
def text_voice(txt):
  import requests

  client_id = ID
  client_secret = ""

  text = txt

  speaker = "nara"
  speed = "0"
  pitch = "0"
  emotion = "0"
  format = "mp3"

  val = {
      "speaker": speaker,
      "speed": speed,
      "text": text
  }

  url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"

  headers = {
      "X-NCP-APIGW-API-KEY-ID": ID,
      "X-NCP-APIGW-API-KEY": Secret,
      "Content-Type": "application/x-www-form-urlencoded"
  }

  response = requests.post(url,  data=val, headers=headers)
  rescode = response.status_code

  if(rescode == 200):
      print(rescode)
      with open('cpv_sample.mp3', 'wb') as f:
          f.write(response.content)
  else:
      print("Error : " + response.text)