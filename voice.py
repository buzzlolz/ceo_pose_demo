# Import the Gtts module for text  
# to speech conversion 
from gtts import gTTS 
  
# import Os module to start the audio file
import os 
  
# mytext = 'ok good'
  
# # Language we want to use 
# language = 'en'
# myobj = gTTS(text=mytext, lang=language, slow=False) 
# myobj.save("ok.mp3") 
  
# Play the converted file 
# os.system("start output.mp3")

tts = gTTS("身體挺直一點", lang="zh-tw")
tts.save("audio/straight_body.mp3")


