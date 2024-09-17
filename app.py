import locale
import subprocess
import os
from pydub import AudioSegment
import speech_recognition as sr
from translate import Translator
import noisereduce as nr
import torch
from TTS.api import TTS

# Función para obtener la codificación preferida
def getpreferredencoding(do_setlocale=True):
  return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# Instalación de dependencias (esto se haría en la terminal, no en el código)
# !pip install ffmpeg pydub SpeechRecognition translate TTS moviepy noisereduce

# Paso 1: Extraer audio de un video
def extract_audio(input_video_path):
  output_audio_path = "only_audio.wav"
  subprocess.run(["ffmpeg", "-i", input_video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", output_audio_path])
  return output_audio_path

# Paso 2: Reducir ruido en el audio
def remove_background_noise(audio_file):
  audio = AudioSegment.from_wav(audio_file)
  reduced_noise = nr.reduce_noise(audio.get_array_of_samples(), audio.frame_rate)
  cleaned_audio = AudioSegment(data=reduced_noise.tobytes(), sample_width=audio.sample_width, frame_rate=audio.frame_rate, channels=audio.channels)
  cleaned_audio.export("vocals.wav", format="wav")
  return "vocals.wav"

# Paso 3: Transcribir y traducir audio
def transcribe_and_translate_audio(audio_path, target_language='en'):
  recognizer = sr.Recognizer()
  audio = AudioSegment.from_wav(audio_path)

  with sr.AudioFile(audio_path) as source:
      audio_data = recognizer.record(source)

  try:
      text = recognizer.recognize_google(audio_data)
      translator = Translator(to_lang=target_language)
      translated_text = translator.translate(text)
      return translated_text
  except sr.UnknownValueError:
      print("Speech Recognition could not understand audio")
      return ""
  except sr.RequestError as e:
      print(f"Could not request results from Google Speech Recognition service; {e}")
      return ""

# Paso 4: Generar audio a partir del texto traducido
def generate_audio_from_text(vocals_text, speaker_wav):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
  tts.tts_to_file(text=vocals_text, speaker_wav=speaker_wav, language="hi", file_path="output.wav")

# Paso 5: Fusionar video y audio
def merge_video_audio(input_video_path, input_audio_path):
  output_merged_path = "Final_output.mp4"
  subprocess.run(["ffmpeg", "-i", input_video_path, "-i", input_audio_path, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", "-map", "0:v:0", "-map", "1:a:0", output_merged_path])

# Ejecución del flujo de trabajo
if __name__ == "__main__":
  input_video_path = "sample_video.mp4"  # Cambia esto por la ruta de tu video
  audio_path = extract_audio(input_video_path)
  cleaned_audio_file = remove_background_noise(audio_path)
  vocals_text = transcribe_and_translate_audio(cleaned_audio_file, target_language='hi')
  generate_audio_from_text(vocals_text, cleaned_audio_file)
  merge_video_audio("only_video.mp4", "output.wav")