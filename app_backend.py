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

# Paso 1: Extraer audio de un video
def extract_audio(input_video_path):
  output_audio_path = "only_audio.wav"
  try:
      subprocess.run(["ffmpeg", "-i", input_video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", output_audio_path], check=True)
      return output_audio_path
  except subprocess.CalledProcessError as e:
      print(f"Error extracting audio: {e}")
      return None

# Paso 2: Reducir ruido en el audio
def remove_background_noise(audio_file):
  try:
      audio = AudioSegment.from_wav(audio_file)
      reduced_noise = nr.reduce_noise(audio.get_array_of_samples(), audio.frame_rate)
      cleaned_audio = AudioSegment(data=reduced_noise.tobytes(), sample_width=audio.sample_width, frame_rate=audio.frame_rate, channels=audio.channels)
      cleaned_audio.export("vocals.wav", format="wav")
      return "vocals.wav"
  except Exception as e:
      print(f"Error reducing background noise: {e}")
      return None

# Paso 3: Transcribir y traducir audio
def transcribe_and_translate_audio(audio_path, target_language):
  recognizer = sr.Recognizer()
  try:
      with sr.AudioFile(audio_path) as source:
          audio_data = recognizer.record(source)
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
  except Exception as e:
      print(f"Error in transcription or translation: {e}")
      return ""

# Paso 4: Generar audio a partir del texto traducido
def generate_audio_from_text(vocals_text, speaker_wav, target_language):
  try:
      device = "cuda" if torch.cuda.is_available() else "cpu"
      tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
      tts.tts_to_file(text=vocals_text, speaker_wav=speaker_wav, language=target_language, file_path="output.wav")
  except Exception as e:
      print(f"Error generating audio from text: {e}")

# Paso 5: Fusionar video y audio
def merge_video_audio(input_video_path, input_audio_path):
  output_merged_path = "Final_output.mp4"
  try:
      subprocess.run(["ffmpeg", "-i", input_video_path, "-i", input_audio_path, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", "-map", "0:v:0", "-map", "1:a:0", output_merged_path], check=True)
  except subprocess.CalledProcessError as e:
      print(f"Error merging video and audio: {e}")

def process_video(input_video_path, target_language):
  try:
      print("Extracting audio...")
      audio_path = extract_audio(input_video_path)
      if not audio_path:
          raise Exception("Audio extraction failed")

      print("Reducing background noise...")
      cleaned_audio_file = remove_background_noise(audio_path)
      if not cleaned_audio_file:
          raise Exception("Noise reduction failed")

      print("Transcribing and translating audio...")
      vocals_text = transcribe_and_translate_audio(cleaned_audio_file, target_language)
      if not vocals_text:
          raise Exception("Transcription or translation failed")

      print("Generating audio from translated text...")
      generate_audio_from_text(vocals_text, cleaned_audio_file, target_language)

      print("Merging video and audio...")
      merge_video_audio(input_video_path, "output.wav")

      print("Process completed successfully!")
      print("Created/Modified files during execution:")
      for file_name in ["only_audio.wav", "vocals.wav", "output.wav", "Final_output.mp4"]:
          if os.path.exists(file_name):
              print(file_name)

  except Exception as e:
      print(f"An error occurred: {str(e)}")

# Ejecución del flujo de trabajo
if __name__ == "__main__":
  input_video_path = input("Enter the path to your video file: ")
  target_language = input("Enter the target language code (e.g., 'en' for English, 'es' for Spanish): ")
  
  process_video(input_video_path, target_language)