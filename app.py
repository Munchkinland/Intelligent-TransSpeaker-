import locale
import subprocess
import os
from pydub import AudioSegment
import speech_recognition as sr
from translate import Translator
import noisereduce as nr
import torch
from TTS.api import TTS
from flask import Flask, render_template, request, send_file
import tempfile

app = Flask(__name__)

# Función para obtener la codificación preferida
def getpreferredencoding(do_setlocale=True):
  return "UTF-8"

locale.getpreferredencoding = getpreferredencoding

# Paso 1: Extraer audio de un video
def extract_audio(input_video_path):
  output_audio_path = tempfile.mktemp(suffix=".wav")
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
      output_path = tempfile.mktemp(suffix=".wav")
      cleaned_audio.export(output_path, format="wav")
      return output_path
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
      output_path = tempfile.mktemp(suffix=".wav")
      tts.tts_to_file(text=vocals_text, speaker_wav=speaker_wav, language=target_language, file_path=output_path)
      return output_path
  except Exception as e:
      print(f"Error generating audio from text: {e}")
      return None

# Paso 5: Fusionar video y audio
def merge_video_audio(input_video_path, input_audio_path):
  output_merged_path = tempfile.mktemp(suffix=".mp4")
  try:
      subprocess.run(["ffmpeg", "-i", input_video_path, "-i", input_audio_path, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", "-map", "0:v:0", "-map", "1:a:0", output_merged_path], check=True)
      return output_merged_path
  except subprocess.CalledProcessError as e:
      print(f"Error merging video and audio: {e}")
      return None

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
      output_audio = generate_audio_from_text(vocals_text, cleaned_audio_file, target_language)
      if not output_audio:
          raise Exception("Audio generation failed")

      print("Merging video and audio...")
      final_output = merge_video_audio(input_video_path, output_audio)
      if not final_output:
          raise Exception("Video and audio merging failed")

      return final_output

  except Exception as e:
      print(f"An error occurred: {str(e)}")
      raise

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
      video_file = request.files['video']
      target_language = request.form['target_language']
      
      # Save the uploaded file
      input_video_path = tempfile.mktemp(suffix=".mp4")
      video_file.save(input_video_path)
      
      try:
          final_output = process_video(input_video_path, target_language)
          return send_file(final_output, as_attachment=True, download_name="Final_output.mp4")
      except Exception as e:
          return f"An error occurred: {str(e)}", 500
      finally:
          # Clean up temporary files
          if os.path.exists(input_video_path):
              os.remove(input_video_path)

  return render_template('index.html')

@app.route('/dem')
def dem():
  return render_template('dem.html')

@app.route('/meth')
def meth():
  return render_template('meth.html')

if __name__ == '__main__':
  app.run(debug=True)