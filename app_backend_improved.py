import os
import subprocess
import torch
import numpy as np
import locale
import shutil
import logging
import argparse
from datetime import datetime
from pydub import AudioSegment
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf
from TTS.api import TTS

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Función para obtener la codificación preferida
def getpreferredencoding(do_setlocale=True):
  return "UTF-8"

locale.getpreferredencoding = getpreferredencoding

# Paso 1: Extraer audio de un video
def extract_audio(input_video_path):
  output_audio_path = "temp_audio.wav"
  try:
      logging.info("Extrayendo audio del video...")
      subprocess.run([
          "ffmpeg", "-y", "-i", input_video_path,
          "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_audio_path
      ], check=True)
      return output_audio_path
  except subprocess.CalledProcessError as e:
      logging.error(f"Error al extraer el audio: {e}")
      return None

# Paso 2: Reducir ruido en el audio
def remove_background_noise(audio_file):
  try:
      logging.info("Reduciendo el ruido de fondo...")
      audio, sr = sf.read(audio_file)
      # Implementación de reducción de ruido (utilizando un método simple aquí)
      # Para métodos más avanzados, se pueden utilizar otros algoritmos o librerías especializadas
      reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr)
      cleaned_audio_file = "cleaned_audio.wav"
      sf.write(cleaned_audio_file, reduced_noise_audio, sr)
      return cleaned_audio_file
  except Exception as e:
      logging.error(f"Error al reducir el ruido de fondo: {e}")
      return None

# Paso 3: Transcribir audio utilizando Whisper
def transcribe_audio_whisper(audio_path, src_language):
  try:
      logging.info("Transcribiendo el audio con Whisper...")
      model = whisper.load_model("medium")  # Se puede seleccionar el modelo según los recursos disponibles
      result = model.transcribe(audio_path, language=src_language)
      text = result["text"]
      return text
  except Exception as e:
      logging.error(f"Error en la transcripción con Whisper: {e}")
      return ""

# Paso 4: Traducir texto utilizando Hugging Face Transformers
def translate_text(text, src_language, target_language):
  try:
      logging.info("Traduciendo el texto...")
      model_name = "Helsinki-NLP/opus-mt-{}-{}".format(src_language, target_language)
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

      inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
      outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
      translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
      return translated_text
  except Exception as e:
      logging.error(f"Error en la traducción: {e}")
      return ""

# Paso 5: Generar audio a partir del texto traducido con clonación de voz
def generate_audio_from_text(translated_text, speaker_wav, target_language):
  try:
      logging.info("Generando audio a partir del texto traducido...")
      device = "cuda" if torch.cuda.is_available() else "cpu"
      tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=(device=="cuda"))

      # Configuración de parámetros
      tts_config = {
          "speaker_wav": speaker_wav,
          "language": target_language,
          "file_path": "output_audio.wav"
      }

      tts.tts_to_file(text=translated_text, **tts_config)
      return "output_audio.wav"
  except Exception as e:
      logging.error(f"Error al generar audio desde el texto: {e}")
      return None

# Paso 6: Fusionar video y audio
def merge_video_audio(input_video_path, input_audio_path):
  output_merged_path = "Final_output_{}.mp4".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
  try:
      logging.info("Fusionando video y audio...")
      subprocess.run([
          "ffmpeg", "-y", "-i", input_video_path, "-i", input_audio_path,
          "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
          output_merged_path
      ], check=True)
      return output_merged_path
  except subprocess.CalledProcessError as e:
      logging.error(f"Error al fusionar el video y el audio: {e}")
      return None

# Limpieza de archivos temporales
def cleanup_temp_files(temp_files):
  logging.info("Eliminando archivos temporales...")
  for file in temp_files:
      if os.path.exists(file):
          os.remove(file)
          logging.debug(f"Archivo eliminado: {file}")

# Función principal de procesamiento
def process_video(input_video_path, src_language, target_language):
  temp_files = []
  try:
      # Verificar que el archivo de video existe
      if not os.path.exists(input_video_path):
          logging.error("El archivo de video especificado no existe.")
          return

      # Paso 1: Extraer audio
      audio_path = extract_audio(input_video_path)
      if not audio_path:
          raise Exception("Fallo en la extracción de audio.")
      temp_files.append(audio_path)

      # Paso 2: Reducir ruido
      cleaned_audio_file = remove_background_noise(audio_path)
      if not cleaned_audio_file:
          raise Exception("Fallo en la reducción de ruido.")
      temp_files.append(cleaned_audio_file)

      # Paso 3: Transcribir audio
      vocals_text = transcribe_audio_whisper(cleaned_audio_file, src_language)
      if not vocals_text:
          raise Exception("Fallo en la transcripción de audio.")

      # Paso 4: Traducir texto
      translated_text = translate_text(vocals_text, src_language, target_language)
      if not translated_text:
          raise Exception("Fallo en la traducción del texto.")

      # Paso 5: Generar audio desde el texto traducido
      output_audio = generate_audio_from_text(translated_text, cleaned_audio_file, target_language)
      if not output_audio:
          raise Exception("Fallo en la generación de audio desde el texto.")
      temp_files.append(output_audio)

      # Paso 6: Fusionar video y audio
      final_output = merge_video_audio(input_video_path, output_audio)
      if not final_output:
          raise Exception("Fallo en la fusión de video y audio.")

      logging.info("¡Proceso completado exitosamente!")
      logging.info(f"Archivo de salida: {final_output}")

  except Exception as e:
      logging.error(f"Ha ocurrido un error: {e}")

  finally:
      # Limpieza de archivos temporales
      cleanup_temp_files(temp_files)

# Función para parsear argumentos de línea de comandos
def parse_arguments():
  parser = argparse.ArgumentParser(description="Clonación y traducción de voz en videos.")
  parser.add_argument("-i", "--input", required=True, help="Ruta al archivo de video.")
  parser.add_argument("-s", "--source_language", default="es", help="Código del idioma fuente (ejemplo: 'es' para español).")
  parser.add_argument("-t", "--target_language", default="en", help="Código del idioma objetivo (ejemplo: 'en' para inglés).")
  args = parser.parse_args()
  return args

# Ejecución del flujo de trabajo
if __name__ == "__main__":
  args = parse_arguments()
  process_video(args.input, args.source_language, args.target_language)