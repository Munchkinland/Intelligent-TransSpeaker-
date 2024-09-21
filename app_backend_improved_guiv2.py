import os
import subprocess
import torch
import numpy as np
import locale
import shutil
import logging
from datetime import datetime
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf
import noisereduce as nr
from TTS.api import TTS
import sys

from PyQt5.QtWidgets import (
  QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
  QHBoxLayout, QVBoxLayout, QMessageBox, QProgressBar, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

print("Iniciando la aplicación...")

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Función para obtener la codificación preferida
def getpreferredencoding(do_setlocale=True):
  return "UTF-8"

locale.getpreferredencoding = getpreferredencoding

# Mapeo de códigos de idioma para TTS
TTS_LANG_MAP = {
  "en": "en",
  "es": "es",
  "fr": "fr-fr",
  "pt": "pt-br",
  "de": "de",
  "it": "it-it",
}

# Lista de idiomas soportados por TTS para el ComboBox
SUPPORTED_LANGUAGES = ["en", "es", "fr", "pt", "de", "it"]

# Clase para el procesamiento en segundo plano
class VideoProcessingThread(QThread):
  progress = pyqtSignal(int)
  finished = pyqtSignal(str)
  error = pyqtSignal(str)

  def __init__(self, input_video_path, src_language, target_language):
      super().__init__()
      self.input_video_path = input_video_path
      self.src_language = src_language
      self.target_language = target_language

  def run(self):
      print(f"Iniciando procesamiento del video: {self.input_video_path}")
      temp_files = []
      try:
          if not os.path.exists(self.input_video_path):
              raise Exception("El archivo de video especificado no existe.")

          # Paso 1: Extraer audio
          print("Paso 1: Extrayendo audio del video...")
          audio_path = extract_audio(self.input_video_path)
          if not audio_path:
              raise Exception("Fallo en la extracción de audio.")
          temp_files.append(audio_path)
          self.progress.emit(10)

          # Paso 2: Reducir ruido
          print("Paso 2: Reduciendo ruido de fondo...")
          cleaned_audio_file = remove_background_noise(audio_path)
          if not cleaned_audio_file:
              raise Exception("Fallo en la reducción de ruido.")
          temp_files.append(cleaned_audio_file)
          self.progress.emit(30)

          # Paso 3: Segmentar audio basado en silencio
          print("Paso 3: Segmentando audio...")
          audio_chunks, pauses = segment_audio_based_on_silence(cleaned_audio_file, min_silence_len=1000, silence_thresh=-40)
          self.progress.emit(50)

          # Paso 4: Transcribir y traducir cada segmento
          translated_segments = []
          for chunk in audio_chunks:
              print("Transcribiendo segmento...")
              vocals_text = transcribe_audio_whisper(chunk, self.src_language)
              if not vocals_text:
                  logging.warning("Segmento no transcrito correctamente, omitiendo.")
                  continue  # Continuar con el siguiente segmento

              print("Traduciendo segmento...")
              translated_text = translate_text(vocals_text, self.src_language, self.target_language)
              if not translated_text:
                  logging.warning("Traducción fallida, omitiendo segmento.")
                  continue  # Continuar con el siguiente segmento
              
              translated_segments.append(translated_text)
              self.progress.emit(70)

          # Paso 5: Generar audio desde el texto traducido
          print("Paso 5: Generando audio desde el texto traducido...")
          output_audio = generate_audio_from_text(" ".join(translated_segments), cleaned_audio_file, self.target_language, pauses)
          if not output_audio:
              raise Exception("Fallo en la generación de audio desde el texto.")
          temp_files.append(output_audio)
          self.progress.emit(90)

          # Paso 6: Fusionar video y audio
          print("Paso 6: Fusionando video y audio...")
          final_output = merge_video_audio(self.input_video_path, output_audio)
          if not final_output:
              raise Exception("Fallo en la fusión de video y audio.")
          self.progress.emit(100)

          print(f"Procesamiento completado. Archivo de salida: {final_output}")
          self.finished.emit(final_output)

      except Exception as e:
          print(f"Error durante el procesamiento: {str(e)}")
          self.error.emit(str(e))

      finally:
          # Limpieza de archivos temporales
          cleanup_temp_files(temp_files)

# Definición de funciones
def extract_audio(input_video_path):
  output_audio_path = "temp_audio.wav"
  try:
      logging.info("Extrayendo audio del video...")
      subprocess.run([
          "ffmpeg", "-y", "-i", input_video_path,
          "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_audio_path,
          "-loglevel", "quiet"
      ], check=True)
      return output_audio_path
  except subprocess.CalledProcessError as e:
      logging.error(f"Error al extraer el audio: {e}")
      return None

def remove_background_noise(audio_file):
  try:
      logging.info("Reduciendo el ruido de fondo...")
      audio, sr = sf.read(audio_file)
      logging.info(f"Audio leído: {audio.shape}, Frecuencia de muestreo: {sr}")
      reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr)
      cleaned_audio_file = "cleaned_audio.wav"
      sf.write(cleaned_audio_file, reduced_noise_audio, sr)
      logging.info("Reducción de ruido completada.")
      return cleaned_audio_file
  except Exception as e:
      logging.error(f"Error al reducir el ruido de fondo: {e}")
      return None

def segment_audio_based_on_silence(audio_path, min_silence_len=1000, silence_thresh=-40):
  audio = AudioSegment.from_wav(audio_path)
  # Detectar segmentos no silenciosos
  nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
  
  # Crear segmentos de audio basados en los rangos no silenciosos
  audio_chunks = [audio[start:end] for start, end in nonsilent_ranges]
  
  # Calcular las pausas en milisegundos
  pauses = []
  for i in range(len(nonsilent_ranges) - 1):
      pause_duration = nonsilent_ranges[i + 1][0] - nonsilent_ranges[i][1]
      pauses.append(pause_duration)

  return audio_chunks, pauses

def transcribe_audio_whisper(audio_segment, src_language):
  try:
      logging.info("Transcribiendo el audio con Whisper...")
      device = "cuda" if torch.cuda.is_available() else "cpu"
      model = whisper.load_model("small").to(device)
      audio_segment.export("temp_chunk.wav", format="wav")
      result = model.transcribe("temp_chunk.wav", language=src_language)
      text = result["text"]
      return text
  except Exception as e:
      logging.error(f"Error en la transcripción con Whisper: {e}")
      return ""

def translate_text(text, src_language, target_language):
  try:
      logging.info("Traduciendo el texto...")
      model_name = "Helsinki-NLP/opus-mt-{}-{}".format(src_language, target_language)
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

      max_length = 512  # Longitud máxima permitida por el modelo
      sentences = text.replace('\n', ' ').split('. ')  # Dividir el texto en oraciones
      translated_sentences = []
      for sentence in sentences:
          # Codificar y truncar si es necesario
          inputs = tokenizer.encode(sentence, return_tensors="pt", max_length=max_length, truncation=True)
          outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
          translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
          translated_sentences.append(translated_text)

      return '. '.join(translated_sentences)
  except Exception as e:
      logging.error(f"Error en la traducción: {e}")
      return ""

def generate_audio_from_text(translated_text, speaker_wav, target_language, pauses):
  try:
      logging.info("Generando audio a partir del texto traducido...")
      device = "cuda" if torch.cuda.is_available() else "cpu"

      tts_language = TTS_LANG_MAP.get(target_language)
      if not tts_language:
          logging.error(f"Idioma objetivo '{target_language}' no soportado por TTS.")
          return None

      tts_model_name = "tts_models/multilingual/multi-dataset/your_tts"
      tts = TTS(model_name=tts_model_name, progress_bar=False, gpu=(device == "cuda"))

      sentences = translated_text.replace('\n', ' ').split('. ')
      audio_segments = []
      for idx, sentence in enumerate(sentences):
          temp_audio_file = f"temp_output_audio_{idx}.wav"
          tts.tts_to_file(text=sentence, speaker_wav=speaker_wav, language=tts_language, file_path=temp_audio_file)
          audio_segments.append(temp_audio_file)

          # Insertar silencio después de cada segmento
          if idx < len(pauses):
              silence_duration = pauses[idx]  # Duración del silencio en milisegundos
              silence = AudioSegment.silent(duration=silence_duration)
              audio_segments.append(silence)

      # Concatenar archivos de audio
      combined_audio = AudioSegment.empty()
      for audio_file in audio_segments:
          if isinstance(audio_file, str):  # Si es un archivo de audio
              segment = AudioSegment.from_wav(audio_file)
              combined_audio += segment
              os.remove(audio_file)  # Eliminar archivo de audio temporal
          else:  # Si es un silencio
              combined_audio += audio_file

      output_audio_path = "output_audio.wav"
      combined_audio.export(output_audio_path, format="wav")

      return output_audio_path
  except Exception as e:
      logging.error(f"Error al generar audio desde el texto: {e}")
      return None

def merge_video_audio(input_video_path, input_audio_path):
  output_merged_path = "Final_output_{}.mp4".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
  try:
      logging.info("Fusionando video y audio...")
      subprocess.run([
          "ffmpeg", "-y", "-loglevel", "quiet", "-i", input_video_path, "-i", input_audio_path,
          "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
          output_merged_path
      ], check=True)
      return output_merged_path
  except subprocess.CalledProcessError as e:
      logging.error(f"Error en la fusión de video y audio: {e}")
      return None

def cleanup_temp_files(temp_files):
  for temp_file in temp_files:
      if os.path.exists(temp_file):
          os.remove(temp_file)

# Interfaz gráfica
class VideoTranscriptionApp(QWidget):
  def __init__(self):
      super().__init__()
      self.init_ui()

  def init_ui(self):
      self.setWindowTitle("Transcriptor y Traductor de Videos")
      self.setGeometry(100, 100, 400, 200)

      self.layout = QVBoxLayout()

      self.video_label = QLabel("Seleccionar video:")
      self.layout.addWidget(self.video_label)

      self.video_path_input = QLineEdit(self)
      self.layout.addWidget(self.video_path_input)

      self.browse_button = QPushButton("Explorar", self)
      self.browse_button.clicked.connect(self.browse_video)
      self.layout.addWidget(self.browse_button)

      self.src_language_label = QLabel("Idioma de origen:")
      self.layout.addWidget(self.src_language_label)

      self.src_language_combo = QComboBox(self)
      self.src_language_combo.addItems(SUPPORTED_LANGUAGES)
      self.layout.addWidget(self.src_language_combo)

      self.target_language_label = QLabel("Idioma de destino:")
      self.layout.addWidget(self.target_language_label)

      self.target_language_combo = QComboBox(self)
      self.target_language_combo.addItems(SUPPORTED_LANGUAGES)
      self.layout.addWidget(self.target_language_combo)

      self.start_button = QPushButton("Iniciar Proceso", self)
      self.start_button.clicked.connect(self.start_process)
      self.layout.addWidget(self.start_button)

      self.progress_bar = QProgressBar(self)
      self.layout.addWidget(self.progress_bar)

      self.setLayout(self.layout)

  def browse_video(self):
      options = QFileDialog.Options()
      file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar video", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options)
      if file_path:
          self.video_path_input.setText(file_path)

  def start_process(self):
      input_video_path = self.video_path_input.text()
      src_language = self.src_language_combo.currentText()
      target_language = self.target_language_combo.currentText()

      if not input_video_path:
          QMessageBox.warning(self, "Error", "Por favor, seleccione un archivo de video.")
          return

      self.thread = VideoProcessingThread(input_video_path, src_language, target_language)
      self.thread.progress.connect(self.update_progress)
      self.thread.finished.connect(self.show_result)
      self.thread.error.connect(self.show_error)
      self.thread.start()

  def update_progress(self, value):
      self.progress_bar.setValue(value)

  def show_result(self, result):
      QMessageBox.information(self, "Proceso Completado", f"Archivo final creado: {result}")

  def show_error(self, error_message):
      QMessageBox.critical(self, "Error", f"Ocurrió un error: {error_message}")

if __name__ == '__main__':
  app = QApplication(sys.argv)
  window = VideoTranscriptionApp()
  window.show()
  sys.exit(app.exec_())