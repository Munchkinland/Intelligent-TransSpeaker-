import os
import subprocess
import torch
import numpy as np
import locale
import shutil
import logging
from datetime import datetime
from pydub import AudioSegment
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf
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
  # Añade más mapeos según sea necesario
}

# Lista de idiomas soportados por TTS para el ComboBox
SUPPORTED_LANGUAGES = ["en", "es", "fr", "pt", "de", "it"]  # Extiende esta lista según TTS

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

          # Paso 3: Transcribir audio
          print("Paso 3: Transcribiendo audio...")
          vocals_text = transcribe_audio_whisper(cleaned_audio_file, self.src_language)
          if not vocals_text:
              raise Exception("Fallo en la transcripción de audio.")
          self.progress.emit(50)

          # Paso 4: Traducir texto
          print("Paso 4: Traduciendo texto...")
          translated_text = translate_text(vocals_text, self.src_language, self.target_language)
          if not translated_text:
              raise Exception("Fallo en la traducción del texto.")
          self.progress.emit(70)

          # Paso 5: Generar audio desde el texto traducido
          print("Paso 5: Generando audio desde el texto traducido...")
          output_audio = generate_audio_from_text(translated_text, cleaned_audio_file, self.target_language)
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
      # Implementación de reducción de ruido (simplificada)
      reduced_noise_audio = audio  # Aquí podrías aplicar un algoritmo de reducción de ruido
      cleaned_audio_file = "cleaned_audio.wav"
      sf.write(cleaned_audio_file, reduced_noise_audio, sr)
      return cleaned_audio_file
  except Exception as e:
      logging.error(f"Error al reducir el ruido de fondo: {e}")
      return None

def transcribe_audio_whisper(audio_path, src_language):
  try:
      logging.info("Transcribiendo el audio con Whisper...")
      model = whisper.load_model("medium")  # Puedes elegir 'tiny', 'base', 'small', 'medium', o 'large'
      result = model.transcribe(audio_path, language=src_language)
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

      inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
      outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
      translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
      return translated_text
  except Exception as e:
      logging.error(f"Error en la traducción: {e}")
      return ""

def generate_audio_from_text(translated_text, speaker_wav, target_language):
  try:
      logging.info("Generando audio a partir del texto traducido...")
      device = "cuda" if torch.cuda.is_available() else "cpu"

      # Mapeo del idioma al formato esperado por TTS
      tts_language = TTS_LANG_MAP.get(target_language)
      if not tts_language:
          logging.error(f"Idioma objetivo '{target_language}' no soportado por TTS.")
          return None

      # Cargar el modelo de TTS adecuado
      tts_model_name = "tts_models/multilingual/multi-dataset/your_tts"  # Asegúrate de que este modelo soporte los idiomas mapeados
      tts = TTS(model_name=tts_model_name, progress_bar=False, gpu=(device=="cuda"))

      # Configuración de parámetros
      tts_config = {
          "speaker_wav": speaker_wav,
          "language": tts_language,
          "file_path": "output_audio.wav"
      }

      tts.tts_to_file(text=translated_text, **tts_config)
      return "output_audio.wav"
  except Exception as e:
      logging.error(f"Error al generar audio desde el texto: {e}")
      return None

def merge_video_audio(input_video_path, input_audio_path):
  output_merged_path = "Final_output_{}.mp4".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
  try:
      logging.info("Fusionando video y audio...")
      subprocess.run([
          "ffmpeg", "-y", "-i", input_video_path, "-i", input_audio_path,
          "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
          output_merged_path,
          "-loglevel", "quiet"
      ], check=True)
      return output_merged_path
  except subprocess.CalledProcessError as e:
      logging.error(f"Error al fusionar el video y el audio: {e}")
      return None

def cleanup_temp_files(temp_files):
  logging.info("Eliminando archivos temporales...")
  for file in temp_files:
      if os.path.exists(file):
          os.remove(file)
          logging.debug(f"Archivo eliminado: {file}")

# Clase de la ventana principal de la GUI
class VoiceCloneApp(QWidget):
  def __init__(self):
      super().__init__()
      print("Inicializando la interfaz gráfica...")
      self.setWindowTitle("Clonación y Traducción de Voz en Videos")
      self.setGeometry(100, 100, 600, 250)
      self.init_ui()

  def init_ui(self):
      layout = QVBoxLayout()

      # Selector de archivo de video
      self.video_label = QLabel("Archivo de video:")
      self.video_path_input = QLineEdit()
      self.video_browse_button = QPushButton("Examinar")
      self.video_browse_button.clicked.connect(self.browse_video_file)

      video_layout = QHBoxLayout()
      video_layout.addWidget(self.video_label)
      video_layout.addWidget(self.video_path_input)
      video_layout.addWidget(self.video_browse_button)

      # Selección de idioma fuente
      self.source_language_label = QLabel("Idioma fuente:")
      self.source_language_combo = QComboBox()
      self.source_language_combo.addItems(SUPPORTED_LANGUAGES)  # Usa la lista de idiomas soportados

      source_language_layout = QHBoxLayout()
      source_language_layout.addWidget(self.source_language_label)
      source_language_layout.addWidget(self.source_language_combo)

      # Selección de idioma objetivo
      self.target_language_label = QLabel("Idioma objetivo:")
      self.target_language_combo = QComboBox()
      self.target_language_combo.addItems(SUPPORTED_LANGUAGES)  # Usa la lista de idiomas soportados

      target_language_layout = QHBoxLayout()
      target_language_layout.addWidget(self.target_language_label)
      target_language_layout.addWidget(self.target_language_combo)

      # Botón de inicio
      self.start_button = QPushButton("Iniciar procesamiento")
      self.start_button.clicked.connect(self.start_processing)

      # Barra de progreso
      self.progress_bar = QProgressBar()
      self.progress_bar.setAlignment(Qt.AlignCenter)

      # Añadir widgets al layout
      layout.addLayout(video_layout)
      layout.addLayout(source_language_layout)
      layout.addLayout(target_language_layout)
      layout.addWidget(self.start_button)
      layout.addWidget(self.progress_bar)

      self.setLayout(layout)

  def browse_video_file(self):
      print("Seleccionando archivo de video...")
      options = QFileDialog.Options()
      file_name, _ = QFileDialog.getOpenFileName(
          self,
          "Seleccionar archivo de video",
          "",
          "Videos (*.mp4 *.avi *.mov)",
          options=options
      )
      if file_name:
          print(f"Archivo de video seleccionado: {file_name}")
          self.video_path_input.setText(file_name)

  def start_processing(self):
      print("Iniciando el procesamiento del video...")
      video_path = self.video_path_input.text()
      src_language = self.source_language_combo.currentText()
      target_language = self.target_language_combo.currentText()

      if not video_path:
          print("Advertencia: No se ha seleccionado ningún archivo de video.")
          QMessageBox.warning(self, "Advertencia", "Por favor, selecciona un archivo de video.")
          return

      print(f"Configuración: Video: {video_path}, Idioma fuente: {src_language}, Idioma objetivo: {target_language}")
      self.thread = VideoProcessingThread(video_path, src_language, target_language)
      self.thread.progress.connect(self.update_progress)
      self.thread.finished.connect(self.processing_finished)
      self.thread.error.connect(self.processing_error)
      self.thread.start()
      self.start_button.setEnabled(False)
      self.progress_bar.setValue(0)

  def update_progress(self, value):
      print(f"Progreso: {value}%")
      self.progress_bar.setValue(value)

  def processing_finished(self, output_file):
      print(f"Procesamiento completado. Archivo de salida: {output_file}")
      self.start_button.setEnabled(True)
      QMessageBox.information(self, "Éxito", f"¡Procesamiento completado!\nArchivo de salida: {output_file}")
      self.progress_bar.setValue(100)

  def processing_error(self, error_message):
      print(f"Error durante el procesamiento: {error_message}")
      self.start_button.setEnabled(True)
      QMessageBox.critical(self, "Error", f"Ha ocurrido un error:\n{error_message}")
      self.progress_bar.setValue(0)

# Función principal
def main():
  print("Iniciando la aplicación principal...")
  app = QApplication(sys.argv)
  window = VoiceCloneApp()
  window.show()
  print("Aplicación iniciada y en ejecución.")
  sys.exit(app.exec_())

if __name__ == "__main__":
  main()