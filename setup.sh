#!/bin/bash

# Versión de Python deseada
PYTHON_VERSION="3.11.0"

# Verificar si pyenv está instalado
if command -v pyenv > /dev/null; then
    echo "pyenv está instalado."
else
    echo "pyenv no está instalado. Por favor, instálalo primero."
    exit 1
fi

# Verificar si la versión de Python está instalada
if pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "La versión de Python $PYTHON_VERSION ya está instalada."
else
    echo "Instalando Python $PYTHON_VERSION..."
    pyenv install $PYTHON_VERSION
fi

# Establecer la versión de Python para el proyecto
echo "Configurando el entorno de Poetry para usar Python $PYTHON_VERSION..."
poetry env use $PYTHON_VERSION

# Crear un nuevo proyecto de Poetry si no existe
if [ ! -f "pyproject.toml" ]; then
    echo "Creando un nuevo proyecto de Poetry..."
    poetry init --no-interaction
fi

# Instalar las dependencias
echo "Instalando dependencias..."
poetry add torch numpy pydub transformers soundfile TTS openai-whisper PyQt5

# Verificar la versión de Python en el entorno virtual
echo "La versión de Python en el entorno virtual es:"
poetry run python --version

echo "Configuración completada."

#Hacer script ejecutable: chmod +x setup.sh