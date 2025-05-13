import os
import uuid
import logging
import base64
import re
from functools import lru_cache
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import google.generativeai as genai

# Configuración de logging
typing=logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración del API de Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")

# Inicializar Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")
CORS(app, supports_credentials=True)

# Variables globales de sesiones
session_steps = {}
session_data = {}
session_admin = {}

# Instrucción del sistema para la IA
SYSTEM_PROMPT = """Eres una inteligencia artificial médica especializada en apoyar a médicos en la evaluación y comparación de diagnósticos. Tu objetivo es proporcionar análisis clínicos basados en la información suministrada por el profesional de la salud, para ayudar a confirmar, descartar o ampliar hipótesis diagnósticas. No estás autorizada para sustituir el juicio del médico, solo para complementarlo.

Antes de generar cualquier diagnóstico diferencial, interpretación o sugerencia, debes recopilar al menos la siguiente información clínica básica del paciente:
1. Edad
2. Sexo
3. Motivo de consulta (síntoma principal)
4. Duración de los síntomas
5. Intensidad
6. Antecedentes médicos
"""

@lru_cache(maxsize=128)
def get_cached_response(prompt):
    try:
        response = genai.chat.create(model=MODEL_NAME, prompt=prompt)
        return response.last
    except Exception as e:
        logger.error(f"Error al llamar a Gemini: {e}")
        raise

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_input = data.get("message", "")
    session_id = data.get("session_id", str(uuid.uuid4()))

    # Inicializar datos de sesión si no existen
    if session_id not in session_data:
        session_data[session_id] = []
        session_steps[session_id] = 0
        # Agregar mensaje de sistema al historial
        session_data[session_id].append({"role": "system", "content": SYSTEM_PROMPT})

    # Agregar mensaje del usuario
    session_data[session_id].append({"role": "user", "content": user_input})

    # Preparar prompt completo
    messages = session_data[session_id]
    full_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    try:
        ai_response = get_cached_response(full_prompt)
        # Agregar respuesta de IA al historial
        session_data[session_id].append({"role": "assistant", "content": ai_response})
        return jsonify({"session_id": session_id, "response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/image", methods=["POST"])
def analyze_image():
    data = request.json or {}
    image_data = data.get("image_base64", "")
    session_id = data.get("session_id", str(uuid.uuid4()))

    if not image_data:
        return jsonify({"error": "No se proporcionó imagen"}), 400

    # Decodificar imagen
    try:
        content = base64.b64decode(image_data.split(",")[-1])
        response = genai.chat.create(
            model=MODEL_NAME,
            prompt="Analiza profundamente esta imagen y describe lo que ves.",
            image=content
        )
        description = response.last
        return jsonify({"session_id": session_id, "description": description})
    except Exception as e:
        logger.error(f"Error al analizar imagen: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
