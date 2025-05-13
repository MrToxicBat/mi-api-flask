import os
import uuid
import logging
import base64
import re
from functools import lru_cache
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar la API Key de Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Inicializar modelo generativo
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
model = genai.GenerativeModel(MODEL_NAME)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")
CORS(app, supports_credentials=True)

# Manejo de sesiones
session_steps = {}
session_data = {}
session_admin = {}

# Prompt del sistema
SYSTEM_PROMPT = '''
Eres una inteligencia artificial médica especializada en apoyar a médicos en la evaluación y comparación de diagnósticos. Tu objetivo es proporcionar análisis clínicos basados en la información suministrada por el profesional de la salud, para ayudar a confirmar, descartar o ampliar hipótesis diagnósticas. No estás autorizada para sustituir el juicio del médico, solo para complementarlo.

Antes de generar cualquier diagnóstico diferencial, interpretación o sugerencia, debes recopilar al menos la siguiente **información clínica básica** del paciente:

1. Edad  
2. Sexo  
3. Motivo de consulta (síntoma principal, causa de la visita)  
4. Tiempo de evolución de los síntomas  
5. Antecedentes personales patológicos (enfermedades previas, condiciones crónicas, cirugías, etc.)  
6. Medicación actual (principios activos o nombres comerciales, dosis si es posible)  
7. Alergias conocidas (medicamentosas, alimentarias, ambientales, etc.)  
8. Antecedentes familiares de enfermedades relevantes (genéticas, crónicas o malignas)  
9. Estudios diagnósticos realizados (análisis clínicos, imágenes, biopsias, etc., con resultados si se conocen)
'''

questions = {
    1: "👤 Edad del paciente:",
    2: "🚻 Sexo asignado al nacer y género actual:",
    3: "📍 Motivo principal de consulta:",
    4: "⏳ ¿Desde cuándo presenta estos síntomas? ¿Han cambiado con el tiempo?",
    5: "📋 Antecedentes médicos personales (crónicos, quirúrgicos, etc.):",
    6: "💊 Medicamentos actuales (nombre, dosis, frecuencia):",
    7: "⚠️ Alergias conocidas (medicamentos, alimentos, etc.):",
    8: "👪 Antecedentes familiares relevantes:",
    9: "🧪 Estudios diagnósticos realizados y resultados si se conocen:"
}

@lru_cache(maxsize=100)
def get_cached_response(parts_key):
    try:
        # parts_key es una tupla serializada de partes para cache
        return model.generate_content(parts_key).text.strip()
    except Exception as e:
        logger.error(f"Error al generar contenido: {e}", exc_info=True)
        raise

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    session_id = data.get('session_id')
    user_message = data.get('message', '').strip()
    image_data = data.get('image')

    # Inicializar nueva sesión
    if not session_id or session_id not in session_steps:
        session_id = str(uuid.uuid4())
        session_steps[session_id] = 1
        session_data[session_id] = []
        session_admin[session_id] = False
        return jsonify({"session_id": session_id, "response": questions[1]})

    step = session_steps[session_id]
    is_admin = session_admin[session_id]

    # Activar modo admin
    if user_message.lower() == "admin":
        session_admin[session_id] = True
        return jsonify({"session_id": session_id, "response": "🔓 Modo Admin activado. Ahora puedes escribir libremente o subir imágenes."})

    # Validar respuesta del usuario
    def is_valid_response(text):
        if is_admin:
            return True
        if not text:
            return False
        if step == 1:
            return bool(re.search(r'\d{1,3}', text))
        if step == 2:
            return any(g in text.lower() for g in ["masculino", "femenino", "m", "f", "hombre", "mujer"])
        return True

    # Manejo de mensajes de texto
    if user_message:
        if is_valid_response(user_message):
            session_data[session_id].append(user_message)
            if step < len(questions):
                session_steps[session_id] += 1
                return jsonify({"session_id": session_id, "response": questions[step + 1]})
        else:
            return jsonify({"session_id": session_id, "response": "⚠️ Por favor, proporcione una respuesta válida."})

    # Preparar y enviar análisis final (texto e imagen)
    if step > len(questions) or is_admin:
        # Verificar datos obligatorios
        respuestas = dict(zip(questions.values(), session_data[session_id]))
        if not is_admin:
            for key in ["👤 Edad del paciente:", "🚻 Sexo asignado al nacer y género actual:", "📍 Motivo principal de consulta:"]:
                if not respuestas.get(key, '').strip():
                    return jsonify({"session_id": session_id, "response": "⚠️ Necesito edad, sexo y motivo de consulta para continuar."})

        # Construir prompt
        info = "\n".join(f"{i+1}. {q}\n→ {a}" for i, (q, a) in enumerate(zip(questions.values(), session_data[session_id])))
        analysis_prompt = (
            "📝 Informe Clínico Detallado\n" +
            "📌 Datos Recopilados:\n" + info + "\n\n" +
            "🔍 Análisis Clínico y sugerencias diagnósticas basadas en evidencia."
        )
        parts = [
            {"role": "system", "parts": [SYSTEM_PROMPT]},
            {"role": "user", "parts": [analysis_prompt]}
        ]
        # Adjuntar imagen si existe
        if image_data:
            try:
                img_bytes = base64.b64decode(image_data.split(',')[-1])
                parts[1]["parts"].append({"inline_data": {"mime_type": "image/png", "data": img_bytes}})
            except Exception:
                logger.warning("Imagen no procesada correctamente.", exc_info=True)

        # Generar respuesta de IA
        key = tuple(str(p) for p in parts)
        try:
            ai_response = get_cached_response(key)
            return jsonify({"session_id": session_id, "response": ai_response})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Enviar siguiente pregunta si aún faltan datos
    return jsonify({"session_id": session_id, "response": questions[step]})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
