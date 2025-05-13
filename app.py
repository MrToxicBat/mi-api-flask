import os
import uuid
import logging
import base64
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from functools import lru_cache

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar la API de Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

# Estado por sesión
session_steps = {}
session_data = {}
session_admin = {}

SYSTEM_PROMPT = '''
Eres una inteligencia artificial médica especializada en apoyar a médicos en la evaluación y comparación de diagnósticos. Tu objetivo es proporcionar análisis clínicos basados en la información suministrada por el profesional de la salud, para ayudar a confirmar, descartar o ampliar hipótesis diagnósticas. No estás autorizada para sustituir el juicio del médico, solo para complementarlo.
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
def get_cached_response(parts):
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(parts)
    return getattr(response, 'text', '').strip()

@app.route('/api/chat', methods=['POST'])
def chat():
    data       = request.json or {}
    session_id = data.get('session_id')
    user_msg   = data.get('message', '').strip()
    image_data = data.get('image')

    # ── PRIMERA RAMA: si viene una imagen la procesamos de inmediato ─────────────────
    if image_data:
        # Aseguramos que exista la sesión
        if not session_id or session_id not in session_steps:
            session_id = str(uuid.uuid4())
        session_steps.setdefault(session_id, 1)
        session_data.setdefault(session_id, [])
        # Activamos modo admin para permitir libertad tras la descripción
        session_admin[session_id] = True

        # Decodificar la imagen y generar prompt a Gemini
        try:
            image_bytes = base64.b64decode(image_data.split(',')[-1])
            parts = [
                {"role": "system", "parts": [SYSTEM_PROMPT]},
                {"role": "user", "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": image_bytes}},
                    # Instrucción para el modelo
                    "Por favor, describe detalladamente lo que ves en esta imagen y luego pregunta al solicitante qué te gustaría que haga a continuación."
                ]}
            ]
            ai_response = get_cached_response(tuple(map(str, parts)))
            return jsonify({"session_id": session_id, "response": ai_response})
        except Exception as e:
            logger.error("Error procesando imagen", exc_info=True)
            return jsonify({
                "session_id": session_id,
                "response": "⚠️ Hubo un error al procesar la imagen. Intenta de nuevo."
            })

    # ── SEGUNDA RAMA: creación inicial de sesión (si no existía) ───────────────────────
    if not session_id or session_id not in session_steps:
        session_id = str(uuid.uuid4())
        session_steps[session_id] = 1
        session_data[session_id]  = []
        session_admin[session_id] = False
        # Iniciamos el cuestionario
        return jsonify({"session_id": session_id, "response": questions[1]})

    # ── A partir de aquí ya tenemos session_id existente ────────────────────────────────
    step    = session_steps[session_id]
    is_admin= session_admin.get(session_id, False)

    # Activar modo admin si el usuario envía “admin”
    if user_msg.lower() == "admin":
        session_admin[session_id] = True
        return jsonify({
            "session_id": session_id,
            "response": "🔓 Modo Admin activado. Ahora puedes escribir libremente o subir imágenes."
        })

    # Validaciones y avances en el cuestionario
    def is_valid_response(text):
        if is_admin: return True
        if not text.strip(): return False
        if step == 1:
            return bool(re.search(r'\d{1,3}', text))
        if step == 2:
            return any(g in text.lower() for g in ["masculino","femenino","m","f","hombre","mujer"])
        return True

    if user_msg:
        if is_valid_response(user_msg):
            session_data[session_id].append(user_msg)
            if step < len(questions):
                session_steps[session_id] += 1
                return jsonify({
                  "session_id": session_id,
                  "response": questions[step+1]
                })
        else:
            return jsonify({
              "session_id": session_id,
              "response": "⚠️ Por favor, proporciona una respuesta válida."
            })

    # ── Cuando terminamos el cuestionario o estamos en modo admin, generamos el informe ─
    if step > len(questions) or is_admin:
        # Preparamos el informe clínico...
        info = "\n".join(f"{i+1}. {q}\n→ {a}"
                         for i, (q,a) in enumerate(zip(questions.values(), session_data[session_id])))
        analysis_prompt = (
            f"📝 **Informe Clínico Detallado**\n\n📌 Datos Recopilados:\n{info}\n\n"
            "🔍 **Análisis Clínico**\n"
            "Interpreta esta información desde el punto de vista médico y sugiere hipótesis diagnósticas "
            "posibles con base en evidencia científica, factores de riesgo y presentación del caso. "
            "Finaliza con recomendaciones para el médico tratante."
        )
        parts = [
            {"role":"system","parts":[SYSTEM_PROMPT]},
            {"role":"user","parts":[analysis_prompt]}
        ]
        try:
            ai_response = get_cached_response(tuple(map(str, parts)))
            return jsonify({"session_id": session_id, "response": ai_response})
        except Exception as e:
            logger.error("Error en /api/chat:", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Por defecto, devolvemos la siguiente pregunta
    return jsonify({"session_id": session_id, "response": questions[step]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
