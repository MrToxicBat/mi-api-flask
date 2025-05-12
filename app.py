import logging
import os
import base64
import uuid

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
import google.generativeai as genai

# ─── Configuración básica ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.update({
    "SECRET_KEY": os.getenv("SECRET_KEY", str(uuid.uuid4())),
    "SESSION_TYPE": "filesystem",
    "SESSION_FILE_DIR": "./.flask_session/",
    "SESSION_PERMANENT": False,
    "SESSION_COOKIE_NAME": "session"
})
# Parche para Flask-Session en versiones recientes de Flask
app.session_cookie_name = app.config["SESSION_COOKIE_NAME"]
Session(app)
CORS(app)

# ─── Preguntas del cuestionario ─────────────────────────────────────────────────
questions = [
    "Nombre completo del paciente:",
    "Edad del paciente:",
    "Sexo asignado al nacer y género actual:",
    "Motivo principal de consulta:",
    "¿Desde cuándo presenta estos síntomas? ¿Han cambiado con el tiempo?",
    "Antecedentes médicos personales (crónicos, quirúrgicos, etc.):",
    "Medicamentos actuales (nombre, dosis, frecuencia):",
    "Alergias conocidas (medicamentos, alimentos, etc.):",
    "Antecedentes familiares relevantes:",
    "Estudios diagnósticos realizados y resultados si se conocen:"
]

SYSTEM_PROMPT = """
Eres un asistente médico inteligente cuyo objetivo es recopilar información clínica básica
de forma estructurada y ordenada, y luego generar un informe clínico con posibles
hipótesis diagnósticas basadas en evidencia científica.
"""

# ─── Endpoint /api/chat ─────────────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_message = data.get('message', '').strip()
    image_data = data.get('image')

    # 1) Inicializar la sesión si es nueva
    if 'step' not in session:
        session['step'] = 0  # Empezamos en 0 para coincidir con el índice de la lista
        session['data'] = {}
        session['admin'] = False
        logger.info("Nueva sesión iniciada")
        return jsonify(response="👋 ¡Hola! Soy tu asistente médico inteligente. Vamos a recopilar los datos del paciente.\n\n" + questions[0])

    current_step = session['step']
    is_admin = session.get('admin', False)
    logger.info(f"Paso actual: {current_step}, Admin: {is_admin}, Mensaje: {user_message}")

    # 2) Modo admin
    if user_message.lower() == "admin":
        session['admin'] = True
        return jsonify(response="🔓 Modo Admin activado. Puedes escribir libremente.")

    # 3) Procesar respuesta del usuario
    if user_message:
        # Guardar la respuesta en el paso actual
        session['data'][questions[current_step]] = user_message
        
        # Avanzar al siguiente paso si no estamos en modo admin y hay más preguntas
        if not is_admin and current_step < len(questions) - 1:
            session['step'] = current_step + 1
            next_question = questions[current_step + 1]
            return jsonify(response=next_question)
        else:
            session['step'] = len(questions)  # Marcamos como completado

    # 4) Generar informe cuando esté completo o en modo admin
    if session['step'] >= len(questions) or is_admin:
        # Validar datos mínimos
        required_questions = questions[:3]
        if not is_admin and not all(q in session['data'] for q in required_questions):
            missing = [q for q in required_questions if q not in session['data']]
            return jsonify(response=f"⚠️ Faltan datos obligatorios: {', '.join(missing)}")

        # Preparar prompt para el modelo
        collected_data = "\n".join(f"• {q}\n  → {a}" for q, a in session['data'].items())
        prompt = f"""
        Datos del paciente:
        {collected_data}

        Por favor genera:
        1. Un resumen clínico conciso
        2. Hipótesis diagnósticas basadas en evidencia
        3. Recomendaciones para evaluación adicional
        """

        # Procesar imagen si existe
        parts = [{"text": SYSTEM_PROMPT + prompt}]
        if image_data:
            try:
                img_bytes = base64.b64decode(image_data.split(',')[-1])
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_bytes
                    }
                })
            except Exception as e:
                logger.error(f"Error procesando imagen: {str(e)}")

        # Generar respuesta con el modelo
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(parts)
            analysis = response.text.strip() or "No se pudo generar el análisis."
            return jsonify(response=analysis)
        except Exception as e:
            logger.error(f"Error en el modelo: {str(e)}")
            return jsonify(response="❌ Error al procesar la solicitud. Intente nuevamente."), 500

    # 5) Continuar con la siguiente pregunta
    return jsonify(response=questions[session['step']])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
