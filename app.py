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
questions = {
    1: "Nombre completo del paciente:",
    2: "Edad del paciente:",
    3: "Sexo asignado al nacer y género actual:",
    4: "Motivo principal de consulta:",
    5: "¿Desde cuándo presenta estos síntomas? ¿Han cambiado con el tiempo?",
    6: "Antecedentes médicos personales (crónicos, quirúrgicos, etc.):",
    7: "Medicamentos actuales (nombre, dosis, frecuencia):",
    8: "Alergias conocidas (medicamentos, alimentos, etc.):",
    9: "Antecedentes familiares relevantes:",
    10: "Estudios diagnósticos realizados y resultados si se conocen:"
}

SYSTEM_PROMPT = """
Eres un asistente médico inteligente cuyo objetivo es recopilar información clínica básica
de forma estructurada y ordenada, y luego generar un informe clínico con posibles
hipótesis diagnósticas basadas en evidencia científica.
"""

# ─── Endpoint /api/chat ─────────────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    # Asegúrate de que tu cliente envíe el texto bajo "message" o "prompt"
    user_message = data.get('message', data.get('prompt', '')).strip()
    image_data   = data.get('image')

    # 1) Inicializar la sesión la primera vez
    if 'step' not in session:
        session['step'] = 1
        session['data'] = []
        session['admin'] = False
        logger.info("Sesión nueva iniciada.")
        greeting = (
            "👋 ¡Hola! Soy tu asistente médico inteligente. "
            "Te ayudaré a recopilar datos del paciente paso a paso. "
            "Para comenzar, ¿puedes decirme el nombre completo del paciente?"
        )
        if not user_message and not image_data:
            return jsonify(response=greeting)

    step     = session['step']
    is_admin = session.get('admin', False)
    logger.info(f"[session] step={step}, admin={is_admin}, msg={user_message!r}")

    # 2) Comando para activar modo admin
    if user_message.lower() == "admin":
        session['admin'] = True
        return jsonify(response="🔓 Modo Admin activado. Ahora puedes escribir libremente o subir imágenes.")

    # 3) Si envían solo imagen, repetimos la misma pregunta
    if image_data and not user_message:
        return jsonify(response=questions.get(step, questions[1]))

    # 4) Validación sencilla: aceptamos cualquier texto no vacío (salvo modo admin)
    def is_valid(text):
        if is_admin:
            return True
        return bool(text and text.strip())

    # 5) Procesar respuesta de texto
    if user_message:
        if not is_valid(user_message):
            return jsonify(response="😅 Ups, no entendí eso. ¿Podrías escribirlo de otra forma o con más detalle?")
        # Guardar o actualizar la respuesta en la sesión
        data_list = session['data']
        if len(data_list) < step:
            data_list.append(user_message)
        else:
            data_list[step-1] = user_message
        session['data'] = data_list

        # Avanzar al siguiente paso o marcar finalizado
        if step < len(questions):
            session['step'] = step + 1
            return jsonify(response=questions[step+1])
        else:
            session['step'] = len(questions) + 1

    # 6) Una vez respondidas todas (o si es admin), generamos el informe
    if session['step'] > len(questions) or is_admin:
        respuestas = {questions[i+1]: ans for i, ans in enumerate(session['data'])}

        # Validar que al menos nombre, edad y sexo estén presentes
        if (not is_admin and (
            not respuestas.get(questions[1]) or
            not respuestas.get(questions[2]) or
            not respuestas.get(questions[3])
        )):
            return jsonify(response="⚠️ Antes de continuar, necesito al menos el nombre, la edad y el sexo del paciente.")

        # Construir prompt para Gemini
        info = "\n".join(f"{idx}. {q}\n→ {a}"
                         for idx, (q, a) in enumerate(respuestas.items(), start=1))
        analysis_prompt = (
            "📝 **Informe Clínico**\n\n" +
            info +
            "\n\n🔍 **Análisis**\nInterpreta estos datos y sugiere posibles diagnósticos basados en evidencia."
        )

        parts = [
            {"role": "system", "parts": [SYSTEM_PROMPT]},
            {"role": "user",   "parts": [analysis_prompt]}
        ]

        # Adjuntar imagen si existe
        if image_data:
            try:
                img_bytes = base64.b64decode(image_data.split(',')[-1])
                parts[1]["parts"].append({
                    "inline_data": {"mime_type": "image/png", "data": img_bytes}
                })
            except Exception:
                logger.warning("No se pudo decodificar imagen.", exc_info=True)

        # Llamada a Gemini
        try:
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            resp  = model.generate_content(parts)
            text  = getattr(resp, 'text', '').strip() or "🤖 **Análisis preliminar**: no hubo respuesta del modelo."
            return jsonify(response=text)
        except Exception:
            logger.error("Error generando análisis:", exc_info=True)
            return jsonify(response="❌ Error interno al generar análisis."), 500

    # 7) Fallback: repetir la pregunta actual
    return jsonify(response=questions.get(step, questions[1]))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
