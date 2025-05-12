import logging
import os
import base64
import uuid
import re

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
import google.generativeai as genai

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
app.session_cookie_name = app.config["SESSION_COOKIE_NAME"]
Session(app)
CORS(app)

questions = {
    1: "Edad del paciente:",
    2: "Sexo asignado al nacer y género actual:",
    3: "Motivo principal de consulta:",
    4: "¿Desde cuándo presenta estos síntomas? ¿Han cambiado con el tiempo?",
    5: "Antecedentes médicos personales (crónicos, quirúrgicos, etc.):",
    6: "Medicamentos actuales (nombre, dosis, frecuencia):",
    7: "Alergias conocidas (medicamentos, alimentos, etc.):",
    8: "Antecedentes familiares relevantes:",
    9: "Estudios diagnósticos realizados y resultados si se conocen:"
}

SYSTEM_PROMPT = """Eres un asistente médico inteligente cuyo objetivo es recopilar información clínica básica..."""

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    # Asegúrate de leer la clave que tú envíes: 'message' o 'prompt'
    user_message = data.get('message', data.get('prompt', '')).strip()
    image_data   = data.get('image')

    # Inicializar la sesión la primera vez
    if 'step' not in session:
        session['step'] = 1
        session['data'] = []
        session['admin'] = False
        logger.info("Sesión nueva iniciada.")
        if not user_message and not image_data:
            return jsonify(response=questions[1])
        # si ya mandaron algo, seguimos al flujo normal

    step     = session['step']
    is_admin = session.get('admin', False)
    logger.info(f"[session] step={step}, admin={is_admin}, msg={user_message!r}")

    # Activar modo admin
    if user_message.lower() == "admin":
        session['admin'] = True
        return jsonify(response="🔓 Modo Admin activado. Ahora puedes escribir libremente o subir imágenes.")

    # Si solo envían imagen sin texto, repetir la pregunta actual
    if image_data and not user_message:
        return jsonify(response=questions.get(step, questions[1]))

    # Validación mínima de respuestas
    def is_valid(text):
        if is_admin:
            return True
        if not text:
            return False
        if step == 1:
            return bool(re.search(r'\d{1,3}', text))
        if step == 2:
            return any(g in text.lower() for g in ["masculino","femenino","m","f","hombre","mujer"])
        return True

    # Procesar mensaje de texto
    if user_message:
        if not is_valid(user_message):
            return jsonify(response="⚠️ Por favor, proporcione una respuesta válida para avanzar.")
        # Guardar la respuesta
        data_list = session['data']
        if len(data_list) < step:
            data_list.append(user_message)
        else:
            data_list[step-1] = user_message
        session['data'] = data_list

        # Avanzar al siguiente paso
        if step < len(questions):
            session['step'] = step + 1
            return jsonify(response=questions[step+1])
        else:
            session['step'] = len(questions) + 1

    # Una vez completado (o en admin), generar el informe
    if session['step'] > len(questions) or is_admin:
        respuestas = {questions[i+1]: ans for i, ans in enumerate(session['data'])}
        if (not is_admin and (
            not respuestas.get(questions[1]) or
            not respuestas.get(questions[2]) or
            not respuestas.get(questions[3])
        )):
            return jsonify(response="⚠️ Necesito edad, sexo y motivo de consulta antes de continuar.")

        info = "\n".join(f"{idx}. {q}\n→ {a}"
                         for idx, (q,a) in enumerate(respuestas.items(), start=1))
        analysis_prompt = (
            "📝 **Informe Clínico**\n\n"
            + info +
            "\n\n🔍 **Análisis**\nInterpreta estos datos y sugiere hipótesis diagnósticas con base en evidencia."
        )
        parts = [
            {"role": "system", "parts": [SYSTEM_PROMPT]},
            {"role": "user",   "parts": [analysis_prompt]}
        ]
        if image_data:
            try:
                img_bytes = base64.b64decode(image_data.split(',')[-1])
                parts[1]["parts"].append({
                    "inline_data": {"mime_type": "image/png", "data": img_bytes}
                })
            except Exception:
                logger.warning("No se pudo decodificar imagen.", exc_info=True)
        try:
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            resp  = model.generate_content(parts)
            text = getattr(resp, 'text', '').strip() or "🤖 **Análisis preliminar**: no hubo respuesta del modelo."
            return jsonify(response=text)
        except Exception:
            logger.error("Error generando análisis:", exc_info=True)
            return jsonify(response="❌ Error interno al generar análisis."), 500

    # Fallback: repetir la pregunta actual
    return jsonify(response=questions.get(step, questions[1]))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
