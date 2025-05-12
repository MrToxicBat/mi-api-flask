import logging
import os
import base64
import uuid
import re

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
    "SESSION_TYPE": "filesystem",       # Persiste en archivos dentro de tu contenedor
    "SESSION_FILE_DIR": "./.flask_session/",
    "SESSION_PERMANENT": False,
    "SESSION_COOKIE_NAME": "session"    # Nombre de la cookie de sesión
})

# ── Parche para que Flask-Session no falle en Flask recientes ───────────────────
# Flask-Session busca `app.session_cookie_name`, que ya no existe por defecto.
app.session_cookie_name = app.config.get("SESSION_COOKIE_NAME", "session")

Session(app)
CORS(app)

# ─── Cuestionario ───────────────────────────────────────────────────────────────
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

SYSTEM_PROMPT = """Eres un asistente médico inteligente cuyo objetivo es recopilar información clínica básica de forma estructurada y ordenada.

Instrucciones:
1. Formula **solo una vez** cada una de las siguientes preguntas en este orden:
   1) Edad del paciente  
   2) Sexo asignado al nacer y género actual  
   3) Motivo principal de consulta  
   4) ¿Desde cuándo presenta estos síntomas?  
   5) Antecedentes médicos personales  
   6) Medicamentos actuales  
   7) Alergias conocidas  
   8) Antecedentes familiares relevantes  
   9) Estudios diagnósticos realizados  

2. No repitas una pregunta que ya tenga respuesta válida.  
3. Si la respuesta no cumple el formato mínimo (p. ej. edad sin números, sexo sin identificador, etc.), pide aclaración de esa misma pregunta.  
4. Solo tras haber recibido respuestas válidas a las 9 preguntas, genera un **Informe Clínico** con análisis y posibles diagnósticos diferenciales basados en evidencia científica.

Mantén un tono profesional y claro, y recuerda que tu función es asistir, no sustituir al médico."""

# ─── Endpoint /api/chat ─────────────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_message = data.get('message', '').strip()
    image_data   = data.get('image')

    # Inicializar la sesión si es la primera vez
    if 'step' not in session:
        session['step'] = 1
        session['data'] = []
        session['admin'] = False
        logger.info("Sesión nueva iniciada en Flask-Session.")
        return jsonify(response=questions[1])

    step     = session['step']
    is_admin = session.get('admin', False)
    logger.info(f"[session] step={step}, admin={is_admin}, msg={user_message!r}")

    # Comando para activar modo admin
    if user_message.lower() == "admin":
        session['admin'] = True
        return jsonify(response="🔓 Modo Admin activado. Ahora puedes escribir libremente o subir imágenes.")

    # Si reciben solo imagen, repiten la misma pregunta
    if image_data and not user_message:
        return jsonify(response=questions.get(step, questions[1]))

    # Validación mínima de respuestas
    def is_valid(text):
        if is_admin:
            return True
        if not text:
            return False
        if step == 0:
             bool(re.search(r'\d{1,3}', text))
        if step == 2:
            return any(g in text.lower() for g in ["masculino","femenino","m","f","hombre","mujer"])
        return True

    # Procesar respuesta textual
    if user_message:
        if not is_valid(user_message):
            return jsonify(response="⚠️ Por favor, proporcione una respuesta válida para avanzar.")
        # Guardar o actualizar
        data_list = session['data']
        if len(data_list) < step:
            data_list.append(user_message)
        else:
            data_list[step-1] = user_message
        session['data'] = data_list

        # Avanzar paso
        if step < len(questions):
            session['step'] = step + 1
            return jsonify(response=questions[step+1])
        else:
            # Marca fin de cuestionario
            session['step'] = len(questions) + 1

    # Si completó o está en admin, generar análisis
    if session['step'] > len(questions) or is_admin:
        respuestas = {
            questions[i+1]: ans
            for i, ans in enumerate(session['data'])
        }

        # Validar datos esenciales
        if (not is_admin and (
            not respuestas.get(questions[1]) or
            not respuestas.get(questions[2]) or
            not respuestas.get(questions[3])
        )):
            return jsonify(response="⚠️ Necesito edad, sexo y motivo de consulta antes de continuar.")

        # Construir prompt
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

        # Adjuntar imagen si la hay
        if image_data:
            try:
                img_bytes = base64.b64decode(image_data.split(',')[-1])
                parts[1]["parts"].append({
                    "inline_data": {"mime_type": "image/png", "data": img_bytes}
                })
            except Exception:
                logger.warning("No se pudo decodificar imagen.", exc_info=True)

        # Llamar a Gemini
        try:
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            resp  = model.generate_content(parts)
            text = getattr(resp, 'text', '').strip()
            if not text:
                text = "🤖 **Análisis preliminar**: no hubo respuesta del modelo."
            return jsonify(response=text)
        except Exception:
            logger.error("Error generando análisis:", exc_info=True)
            return jsonify(response="❌ Error interno al generar análisis."), 500

    # Fallback: repetir la pregunta actual
    return jsonify(response=questions.get(step, questions[1]))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
