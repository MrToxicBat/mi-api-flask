import os
import uuid
import logging
import base64
from flask import Flask, request, jsonify, session as flask_session
from flask_cors import CORS
import google.generativeai as genai

# ——— Configuración básica ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# Solo tu dominio y cookies
CORS(app,
     supports_credentials=True,
     origins=[os.getenv("FRONTEND_ORIGIN", "https://code-soluction.com")])

# ——— Inicializar Gemini ———
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Modelo multimodal válido (detectado con /api/list-models)
MODEL_NAME = "models/gemini-2.0-flash"

# ——— Flujo de preguntas ———
required_fields = [
    "motivo_principal",
    "duracion_sintomas",
    "intensidad",
    "edad",
    "sexo",
    "antecedentes_medicos",
]
field_prompts = {
    "motivo_principal":
        "👋 Hola, doctor/a. ¿Cuál considera usted que es el motivo principal de consulta de este paciente?",
    "duracion_sintomas":
        "Gracias. Me dice que el motivo es “{motivo_principal}”. ¿Cuánto tiempo lleva con esos síntomas?",
    "intensidad":
        "Entendido. ¿Qué tan severos son esos síntomas (leve, moderado, severo)?",
    "edad":
        "Perfecto. ¿Qué edad tiene el paciente?",
    "sexo":
        "Bien. ¿Cuál es el sexo asignado al nacer y el género actual?",
    "antecedentes_medicos":
        "¿Antecedentes médicos relevantes (enfermedades previas, cirugías, alergias, medicación)?",
}

def get_system_instruction():
    return (
        "Eres una IA médica multimodal. "
        "Recopila primero estos datos paso a paso: motivo principal, duración de síntomas, intensidad, edad, sexo y antecedentes. "
        "Cuando estén completos, analiza la información clínica (y cualquier imagen médica) y sugiere posibles diagnósticos y recomendaciones."
    )

# Sesiones en memoria
session_data = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    data       = request.json or {}
    user_text  = data.get('message', '').strip()
    image_b64  = data.get('image')       # Base64 puro
    image_type = data.get('image_type')  # p.ej. "image/png"

    # — Sesión y paso —
    sid = flask_session.get('session_id')
    if not sid or sid not in session_data:
        sid = str(uuid.uuid4())
        flask_session['session_id'] = sid
        flask_session['step'] = 0
        session_data[sid] = {}
        logger.info(f"Nueva sesión {sid}")

    step      = flask_session['step']
    collected = session_data[sid]

    # — Decide el texto del prompt según el flujo —
    if image_b64 and not user_text:
        # Análisis multimodal inmediato
        prompt = "Por favor, analiza esta imagen médica."
    else:
        # Primera parte: guardamos la respuesta si falta campo
        if user_text and step < len(required_fields):
            campo = required_fields[step]
            collected[campo] = user_text
            logger.info(f"Sesión {sid}: guardado {campo} = {user_text!r}")
            step += 1
            flask_session['step'] = step

        # Si aún faltan, pedimos el siguiente
        if step < len(required_fields):
            siguiente = required_fields[step]
            prompt = field_prompts[siguiente].format(**collected)
        else:
            # Todos respondidos → análisis final
            info = "\n".join(f"- {k}: {v}" for k, v in collected.items())
            prompt = (
                "Gracias por toda la información. Con estos datos, analiza en profundidad los hallazgos "
                "y sugiere diagnósticos, hipótesis y recomendaciones.\n\n"
                f"Información recopilada:\n{info}"
            )
            # Limpiar para nueva sesión
            session_data.pop(sid, None)
            flask_session.pop('session_id', None)
            flask_session.pop('step', None)
            logger.info(f"Sesión {sid} completada")

    # — Construir único full_prompt con instrucción + pregunta —
    full_prompt = f"{get_system_instruction()}\n\n{prompt}"

    # — Montamos inputs: imagen primero (si la hay), luego texto —
    inputs = []
    if image_b64 and not user_text:
        inputs.append({"image": {"data": image_b64, "mime_type": image_type}})
    inputs.append({"text": full_prompt})

    # — Llamada multimodal —
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp  = model.generate_content(inputs)
        ai_text = getattr(resp, "text", "").strip()
        return jsonify({"response": ai_text})
    except Exception as e:
        logger.error("Error en /api/chat", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
