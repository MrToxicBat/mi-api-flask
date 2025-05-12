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
# Ajusta aquí tu dominio real
CORS(app, supports_credentials=True, origins=[os.getenv("FRONTEND_ORIGIN", "https://code-soluction.com")])

# ——— Inicializar Gemini ———
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ——— Modelo multimodal válido (lo encontraste en /api/list-models) ———
MODEL_NAME = "models/gemini-2.0-flash"

# ——— Flujo de recopilación ———
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
        "¿El paciente tiene antecedentes médicos relevantes (enfermedades previas, cirugías, alergias, medicación)?",
}

def get_system_instruction():
    return (
        "Eres una IA médica multimodal. "
        "Primero recopila información clínica básica paso a paso: "
        "motivo principal, duración de síntomas, intensidad, edad, sexo y antecedentes. "
        "Una vez completos esos datos, analiza la información (y cualquier imagen médica) "
        "y sugiere posibles diagnósticos y recomendaciones. "
        "Si recibes algo que no sea información médica, responde: "
        "'Lo siento, solo proceso información médica.'"
    )

# ——— Almacén de sesiones en memoria ———
session_data = {}

# ——— Endpoint de prueba para listar modelos ———
@app.route('/api/list-models', methods=['GET'])
def list_models():
    all_models = list(genai.list_models())
    return jsonify({m.name: m.supported_methods for m in all_models})

# ——— Endpoint de chat ———
@app.route('/api/chat', methods=['POST'])
def chat():
    logger.info("→ /api/chat recibida")
    data       = request.json or {}
    user_text  = data.get('message', '').strip()
    image_b64  = data.get('image')       # Base64 puro
    image_type = data.get('image_type')  # e.g. "image/png"

    # — Sesión —
    sid = flask_session.get('session_id')
    if not sid or sid not in session_data:
        sid = str(uuid.uuid4())
        flask_session['session_id'] = sid
        flask_session['step'] = 0
        session_data[sid] = {}
        logger.info(f"Nueva sesión {sid}")

    step      = flask_session['step']
    collected = session_data[sid]
    inputs    = []

    # — Si llega solo la imagen, la analizamos directamente —
    if image_b64 and not user_text:
        prompt = "Por favor, analiza esta imagen médica."
        full_text = f"{get_system_instruction()}\n\n{prompt}"
        inputs.append({"image": {"data": image_b64, "mime_type": image_type}})
        inputs.append({"text": full_text})

    else:
        # — Recopilación paso a paso —
        if user_text and step < len(required_fields):
            campo = required_fields[step]
            collected[campo] = user_text
            logger.info(f"Sesión {sid}: guardado {campo} = {user_text!r}")
            step += 1
            flask_session['step'] = step

        if step < len(required_fields):
            siguiente = required_fields[step]
            prompt = field_prompts[siguiente].format(**collected)
            full_text = f"{get_system_instruction()}\n\n{prompt}"
            inputs.append({"text": full_text})
        else:
            # — Análisis final con todo lo recogido —
            info_lines = "\n".join(f"- {k}: {v}" for k, v in collected.items())
            prompt = (
                "Gracias por la información. Con estos datos, analiza en profundidad los hallazgos "
                "y sugiere posibles diagnósticos y recomendaciones.\n\n"
                f"Información del paciente:\n{info_lines}"
            )
            full_text = f"{get_system_instruction()}\n\n{prompt}"
            inputs.append({"text": full_text})

            # Limpiar sesión
            session_data.pop(sid, None)
            flask_session.pop('session_id', None)
            flask_session.pop('step', None)
            logger.info(f"Sesión {sid} completada")

    # — Llamada multimodal —
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        logger.info(f"Generando con modelo {MODEL_NAME} y inputs: {inputs}")
        resp = model.generate_content(inputs)
        ai_text = getattr(resp, "text", "").strip()
        logger.info(f"→ Respuesta IA: {ai_text!r}")
        return jsonify({"response": ai_text})
    except Exception as e:
        logger.error("Error en /api/chat", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
