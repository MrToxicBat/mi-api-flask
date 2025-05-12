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
CORS(app, supports_credentials=True, origins=[os.getenv("FRONTEND_ORIGIN","*")])

# ——— Inicializar Gemini ———
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ——— Detectar dinámicamente un modelo multimodal válido ———
MODEL_NAME = None
try:
    all_models = list(genai.list_models())  # ← convertimos el generator en lista
    for m in all_models:
        if "generateContent" in m.supported_methods:
            MODEL_NAME = m.name
            logger.info(f"Usando modelo multimodal: {MODEL_NAME}")
            break
    if MODEL_NAME is None:
        raise RuntimeError("Ningún modelo soporta generateContent")
except Exception:
    # Si algo falla, caemos en fallback de texto
    MODEL_NAME = "models/gemini-2.0-preview"
    logger.info(f"Usando modelo de fallback: {MODEL_NAME}")

# ——— Campos a recopilar ———
required_fields = [
    "motivo_principal",
    "duracion_sintomas",
    "intensidad",
    "edad",
    "sexo",
    "antecedentes_medicos",
]

field_prompts = {
    "motivo_principal":   "👋 Hola, doctor/a. ¿Cuál es el motivo principal de consulta de este paciente?",
    "duracion_sintomas":   "Gracias. Me dice que es «{motivo_principal}». ¿Cuánto tiempo lleva con esos síntomas?",
    "intensidad":          "Entendido. ¿Qué tan severos son (leve, moderado, severo)?",
    "edad":                "Perfecto. ¿Qué edad tiene el paciente?",
    "sexo":                "Bien. ¿Sexo asignado al nacer y género actual?",
    "antecedentes_medicos":"¿Antecedentes médicos relevantes (enfermedades previas, cirugías, alergias, medicación)?",
}

def get_system_instruction():
    return (
        "Eres una IA médica multimodal. Puedes procesar texto e imágenes médicas para análisis. "
        "Solo respondes diagnósticos y recomendaciones basadas en datos clínicos e imágenes. "
        "Si recibes otra cosa, di: 'Lo siento, solo proceso información médica.'"
    )

session_data = {}

@app.route('/api/list-models', methods=['GET'])
def list_models():
    all_models = list(genai.list_models())
    return jsonify({m.name: m.supported_methods for m in all_models})

@app.route('/api/chat', methods=['POST'])
def chat():
    logger.info("→ Llega /api/chat")
    data       = request.json or {}
    user_text  = data.get('message','').strip()
    image_b64  = data.get('image')
    image_type = data.get('image_type')

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

    if image_b64 and not user_text:
        inputs.append({"image": {"data": image_b64, "mime_type": image_type}})
        inputs.append({"text": "Por favor, analiza esta imagen médica."})
    else:
        if user_text and step < len(required_fields):
            campo = required_fields[step]
            collected[campo] = user_text
            logger.info(f"Sesión {sid}: guardado {campo} = {user_text!r}")
            step += 1
            flask_session['step'] = step

        if step < len(required_fields):
            siguiente = required_fields[step]
            prompt = field_prompts[siguiente].format(**collected)
            inputs.append({"text": get_system_instruction()})
            inputs.append({"text": prompt})
        else:
            info_lines = "\n".join(f"- {k}: {v}" for k, v in collected.items())
            final_prompt = (
                "Gracias por la información. Con estos datos, analiza en profundidad los hallazgos "
                "y sugiere diagnósticos, hipótesis y recomendaciones.\n\n"
                f"Información del paciente:\n{info_lines}"
            )
            inputs.append({"text": get_system_instruction()})
            inputs.append({"text": final_prompt})
            session_data.pop(sid, None)
            flask_session.pop('session_id', None)
            flask_session.pop('step', None)
            logger.info(f"Sesión {sid} completada")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        logger.info(f"Llamando generate_content con modelo {MODEL_NAME}")
        resp = model.generate_content(inputs)
        ai_text = getattr(resp, "text", "").strip()
        return jsonify({"response": ai_text})
    except Exception as e:
        logger.error("Error en /api/chat", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
