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

# ——— CORS: permitir tu dominio y localhost ———
CORS(
    app,
    resources={r"/api/*": {
        "origins": [
            "https://code-solution.com",
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ]
    }},
    supports_credentials=True
)

# ——— Inicializar Gemini ———
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "models/gemini-2.0-flash"

# ——— Flujo clínico paso a paso ———
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
        "Primero analiza cualquier imagen médica que te envíen. "
        "Solo después, recopila datos clínicos paso a paso y al final sugiere diagnósticos y recomendaciones."
    )

def build_summary(collected: dict) -> str:
    if not collected:
        return ""
    lines = [
        f"- **{k.replace('_',' ').capitalize()}**: {v}"
        for k, v in collected.items()
    ]
    return "Información recopilada hasta ahora:\n" + "\n".join(lines) + "\n\n"

# ——— Estado en memoria (sólo para testing) ———
session_data = {}

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    data = request.get_json() or {}
    img_b64 = data.get('image')
    logger.info(f"/api/analyze-image received. image length: {len(img_b64) if img_b64 else 'None'}")

    if not img_b64:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # decodificamos a bytes
        img_bytes = base64.b64decode(img_b64)
        logger.info(f"Decoded image bytes: {len(img_bytes)} bytes")

        # llamamos a Gemini con bytes
        resp = genai.annotate_image(
            model="models/gemini-image-alpha",
            image=img_bytes,
            supports=["TEXT"]
        )

        description = ""
        if resp and getattr(resp, "annotations", None):
            description = resp.annotations[0].text
            logger.info(f"Annotation result: {description[:100]}…")

        return jsonify({'description': description}), 200

    except Exception as e:
        logger.exception("Error procesando /api/analyze-image")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    data       = request.get_json() or {}
    user_text  = data.get('message', '').strip()
    image_b64  = data.get('image')
    image_type = data.get('image_type')

    # inicializar sesión
    sid = flask_session.get('session_id')
    if not sid or sid not in session_data:
        sid = str(uuid.uuid4())
        flask_session['session_id'] = sid
        flask_session['step'] = 0
        session_data[sid] = {"fields": {}, "image_analyzed": False}
        logger.info(f"Nueva sesión: {sid}")

    step       = flask_session.get('step', 0)
    state      = session_data[sid]
    collected  = state["fields"]
    image_done = state["image_analyzed"]

    parts = []

    # caso imagen multimodal
    if image_b64 and not image_done:
        img_bytes = base64.b64decode(image_b64)
        parts.append({"mime_type": image_type, "data": img_bytes})
        prompt_text = "Por favor, analiza esta imagen médica y describe hallazgos relevantes."
        state["image_analyzed"] = True
    else:
        # guardamos texto
        if user_text and step < len(required_fields):
            campo = required_fields[step]
            collected[campo] = user_text
            step += 1
            flask_session['step'] = step
            logger.info(f"{sid}: guardado {campo} = {user_text!r}")

        # siguiente prompt o cierre
        if step < len(required_fields):
            siguiente = required_fields[step]
            question = field_prompts[siguiente].format(**collected)
            summary  = build_summary(collected)
            prompt_text = summary + question
        else:
            info = "\n".join(f"- {k}: {v}" for k, v in collected.items())
            prompt_text = (
                "Gracias por toda la información. Con estos datos, analiza en profundidad "
                "los hallazgos y sugiere diagnósticos, hipótesis y recomendaciones.\n\n"
                f"Información recopilada:\n{info}"
            )
            session_data.pop(sid, None)
            flask_session.pop('session_id', None)
            flask_session.pop('step', None)
            logger.info(f"Sesión {sid} completada")

    full_prompt = f"{get_system_instruction()}\n\n{prompt_text}"
    parts.append({"text": full_prompt})

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp  = model.generate_content({"parts": parts})
        ai_text = getattr(resp, "text", "").strip()
        return jsonify({"response": ai_text}), 200

    except Exception as e:
        logger.exception("Error en /api/chat")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
