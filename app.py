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

# CORS para tu frontend
CORS(app,
     supports_credentials=True,
     origins=[os.getenv("FRONTEND_ORIGIN", "https://code-soluction.com")])

# ——— Inicializar Gemini ———
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Modelo multimodal válido
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
        "Primero analiza cualquier imagen médica que te envíen. "
        "Solo después, recopila datos clínicos paso a paso y al final sugiere diagnósticos."
    )

def build_summary(collected: dict) -> str:
    if not collected:
        return ""
    lines = [f"- **{k.replace('_',' ').capitalize()}**: {v}"
             for k, v in collected.items()]
    return "Información recopilada hasta ahora:\n" + "\n".join(lines) + "\n\n"

# ——— Almacén en memoria con estructura avanzada ———
# session_data[sid] = {
#   "fields": { ... },
#   "image_analyzed": bool
# }
session_data = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    data       = request.json or {}
    user_text  = data.get('message', '').strip()
    image_b64  = data.get('image')
    image_type = data.get('image_type')

    # — Inicializar sesión si es nueva —
    sid = flask_session.get('session_id')
    if not sid or sid not in session_data:
        sid = str(uuid.uuid4())
        flask_session['session_id'] = sid
        flask_session['step'] = 0
        session_data[sid] = {
            "fields": {},
            "image_analyzed": False
        }
        logger.info(f"Nueva sesión {sid}")

    step       = flask_session['step']
    state      = session_data[sid]
    collected  = state["fields"]
    image_done = state["image_analyzed"]

    # — Caso 1: imagen nueva y aún no la analizamos —
    if image_b64 and not image_done:
        # Vamos a analizar la imagen primero
        full_prompt = (
            f"{get_system_instruction()}\n\n"
            "Por favor, analiza esta imagen médica y describe hallazgos relevantes."
        )
        inputs = [
            {"image": {"data": image_b64, "mime_type": image_type}},
            {"text": full_prompt}
        ]

        # Marcamos la imagen como ya analizada
        state["image_analyzed"] = True

        try:
            model = genai.GenerativeModel(MODEL_NAME)
            resp  = model.generate_content(inputs)
            ai_text = getattr(resp, "text", "").strip()
            return jsonify({"response": ai_text})
        except Exception as e:
            logger.error("Error analizando imagen", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # — Caso 2: después de imagen (o sin imagen) → flujo de preguntas —

    # Si el usuario envía texto y faltan campos, lo guardamos
    if user_text and step < len(required_fields):
        campo = required_fields[step]
        collected[campo] = user_text
        logger.info(f"Sesión {sid}: guardado {campo} = {user_text!r}")
        step += 1
        flask_session['step'] = step

    # Construir el siguiente prompt
    if step < len(required_fields):
        # Pedimos el siguiente dato
        siguiente   = required_fields[step]
        question    = field_prompts[siguiente].format(**collected)
        summary_txt = build_summary(collected)
        full_prompt = (
            f"{get_system_instruction()}\n\n"
            f"{summary_txt}{question}"
        )
    else:
        # Ya recogimos todo: hacemos análisis final
        info_lines = "\n".join(f"- {k}: {v}" for k, v in collected.items())
        conclusion = (
            "Gracias por toda la información. Con estos datos, analiza en profundidad los hallazgos "
            "y sugiere diagnósticos, hipótesis y recomendaciones.\n\n"
            f"Información recopilada:\n{info_lines}"
        )
        full_prompt = f"{get_system_instruction()}\n\n{conclusion}"
        # Limpiar sesión para nueva conversación
        session_data.pop(sid, None)
        flask_session.pop('session_id', None)
        flask_session.pop('step', None)
        logger.info(f"Sesión {sid} completada")

    # Enviamos al modelo
    inputs = [{"text": full_prompt}]
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
