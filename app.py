import os
import uuid
import logging
import base64
from flask import Flask, request, jsonify, make_response, session as flask_session
from flask_cors import CORS
import google.generativeai as genai

# ——— Configuración básica ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# Habilitar CORS para todas las rutas /api/*
CORS(app,
     supports_credentials=True,
     resources={r"/api/*": {"origins": ["https://code-soluction.com"]}})

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
        "⏳ Gracias. Me dice que el motivo es “{motivo_principal}”. ¿Cuánto tiempo lleva con esos síntomas?",
    "intensidad":
        "⚖️ Entendido. ¿Qué tan severos son esos síntomas (leve, moderado, severo)?",
    "edad":
        "🎂 Perfecto. ¿Qué edad tiene el paciente?",
    "sexo":
        "🚻 Bien. ¿Cuál es el sexo asignado al nacer y el género actual?",
    "antecedentes_medicos":
        "📝 ¿Antecedentes médicos relevantes (enfermedades previas, cirugías, alergias, medicación)?",
}

def get_system_instruction():
    return (
        "Eres una IA médica multimodal experta en interpretación de imágenes médicas. "
        "Al recibir una imagen, realiza un análisis profundo y estructurado. "
        "Luego, recopila datos clínicos paso a paso y al final sugiere diagnósticos y recomendaciones."
    )

def build_summary(collected: dict) -> str:
    if not collected:
        return ""
    lines = [f"- **{k.replace('_',' ').capitalize()}**: {v}"
             for k, v in collected.items()]
    return "📋 Información recopilada hasta ahora:\n" + "\n".join(lines) + "\n\n"

# ——— Estado en memoria ———
# session_data[sid] = { "fields": {...}, "image_analyzed": bool }
session_data = {}

# ——— Nuevo endpoint para solo análisis de imagen ———
@app.route('/api/analyze-image', methods=['OPTIONS','POST'])
def analyze_image():
    if request.method == 'OPTIONS':
        return make_response()
    data = request.json or {}
    image_b64  = data.get('image')
    image_type = data.get('image_type')
    if not image_b64 or not image_type:
        return jsonify({"error": "Falta image o image_type"}), 400

    prompt = (
        "🖼️ **Análisis exhaustivo de imagen**:\n"
        "1. 🔍 **Calidad técnica**: evalúa proyección, resolución, contraste y artefactos.\n"
        "2. 🧩 **Estructuras y morfología**: describe anatomía visible, contornos y simetría.\n"
        "3. 📐 **Medidas y proporciones**: menciona dimensiones y relaciones relevantes.\n"
        "4. ⚠️ **Hallazgos patológicos**: identifica lesiones, masas, calcificaciones, edema.\n"
        "5. 💡 **Hipótesis diagnóstica diferencial**: posibles causas, jerarquizadas.\n"
        "6. 📝 **Recomendaciones**: estudios adicionales y pasos clínicos.\n"
        "Usa una respuesta bien seccionada, con emojis moderados."
    )
    parts = [
        {"mime_type": image_type, "data": image_b64},
        {"text": f"{get_system_instruction()}\n\n{prompt}"}
    ]
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp  = model.generate_content({"parts": parts})
        return jsonify({"response": resp.text.strip()})
    except Exception as e:
        logger.error("Error en /api/analyze-image", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ——— Endpoint principal de chat ———
@app.route('/api/chat', methods=['POST'])
def chat():
    data       = request.json or {}
    user_text  = data.get('message', '').strip()
    image_b64  = data.get('image')
    image_type = data.get('image_type')

    # Inicializar sesión si no existe
    sid = flask_session.get('session_id')
    if not sid or sid not in session_data:
        sid = str(uuid.uuid4())
        flask_session['session_id'] = sid
        flask_session['step'] = 0
        session_data[sid] = {"fields": {}, "image_analyzed": False}
        logger.info(f"Nueva sesión {sid}")

    step       = flask_session.get('step', 0)
    state      = session_data[sid]
    collected  = state["fields"]
    image_done = state["image_analyzed"]

    parts = []
    # Caso 1: imagen primero
    if image_b64 and not image_done:
        parts.append({"mime_type": image_type, "data": image_b64})
        prompt_text = (
            "🖼️ **Análisis exhaustivo de imagen**:\n"
            "1. 🔍 Calidad técnica\n"
            "2. 🧩 Estructuras y morfología\n"
            "3. 📐 Medidas y proporciones\n"
            "4. ⚠️ Hallazgos patológicos\n"
            "5. 💡 Hipótesis diagnóstica\n"
            "6. 📝 Recomendaciones\n"
            "Responde de forma seccionada y detallada."
        )
        state["image_analyzed"] = True

    else:
        # Guardar texto en campo si corresponde
        if user_text and step < len(required_fields):
            campo = required_fields[step]
            collected[campo] = user_text
            logger.info(f"Sesión {sid}: guardado {campo} = {user_text!r}")
            step += 1
            flask_session['step'] = step

        # Preparar siguiente prompt
        if step < len(required_fields):
            pregunta = field_prompts[required_fields[step]].format(**collected)
            summary = build_summary(collected)
            prompt_text = summary + pregunta
        else:
            info = "\n".join(f"- {k}: {v}" for k, v in collected.items())
            prompt_text = (
                "✅ Gracias por la información clínica.\n"
                "🔍 Hallazgos\n"
                "💡 Hipótesis diagnóstica\n"
                "📝 Recomendaciones\n\n"
                f"📋 Datos completos:\n{info}"
            )
            # Reset sesión
            session_data.pop(sid, None)
            flask_session.pop('session_id', None)
            flask_session.pop('step', None)
            logger.info(f"Sesión {sid} completada")

    # Añadir texto con instrucción del sistema
    full_text = f"{get_system_instruction()}\n\n{prompt_text}"
    parts.append({"text": full_text})

    # Llamada al modelo
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp  = model.generate_content({"parts": parts})
        return jsonify({"response": resp.text.strip()})
    except Exception as e:
        logger.error("Error en /api/chat", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
