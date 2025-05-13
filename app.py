import os
import uuid
import logging
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

# Inicializar Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "models/gemini-2.0-flash"

# Flujo de preguntas
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
    lines = [f"- **{k.replace('_',' ').capitalize()}**: {v}" for k, v in collected.items()]
    return "📋 Información recopilada hasta ahora:\n" + "\n".join(lines) + "\n\n"

# Memoria de sesión
session_data = {}  # {sid: {fields: {}, image_analyzed: bool}}

# ——— Endpoint para análisis de imagen ———
@app.route('/api/analyze-image', methods=['OPTIONS', 'POST'])
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
        "1. 🔍 Calidad técnica: proyección, contraste, artefactos.\n"
        "2. 🧩 Estructuras y morfología: anatomía, contornos, simetría.\n"
        "3. 📐 Medidas y proporciones: dimensiones clave.\n"
        "4. ⚠️ Hallazgos patológicos: lesiones, masas, calcificaciones.\n"
        "5. 💡 Hipótesis diagnóstica diferencial.\n"
        "6. 📝 Recomendaciones clínicas.\n"
        "Responde en secciones claras usando emojis moderados."
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
    user_text  = data.get('message','').strip()
    image_b64  = data.get('image')
    image_type = data.get('image_type')

    # Detección de palabras clave para respuestas basadas en imagen
    if user_text.lower() in ["resumen", "diagnostico", "tratamiento"] and image_b64:
        prompt_text = f"🖼️ Basado únicamente en la imagen, por favor proporciona el {user_text.capitalize()}."
        parts = [
            {"mime_type": image_type, "data": image_b64},
            {"text": f"{get_system_instruction()}\n\n{prompt_text}"}
        ]
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            resp  = model.generate_content({"parts": parts})
            return jsonify({"response": resp.text.strip()})
        except Exception as e:
            logger.error("Error en /api/chat (keyword image)", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Inicializar sesión
    sid = flask_session.get('session_id')
    if not sid or sid not in session_data:
        sid = str(uuid.uuid4())
        flask_session['session_id'] = sid
        flask_session['step'] = 0
        session_data[sid] = {"fields": {}, "image_analyzed": False}
        logger.info(f"Nueva sesión {sid}")

    step       = flask_session.get('step',0)
    state      = session_data[sid]
    collected  = state["fields"]
    image_done = state["image_analyzed"]

    parts = []
    # Si llega imagen y aún no se analizó
    if image_b64 and not image_done:
        parts.append({"mime_type": image_type, "data": image_b64})
        state["image_analyzed"] = True
        prompt_text = (
            "🖼️ **Análisis exhaustivo de imagen**:\n"
            "1. 🔍 Calidad técnica\n"
            "2. 🧩 Estructuras y morfología\n"
            "3. 📐 Medidas y proporciones\n"
            "4. ⚠️ Hallazgos patológicos\n"
            "5. 💡 Hipótesis diagnóstica\n"
            "6. 📝 Recomendaciones"
        )
    else:
        # Guardar texto en campo
        if user_text and step < len(required_fields):
            campo = required_fields[step]
            collected[campo] = user_text
            step += 1
            flask_session['step'] = step

        # Si faltan campos, preguntar
        if step < len(required_fields):
            pregunta = field_prompts[required_fields[step]].format(**collected)
            prompt_text = build_summary(collected) + pregunta
        else:
            info = "\n".join(f"- {k}: {v}" for k,v in collected.items())
            prompt_text = (
                "✅ Datos completos. Ahora realiza:\n"
                "• 🔍 Hallazgos\n"
                "• 💡 Hipótesis diagnóstica\n"
                "• 📝 Recomendaciones\n\n"
                f"📋 Información:\n{info}"
            )
            # reset
            session_data.pop(sid,None)
            flask_session.pop('session_id',None)
            flask_session.pop('step',None)

    # Construir prompt y llamar a Gemini
    full = f"{get_system_instruction()}\n\n{prompt_text}"
    parts.append({"text": full})

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp  = model.generate_content({"parts": parts})
        return jsonify({"response": resp.text.strip()})
    except Exception as e:
        logger.error("Error en /api/chat", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__=='__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT",5000)))
