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

# Estado en memoria
# session_data[sid] = { "fields": {...}, "image_analyzed": bool }
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

    step       = flask_session.get('step', 0)
    state      = session_data[sid]
    collected  = state["fields"]
    image_done = state["image_analyzed"]

    parts = []
    # — Caso 1: analizar imagen antes que nada —
    if image_b64 and not image_done:
        parts.append({
            "mime_type": image_type,
            "data": image_b64
        })
        # Nuevo prompt de análisis super detallado
        prompt_text = (
            "🖼️ **Análisis exhaustivo de imagen**:\n"
            "1. 🔍 **Calidad técnica**: evalúa proyección, resolución, contraste y posibles artefactos.\n"
            "2. 🧩 **Estructuras y morfología**: describe directamente la anatomía visible, contornos, simetría, densidades.\n"
            "3. 📐 **Medidas y proporciones**: menciona dimensiones, relaciones anatómicas relevantes.\n"
            "4. ⚠️ **Hallazgos patológicos**: destaca zonas anómalas (lesiones, masas, calcificaciones, edema).\n"
            "5. 💡 **Hipótesis diagnóstica diferencial**: propone posibles causas ordenadas por probabilidad.\n"
            "6. 📝 **Recomendaciones**: sugiere estudios adicionales o pasos clínicos siguientes.\n"
            "Usa una respuesta bien seccionada, con emojis moderados para marcar apartados."
        )
        state["image_analyzed"] = True

    else:
        # — Guardar texto en campo si corresponde —
        if user_text and step < len(required_fields):
            campo = required_fields[step]
            collected[campo] = user_text
            logger.info(f"Sesión {sid}: guardado {campo} = {user_text!r}")
            step += 1
            flask_session['step'] = step

        # — Preparar siguiente prompt —
        if step < len(required_fields):
            siguiente = required_fields[step]
            question = field_prompts[siguiente].format(**collected)
            summary = build_summary(collected)
            prompt_text = summary + question
        else:
            # — Análisis final —
            info_lines = "\n".join(f"- {k}: {v}" for k, v in collected.items())
            prompt_text = (
                "✅ Gracias por toda la información clínica.\n"
                "Con estos datos, realiza un análisis detallado:\n"
                "• 🔍 Hallazgos\n"
                "• 💡 Hipótesis diagnóstica\n"
                "• 📝 Recomendaciones\n\n"
                f"📋 Información completa:\n{info_lines}"
            )
            # reset session
            session_data.pop(sid, None)
            flask_session.pop('session_id', None)
            flask_session.pop('step', None)
            logger.info(f"Sesión {sid} completada")

    # — Construir prompt completo y llamar al modelo —
    full_text = f"{get_system_instruction()}\n\n{prompt_text}"
    parts.append({"text": full_text})

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp  = model.generate_content({"parts": parts})
        ai_text = getattr(resp, "text", "").strip()
        return jsonify({"response": ai_text})
    except Exception as e:
        logger.error("Error en /api/chat", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
