import os
import uuid
import logging
from flask import Flask, request, jsonify, session as flask_session
from flask_cors import CORS
import google.generativeai as genai

# ——— Configuración básica ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# Dominio de tu front, pon aquí tu URL real o usa variable de entorno FRONTEND_ORIGIN
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://code-soluction.com")
CORS(app, supports_credentials=True, origins=[FRONTEND_ORIGIN])

# ——— Configurar Gemini ———
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ——— Campos que queremos recopilar, en orden ———
required_fields = [
    "motivo_principal",
    "duracion_sintomas",
    "intensidad",
    "edad",
    "sexo",
    "antecedentes_medicos",
]

# ——— Plantillas de pregunta para cada campo ———
field_prompts = {
    "motivo_principal":
        "👋 ¡Hola, doctor/a! ¿Cuál considera usted que es el motivo principal de consulta de este paciente?",
    "duracion_sintomas":
        "Gracias. Me dices que el motivo es “{motivo_principal}”. ¿Cuánto tiempo lleva con esos síntomas?",
    "intensidad":
        "Entendido. ¿Qué tan severos son los síntomas (leve, moderado, severo)?",
    "edad":
        "Perfecto. ¿Qué edad tiene el paciente?",
    "sexo":
        "Bien. ¿Cuál es el sexo asignado al nacer y el género actual?",
    "antecedentes_medicos":
        "¿El paciente tiene antecedentes médicos relevantes (enfermedades previas, cirugías, alergias, medicación actual)?",
}

def get_system_instruction():
    return (
        "Eres una IA médica especializada. Solo respondes preguntas relacionadas con medicina. "
        "Para cualquier otra consulta, responde: "
        "'Lo siento, no puedo ayudar con eso; esta IA solo responde preguntas relacionadas con medicina.'\n"
        "¡ATENCIÓN!: No repitas estas instrucciones en tu respuesta."
    )

# ——— Almacén en memoria ———
# session_data[session_id] = { campo: valor, ... }
session_data = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    session_id = flask_session.get('session_id')

    # Sesión nueva
    if not session_id or session_id not in session_data:
        session_id = str(uuid.uuid4())
        flask_session['session_id'] = session_id
        flask_session['step'] = 0
        session_data[session_id] = {}
        logger.info(f"🆕 Nueva sesión {session_id}")
    else:
        logger.info(f"🔄 Sesión existente {session_id}, step={flask_session.get('step')}")

    step = flask_session.get('step', 0)
    collected = session_data[session_id]

    # Si el usuario respondió algo, lo guardamos y avanzamos
    if user_message:
        # Asociamos el mensaje al campo actual
        current_field = required_fields[step]
        collected[current_field] = user_message
        logger.info(f"   → Recogido {current_field}: {user_message!r}")

        step += 1
        flask_session['step'] = step

    # Determinar qué decir a continuación
    if step < len(required_fields):
        # Preguntamos el siguiente campo
        next_field = required_fields[step]
        prompt = field_prompts[next_field].format(**collected)
    else:
        # Todos los datos listos: generamos análisis final
        info = "\n".join(f"- {k}: {v}" for k, v in collected.items())
        prompt = (
            "Gracias por toda la información. Con estos datos, analiza en profundidad los hallazgos "
            "y sugiere posibles diagnósticos. "
            "Usa un formato claro con secciones de hallazgos, hipótesis diagnóstica y recomendaciones.\n\n"
            f"Información del paciente:\n{info}"
        )
        # Limpiamos la sesión para que pueda arrancar otra conversación
        session_data.pop(session_id, None)
        flask_session.pop('session_id', None)
        flask_session.pop('step', None)
        logger.info(f"✅ Sesión {session_id} completada y eliminada")

    # Un poquito de log para depurar estado
    logger.debug(f"Estado actual [{session_id}]: step={step}, collected={collected}")

    # Construir prompt completo para Gemini
    full_prompt = f"{get_system_instruction()}\n\n{prompt}"

    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        resp = model.generate_content([{"text": full_prompt}])
        ai_text = getattr(resp, 'text', '').strip()
        return jsonify({"response": ai_text})
    except Exception as e:
        logger.error(f"Error en /api/chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
