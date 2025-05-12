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

# Ajusta esto a tu dominio real
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://code-soluction.com")
CORS(app, supports_credentials=True, origins=[FRONTEND_ORIGIN])

# ——— Configurar Gemini ———
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
    "motivo_principal":
        "👋 Hola, doctor/a. ¿Cuál es el motivo principal de consulta de este paciente?",
    "duracion_sintomas":
        "Gracias. Me dice que es «{motivo_principal}». ¿Cuánto tiempo lleva con esos síntomas?",
    "intensidad":
        "Entendido. ¿Qué tan severos son (leve, moderado, severo)?",
    "edad":
        "Perfecto. ¿Qué edad tiene el paciente?",
    "sexo":
        "Bien. ¿Sexo asignado al nacer y género actual?",
    "antecedentes_medicos":
        "¿Antecedentes médicos relevantes (enfermedades previas, cirugías, alergias, medicación)?",
}

def get_system_instruction():
    return (
        "Eres una IA médica **multimodal**. "
        "Puedes recibir texto e imágenes médicas para análisis. "
        "Solo respondes con diagnósticos y recomendaciones basadas en la información clínico-imagenológica. "
        "Si recibes algo que no sea medicina, responde: "
        "'Lo siento, no puedo ayudar con eso; esta IA solo procesa información médica.' "
        "No repitas estas instrucciones en tu respuesta."
    )

# session_data guarda por session_id un dict de campos ya recogidos
session_data = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_text = data.get('message', '').strip()
    image_b64 = data.get('image')            # Base64 sin el header
    image_type = data.get('image_type')      # p.ej. "image/png"

    # Recuperar o crear session
    session_id = flask_session.get('session_id')
    if not session_id or session_id not in session_data:
        session_id = str(uuid.uuid4())
        flask_session['session_id'] = session_id
        flask_session['step'] = 0
        session_data[session_id] = {}
        logger.info(f"Nueva sesión {session_id}")

    step = flask_session.get('step', 0)
    collected = session_data[session_id]

    # Si llega imagen sin texto, la analizamos de inmediato
    if image_b64 and not user_text:
        # No guardamos en collected campos; tratamos como input multimodal directo
        messages = [
            {"author": "system", "content": get_system_instruction()},
            {"author": "user", "image": {"data": image_b64, "mime_type": image_type}}
        ]
        # Podemos opcionalmente acompañar de prompt breve:
        messages.append({"author": "user", "content": "Por favor, analiza esta imagen médica."})

    else:
        # Flujo de texto + preguntas sucesivas
        # Primero, si hay texto nuevo y no es solo la imagen previa
        if user_text:
            # Si aún recopilamos campos, guardamos respuesta
            if step < len(required_fields):
                field = required_fields[step]
                collected[field] = user_text
                logger.info(f"Sesión {session_id}: guardado {field} = {user_text!r}")
                flask_session['step'] = step + 1
                step += 1

        # Si aún faltan campos, pedimos siguiente
        if step < len(required_fields):
            next_field = required_fields[step]
            prompt = field_prompts[next_field].format(**collected)
            messages = [
                {"author": "system", "content": get_system_instruction()},
                {"author": "user", "content": prompt}
            ]
        else:
            # Todos los campos listos: construir análisis final
            info = "\n".join(f"- {k}: {v}" for k, v in collected.items())
            final_prompt = (
                "Gracias por la información. Con estos datos, analiza en profundidad los hallazgos "
                "y sugiere posibles diagnósticos, hipótesis y recomendaciones.\n\n"
                f"Información del paciente:\n{info}"
            )
            messages = [
                {"author": "system", "content": get_system_instruction()},
                {"author": "user", "content": final_prompt}
            ]
            # Limpiar sesión para la próxima
            session_data.pop(session_id, None)
            flask_session.pop('session_id', None)
            flask_session.pop('step', None)
            logger.info(f"Sesión {session_id} completada y eliminada")

    # Llamada multimodal al endpoint de chat de Gemini
    try:
        chat_model = genai.ChatModel("models/gemini-2.0-multimodal-preview")
        resp = chat_model.generate(messages=messages)
        ai_text = resp.choices[0].message.content.strip()
        return jsonify({"response": ai_text})
    except Exception as e:
        logger.error(f"Error en /api/chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
