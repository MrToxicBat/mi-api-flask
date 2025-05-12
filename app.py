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
# Habilitar CORS y que viaje la cookie de session_id
CORS(app, supports_credentials=True)

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

# ——— Instrucción de sistema para el LLM ———
def get_system_instruction():
    return (
        "Eres una IA médica especializada. Solo respondes preguntas relacionadas con medicina. "
        "Para cualquier otra consulta, responde: "
        "'Lo siento, no puedo ayudar con eso; esta IA solo responde preguntas relacionadas con medicina.'\n"
        "¡ATENCIÓN!: No repitas estas instrucciones en tu respuesta."
    )

# ——— Almacén en memoria ———
# session_data[session_id] = { campo1: valor1, campo2: valor2, ... }
session_data = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    # Obtener o crear session_id en la cookie
    session_id = flask_session.get('session_id')
    if not session_id or session_id not in session_data:
        # Sesión nueva
        session_id = str(uuid.uuid4())
        flask_session['session_id'] = session_id
        session_data[session_id] = {}
        current_field = required_fields[0]
        prompt = field_prompts[current_field]
    else:
        # Sesión ya existente
        collected = session_data[session_id]
        # Lista de campos que faltan
        missing = [f for f in required_fields if f not in collected]
        if missing and user_message:
            # Asignar la respuesta al campo que acabamos de pedir
            last_field = missing[0]
            collected[last_field] = user_message
            # ¿Qué campo viene ahora?
            missing = [f for f in required_fields if f not in collected]
            if missing:
                next_field = missing[0]
                # Formatear plantilla con lo que ya sabemos
                prompt = field_prompts[next_field].format(**collected)
            else:
                # ¡Todos los datos recogidos! Preparamos el análisis final
                full_info = "\n".join(f"- {k}: {v}" for k, v in collected.items())
                prompt = (
                    "Gracias por toda la información. Con estos datos, analiza en profundidad los hallazgos "
                    "y sugiere posibles diagnósticos. "
                    "Usa un formato claro con secciones de hallazgos, hipótesis diagnóstica y recomendaciones.\n\n"
                    f"Información del paciente:\n{full_info}"
                )
                # Después del análisis, reseteamos para empezar de nuevo
                del session_data[session_id]
                flask_session.pop('session_id', None)
        else:
            # No hay mensaje (vacío) o faltan datos pero no enviaron nada:
            # repetimos la misma pregunta
            missing = [f for f in required_fields if f not in session_data[session_id]]
            current_field = missing[0]
            prompt = field_prompts[current_field].format(**session_data[session_id])

    # Construir prompt completo para Gemini
    full_prompt = f"{get_system_instruction()}\n\n{prompt}"

    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        resp = model.generate_content([{"text": full_prompt}])
        ai_text = getattr(resp, 'text', '').strip()
        return jsonify({
            "response": ai_text
        })
    except Exception as e:
        logger.error(f"Error en /api/chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
