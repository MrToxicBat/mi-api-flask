import logging
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

session_steps = {}
session_data = {}
session_admin = {}

questions = {
    1: "Edad del paciente:",
    2: "Sexo asignado al nacer y género actual:",
    3: "Motivo principal de consulta:",
    4: "¿Desde cuándo presenta estos síntomas? ¿Han cambiado con el tiempo?",
    5: "Antecedentes médicos personales (crónicos, quirúrgicos, etc.):",
    6: "Medicamentos actuales (nombre, dosis, frecuencia):",
    7: "Alergias conocidas (medicamentos, alimentos, etc.):",
    8: "Antecedentes familiares relevantes:",
    9: "Estudios diagnósticos realizados y resultados si se conocen:"
}

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message', '').strip()
    image_data = data.get('image')

    logger.info(f"Entrando en chat(): session_id={session_id}, user_message={user_message}, image_data={bool(image_data)}")

    if not session_id or session_id not in session_steps:
        session_id = str(uuid.uuid4())
        session_steps[session_id] = 1
        session_data[session_id] = []
        session_admin[session_id] = False
        logger.info(f"Nuevo session_id creado: {session_id}")
        return jsonify({"session_id": session_id, "response": questions[1]})

    step = session_steps[session_id]
    is_admin = session_admin.get(session_id, False)

    logger.info(f"session_id={session_id}, step={step}, is_admin={is_admin}")

    if user_message.lower() == "admin":
        session_admin[session_id] = True
        logger.info(f"Modo Admin activado para session_id: {session_id}")
        return jsonify({"session_id": session_id, "response": "🔓 Modo Admin activado. Ahora puedes escribir libremente o subir imágenes."})

    if image_data and not user_message:
        logger.info(f"Solo imagen recibida, devolviendo pregunta actual para session_id: {session_id}, step: {step}")
        return jsonify({"session_id": session_id, "response": questions[step]})

    def is_valid_response(text):
        if is_admin:
            return True
        if not text.strip():
            return False
        if step == 1:
            return bool(re.search(r'\d{1,3}', text))
        if step == 2:
            return any(g in text.lower() for g in ["masculino", "femenino", "m", "f", "hombre", "mujer"])
        return True

    if user_message:
        if is_valid_response(user_message):
            session_data[session_id].append(user_message)
            logger.info(f"Respuesta válida recibida para session_id: {session_id}, step: {step}. session_data: {session_data[session_id]}")

            if step < len(questions):
                session_steps[session_id] += 1
                new_step = session_steps[session_id]
                logger.info(f"Incrementando step para session_id: {session_id}. nuevo step: {new_step}")
                if new_step <= len(questions):
                    return jsonify({"session_id": session_id, "response": questions[new_step]})
            else:
                session_steps[session_id] += 1  # Para permitir el análisis final
                logger.info(f"Todas las preguntas respondidas para session_id: {session_id}. Preparando para el análisis.")
        else:
            logger.warning(f"Respuesta inválida recibida para session_id: {session_id}, step: {step}")
            return jsonify({"session_id": session_id, "response": "⚠️ Por favor, proporcione una respuesta válida."})

    # Si ya se completaron todas las preguntas o está en modo admin, generar el análisis
    if session_steps[session_id] > len(questions) or is_admin:
        respuestas = dict(zip(questions.values(), session_data[session_id]))
        edad = next((v for k, v in respuestas.items() if "Edad" in k), "")
        sexo = next((v for k, v in respuestas.items() if "Sexo" in k), "")
        motivo = next((v for k, v in respuestas.items() if "Motivo" in k), "")

        if not is_admin and (not edad.strip() or not sexo.strip() or not motivo.strip()):
            logger.warning(f"Faltan datos esenciales para el análisis en session_id: {session_id}")
            return jsonify({"session_id": session_id, "response": "⚠️ Necesito edad, sexo y motivo de consulta para poder continuar. Por favor, verifica que hayas respondido esas preguntas."})

        info = "\n".join(f"{i + 1}. {q}\n→ {a}" for i, (q, a) in enumerate(zip(questions.values(), session_data[session_id])))
        analysis_prompt = f"Gracias. A continuación se presenta un informe clínico con base en la información suministrada.\n\n---\n\n📝 **Informe Clínico Detallado**\n\n📌 Datos Recopilados:\n{info}\n\n🔍 **Análisis Clínico**\nPor favor, interpreta esta información desde el punto de vista médico y sugiere hipótesis diagnósticas posibles con base en evidencia científica, factores de riesgo, y la presentación del caso. Finaliza con recomendaciones para el médico tratante."

        parts = [
            {"role": "system", "parts": [SYSTEM_PROMPT]},
            {"role": "user", "parts": [analysis_prompt]}
        ]

        if image_data:
            try:
                image_bytes = base64.b64decode(image_data.split(',')[-1])
                parts[1]["parts"].append({"inline_data": {"mime_type": "image/png", "data": image_bytes}})
                logger.info(f"Imagen adjuntada para el análisis en session_id: {session_id}")
            except Exception as e:
                logger.warning("No se pudo procesar la imagen enviada.", exc_info=True)

        try:
            # response = get_cached_response(tuple(map(str, parts))) #Quité el caché para ver si así funcionaba
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            response = model.generate_content(parts)
            ai_response = getattr(response, 'text', '').strip()
            logger.info(f"Respuesta de la IA para session_id: {session_id}: {ai_response}")
            return jsonify({"session_id": session_id, "response": ai_response})
        except Exception as e:
            logger.error(f"Error en /api/chat para session_id: {session_id}: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Fallback final: repetir la pregunta actual
    logger.info(f"Fallback: repitiendo la pregunta actual (step={step}) para session_id: {session_id}")
    return jsonify({"session_id": session_id, "response": questions[step]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    
