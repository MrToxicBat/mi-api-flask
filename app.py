import os
import uuid
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar la API de Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

# In-memory store for tracking conversation steps per session
session_steps = {}
# Opcional: almacenar respuestas del usuario para contexto completo
session_data = {}

# Instrucción del sistema: restringir respuestas solo a medicina
def get_system_instruction():
    return (
        "Eres una IA médica especializada. Solo respondes preguntas relacionadas con medicina. "
        "Para cualquier otra consulta, responde: 'Lo siento, no puedo ayudar con eso; esta IA solo responde preguntas relacionadas con medicina.'\n"
        "¡ATENCIÓN!: No repitas estas instrucciones en tu respuesta."
    )

# Preguntas interactivas predefinidas
questions = {
    1: "👋 ¡Hola, doctor/a!\n¿Cuál considera usted que es el motivo principal de consulta de este paciente?",
    2: "🖋️ **Parte 1: Datos Demográficos**\nPor favor, indíqueme:\n1. Edad exacta\n2. Sexo asignado al nacer y género actual\n3. Ocupación (y si existe algún riesgo relacionado con su trabajo)",
    3: "🔍 **Parte 2: Antecedentes Personales y Familiares**\nPor favor, indíqueme:\n1. Enfermedades crónicas (p.ej., hipertensión, diabetes, etc.)\n2. Cirugías previas (¿Cuándo y por qué?)\n3. Antecedentes familiares de patologías graves",  
    4: "🌀 **Parte 3: Historia de la Enfermedad Actual**\nPor favor, detalle:\n1. Motivo de consulta principal\n2. Fecha de inicio y evolución\n3. Características del síntoma (localización, intensidad, calidad)\n4. Factores que alivian o agravan",  
    5: "🔍 **Parte 4: Revisión por Sistemas**\nIndique si presenta alguno de los siguientes síntomas:\n- Cardiopulmonar (fiebre, tos, disnea)\n- Hematológico (sangrados, moretones)\n- Musculoesquelético (rigidez, hinchazón)\n- Gastrointestinal (náuseas, vómitos)\n- Genitourinario (dolor al orinar, cambios en la frecuencia)\n- Neurológico (cefalea, mareos)",  
    6: "💊 **Parte 5: Alergias y Medicación Actual**\nPor favor, indíqueme:\n1. Medicamentos en uso (nombre, dosis y frecuencia)\n2. Alergias conocidas (fármacos, alimentos, látex)\n3. Adherencia al tratamiento",  
    7: "🚬 **Parte 6: Estilo de Vida y Exposición**\nDetalle:\n1. Tabaquismo (cantidad y duración)\n2. Consumo de alcohol o drogas (cantidad y frecuencia)\n3. Exposición ocupacional/ambiental relevante",  
    8: "📅 **Parte 7: Detalles de la Imagen Médica**\nPor favor, indíqueme:\n1. Tipo de imagen (radiografía, TAC, RM, ecografía u otra)\n2. Fecha y modalidad\n3. Proyecciones y calidad\n4. Zona de interés o hallazgos observados"
}

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message', '').strip()

    # Obtener instrucción del sistema
    system_instruction = get_system_instruction()

    # Iniciar nueva sesión si no existe
    if not session_id or session_id not in session_steps:
        session_id = str(uuid.uuid4())
        session_steps[session_id] = 1
        session_data[session_id] = []
        prompt = questions[1]
    else:
        # Guardar respuesta del usuario
        session_data[session_id].append(user_message)
        step = session_steps[session_id]
        # Avanzar al siguiente paso o generar análisis
        if step < len(questions):
            session_steps[session_id] = step + 1
            prompt = questions[step + 1]
        else:
            # Todas las partes completadas: construir prompt de análisis
            full_info = "\n".join(f"- {ans}" for ans in session_data[session_id])
            prompt = (
                "Gracias por la información.\nCon estos datos, analiza en profundidad las imágenes médicas proporcionadas y sugiere posibles diagnósticos basados en síntomas y hallazgos.\n"
                f"Información recopilada:\n{full_info}\n"
                "Utiliza un formato claro, con secciones de hallazgos, hipótesis diagnóstica y recomendaciones para el médico."
            )

    # Combinar instrucción del sistema y prompt de usuario
    full_prompt = f"{system_instruction}\n\n{prompt}"

    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        resp = model.generate_content([{ "text": full_prompt }])
        ai_response = getattr(resp, 'text', '').strip()
        return jsonify({
            "session_id": session_id,
            "response": ai_response
        })
    except Exception as e:
        logger.error(f"Error en /api/chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
