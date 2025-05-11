import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from werkzeug.utils import secure_filename

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar la API de Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
# CORS para endpoints de API
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://code-soluction.com",
            "https://mi-api-flask-6i8o.onrender.com"
        ],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Límite de subida: 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        # Mensaje de sistema mejorado: español + viñetas bien formateadas
        SYSTEM_PROMPT = (
            'Eres un asistente experto que responde solo en español. '
            'Organiza tu respuesta con viñetas Markdown, cada punto en una línea nueva que empiece con "- ". '
            'Usa encabezados en negrita separados por una línea en blanco y no incluyas otros idiomas ni asteriscos extra. '
        )
        parts = [{"text": SYSTEM_PROMPT}]

        # Si se envía JSON con historial completo
        if request.is_json:
            data = request.get_json()
            for text in data.get('messages', []):
                parts.append({"text": text})
        else:
            # Fallback a FormData para texto e imagen
            mensaje = request.form.get("mensaje", "").strip()
            imagen  = request.files.get("imagen")
            if mensaje:
                parts.append({"text": mensaje})
            if imagen and allowed_file(imagen.filename):
                filename = secure_filename(imagen.filename)
                imagen_data = {"mime_type": imagen.content_type, "data": imagen.read()}
                parts.append(imagen_data)

        if len(parts) <= 1:
            return jsonify({"error": "Se requiere mensaje o historial"}), 400

        # Generación multimodal con Gemini Flash 2.0
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        response = model.generate_content(parts)

        if not getattr(response, "text", None):
            return jsonify({"error": "No se recibió respuesta de la IA"}), 500

        return jsonify({"respuesta": response.text, "status": "success"})

    except Exception as e:
        logger.error(f"Error en /api/chat: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/api/generate-title", methods=["POST"])
def generate_title():
    try:
        data = request.get_json() or {}
        mensajes = data.get('messages', [])
        prompt = (
            'Dame un título muy breve (5 palabras máx.) que resuma esta conversación en español, '
            'utilizando solo texto.'
            + '\n'.join(mensajes)
        )
        # Usamos Gemini Flash 2.0 para título
        title_model = genai.GenerativeModel("models/gemini-2.0-flash")
        resp = title_model.generate_content([{"text": prompt}])
        titulo = getattr(resp, 'text', '').strip()
        return jsonify({"title": titulo or "Nueva conversación"})

    except Exception as e:
        logger.error(f"Error en /api/generate-title: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
