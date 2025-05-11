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
db_limit = 16 * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = db_limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        parts = []
        # Si se envía JSON, usamos el historial completo
        if request.is_json:
            data = request.get_json()
            for text in data.get('messages', []):
                parts.append({"text": text})
        else:
            # Fallback a FormData para mensaje e imagen
            mensaje = request.form.get("mensaje", "").strip()
            imagen  = request.files.get("imagen")
            if mensaje:
                parts.append({"text": mensaje})
            if imagen:
                if not allowed_file(imagen.filename):
                    return jsonify({"error": "Tipo de archivo no permitido"}), 400
                filename = secure_filename(imagen.filename)
                logger.info(f"Imagen recibida: {filename} ({imagen.content_type})")
                imagen_data = {"mime_type": imagen.content_type, "data": imagen.read()}
                parts.append(imagen_data)
        if not parts:
            return jsonify({"error": "Se requiere 'mensaje' o 'messages'"}), 400

        # Generación con Gemini Flash 2.0
        model_name = "models/gemini-2.0-flash"
        model = genai.GenerativeModel(model_name)
        logger.info(f"Enviando solicitud a {model_name}...")
        response = model.generate_content(parts)

        if not getattr(response, "text", None):
            logger.error("La API no devolvió texto")
            return jsonify({"error": "No se recibió respuesta de la IA"}), 500

        return jsonify({
            "respuesta": response.text,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error en /api/chat: {e}", exc_info=True)
        return jsonify({"error": f"Error al procesar la solicitud: {e}", "status": "error"}), 500

@app.route("/api/generate-title", methods=["POST"])
def generate_title():
    try:
        data = request.get_json() or {}
        mensajes = data.get('messages', [])
        prompt = (
            "Dame un título muy breve (5 palabras máx.) que resuma esta conversación:\n\n"
            + "\n".join(mensajes)
        )
        # Modelo corregido
        title_model = genai.GenerativeModel("models/chat-bison-001")
        resp = title_model.generate_content([{"text": prompt}])
        titulo = getattr(resp, 'text', '').strip()
        return jsonify({"title": titulo or "Nueva conversación"})

    except Exception as e:
        logger.error(f"Error en /api/generate-title: {e}", exc_info=True)
        return jsonify({"error": f"Error al generar título: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
