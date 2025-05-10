from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import logging
from werkzeug.utils import secure_filename

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# Configuración CORS específica para tu dominio WordPress
CORS(app, resources={
    r"/api/chat": {
        "origins": ["https://tudominio.com", "https://mi-api-flask-6i8o.onrender.com"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuración para archivos (16MB máximo)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        # Log de headers para depuración
        logger.info(f"Headers recibidos: {request.headers}")
        logger.info(f"Form data: {request.form}")
        
        # Validar Content-Type
        if 'multipart/form-data' not in request.content_type:
            return jsonify({"error": "Content-Type debe ser multipart/form-data"}), 400

        # Procesar mensaje e imagen
        mensaje = request.form.get("mensaje", "").strip()
        imagen = request.files.get("imagen")
        
        # Validar entrada
        if not mensaje and not imagen:
            return jsonify({"error": "Se requiere 'mensaje' o 'imagen'"}), 400

        # Procesar imagen si existe
        imagen_data = None
        if imagen:
            if not allowed_file(imagen.filename):
                return jsonify({"error": "Tipo de archivo no permitido"}), 400
                
            logger.info(f"Imagen recibida: {imagen.filename} ({imagen.content_type})")
            imagen_data = {
                "mime_type": imagen.content_type,
                "data": imagen.read()
            }

        # Construir prompt para Gemini
        model = genai.GenerativeModel("gemini-pro-vision" if imagen else "gemini-pro")
        
        parts = []
        if mensaje:
            parts.append({"text": mensaje})
        if imagen_data:
            parts.append(imagen_data)

        # Generar respuesta
        logger.info("Enviando solicitud a Gemini...")
        response = model.generate_content(parts)
        
        if not response.text:
            raise ValueError("La API no devolvió texto")

        logger.info("Respuesta generada exitosamente")
        return jsonify({
            "respuesta": response.text,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error en /api/chat: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"Error al procesar la solicitud: {str(e)}",
            "status": "error"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
