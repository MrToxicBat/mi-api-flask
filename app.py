from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os

# Cargar la API Key desde variables de entorno
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

@app.route("/api/chat", methods=["POST"])
def chat():
    imagen = request.files.get("imagen")
    mensaje = request.form.get("mensaje", "")

    if not imagen:
        return jsonify({"respuesta": "No se recibió imagen"})

    imagen_bytes = imagen.read()

    model = genai.GenerativeModel("gemini-pro-vision")
    respuesta = model.generate_content([
        mensaje,
        genai.Image.from_bytes(imagen_bytes)
    ])

    return jsonify({"respuesta": respuesta.text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
