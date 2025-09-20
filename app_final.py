
from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from flask_cors import CORS
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Inicializar Flask y configurar carpeta de plantillas
app = Flask(__name__, template_folder='templates')
CORS(app)

# Carpeta para subir imágenes temporales
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo YOLOv8 entrenado
model = YOLO("best.pt")

@app.route('/')
def index():
    return render_template('index_final.html')  # Asegúrate que está en /templates

@app.route('/procesar', methods=['POST'])
def procesar():
    files = request.files.getlist('files')
    results = []

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        detections = model(img)[0]

        detections_json = []
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = float(box.conf[0])
            width = float(x2 - x1)
            height = float(y2 - y1)
            detections_json.append({
                'x': float(x1),
                'y': float(y1),
                'width': width,
                'height': height,
                'confidence': confidence
            })

        results.append({
            'filename': filename,
            'detections': detections_json
        })

        # os.remove(filepath)

    return jsonify(results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
