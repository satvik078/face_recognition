from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import face_recognition
import pickle
import base64
import os

app = Flask(__name__)
CORS(app)

# Load known encodings and student IDs
with open("/Users/satvikpandey/Downloads/project/website (1)/face recognition/EncodeFile.p", "rb") as file:
    encodeListKnown, studentIds = pickle.load(file)

@app.route('/face-auth', methods=['POST'])
def face_auth():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        # Decode base64 image
        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get face encodings from the uploaded image
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        if not face_encodings:
            return jsonify({'success': False, 'error': 'No face detected'}), 200

        # Compare with known encodings
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
            face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = studentIds[best_match_index]
                return jsonify({'success': True, 'name': name}), 200

        return jsonify({'success': False, 'error': 'Face not recognized'}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # You can change host/port as needed
    app.run(host='0.0.0.0', port=8000)