from flask import Flask, request, jsonify, render_template, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        # Save the uploaded image
        filepath = 'uploaded_image.uploaded_image.jpg'
        file.save(filepath)

        # Process the image and detect parking spaces
        detected_image, parking_spots = detect_parking_spaces(filepath)
        
        # Save the result image
        result_filepath = 'static/result_image.jpg'
        cv2.imwrite(result_filepath, detected_image)

        return render_template('result.html', result_image_url=result_filepath, parking_spots=parking_spots, total_count=len(parking_spots))

def detect_parking_spaces(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    parking_spots = []
    for c in contours:
        if cv2.contourArea(c) > 500:  # Minimum area threshold to avoid small noise
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            parking_spots.append({"x": x, "y": y, "width": w, "height": h})
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image, parking_spots

if __name__ == '__main__':
    app.run(debug=True)
