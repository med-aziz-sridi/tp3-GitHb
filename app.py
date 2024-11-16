from flask import Flask, request, render_template, redirect, url_for
import cv2
import easyocr
import numpy as np
import os
from MedicalReportProcessor import MedicalReportProcessor 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  
import os

@app.route('/upload', methods=['POST'])
def upload():
    print("Request received")
    
   
    if 'file' not in request.files:
        print("No file part")
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return redirect(url_for('index'))

   
    temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(temp_image_path)

    # Process the image
    processor = MedicalReportProcessor(debug=False)  # Disable debug visualization for production
    report_data = processor.process_report(temp_image_path)  # Pass the file path

    # Debugging: Check the extracted report data
    print("Extracted Report Data:", report_data)
    
    # Check if report_data is valid
    if not report_data or 'header' not in report_data or 'test_results' not in report_data:
        print("Report data extraction failed")
        return redirect(url_for('index'))

    # Pass the extracted data to the results template
    return render_template('index.html', header=report_data['header'], test_results=report_data['test_results'])

if __name__ == '__main__':
    app.run(debug=True)
