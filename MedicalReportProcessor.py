import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import json
import re
from typing import Dict, List, Tuple
import pandas as pd
class MedicalReportProcessor:
    def __init__(self, debug=True):
        self.reader = easyocr.Reader(['fr'])
        self.debug = debug
        
    def visualize_ocr_results(self, image: np.ndarray, ocr_results: List, title: str):
        """Visualize OCR detection results"""
        if not self.debug:
            return
            
        # Create a copy of the image for visualization
        vis_img = image.copy()
        
        plt.figure(figsize=(15, 10))
        
        # Original image with boxes
        plt.subplot(1, 2, 1)
        plt.title(f'Original - {title}')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Image with detection boxes
        plt.subplot(1, 2, 2)
        plt.title(f'OCR Detections - {title}')
        
        # Draw detection boxes and text
        for (bbox, text, prob) in ocr_results:
            # Convert bbox points to integers
            bbox = np.array(bbox).astype(int)
            
            # Draw the bounding box
            cv2.polylines(vis_img, [bbox], True, (255, 0, 0), 2)
            
            # Add detected text above the box
            cv2.putText(vis_img, text, (bbox[0][0], bbox[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print detected text with confidence
        print(f"\nDetected text in {title}:")
        for _, text, prob in ocr_results:
            print(f"Text: {text:<50} Confidence: {prob:.2f}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Simple preprocessing - just convert to grayscale"""
        if self.debug:
            plt.figure(figsize=(10, 5))
            
            # Original
            plt.subplot(121)
            plt.title('Original')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            # Grayscale
            plt.subplot(122)
            plt.title('Grayscale')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plt.imshow(gray, cmap='gray')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def extract_header_info(self, header_text: List[Tuple]) -> Dict:
        header_info = {
            'hospital_name': '',
            'phone': '',
            'fax': '',
            'patient_name': '',
            'patient_id': '',
            'reference': '',
            'date': '',
            'category': '',
            'value': ''
        }
        
        # Create a more accessible format for the text elements
        text_elements = []
        for bbox, text, confidence in header_text:
            x, y = bbox[0][0], bbox[0][1]  # Get x,y coordinates
            text_elements.append({
                'text': text.strip(),
                'x': x,
                'y': y,
                'confidence': confidence
            })
        
        # Sort primarily by x position, then by y position
        x_sorted = sorted(text_elements, key=lambda x: (x['x'], x['y']))
        # Sort by y position for vertical scanning
        y_sorted = sorted(text_elements, key=lambda x: (x['y'], x['x']))
        
        if self.debug:
            print("\nProcessing Header Information:")
        
        # Process hospital information (found at the top)
        for elem in y_sorted[:5]:  # Check first few elements
            if "HOPITAL" in elem['text'].upper():
                header_info['hospital_name'] = elem['text']
                break
        
        # Process contact information
        for elem in text_elements:
            if "Tel" in elem['text']:
                header_info['phone'] = re.sub(r'[^0-9-]', '', elem['text'])
            elif "Fax" in elem['text']:
                header_info['fax'] = re.sub(r'[^0-9-]', '', elem['text'])
        
        # Find patient information (in the box on the right, typically around y=160-220)
        patient_box_elements = [elem for elem in text_elements 
                            if 150 < elem['y'] < 220 and elem['x'] > 500]
        
        for i, elem in enumerate(patient_box_elements):
            if re.match(r'^[A-Z\s]+$', elem['text']) and i + 1 < len(patient_box_elements):
                next_elem = patient_box_elements[i + 1]
                if "PERSONNEL" in next_elem['text']:
                    header_info['patient_name'] = elem['text']
                    # Look for patient ID
                    if i + 2 < len(patient_box_elements):
                        id_elem = patient_box_elements[i + 2]
                        if re.match(r'^[A-Z0-9]+/\d+$', id_elem['text']):
                            header_info['patient_id'] = id_elem['text']
        
        # Extract reference using regular expression
    
        reference_match = re.search(r'Référence\s+d[ce]mande\s*:\s*(\d+)\s*(\d+)', ' '.join([elem['text'] for elem in text_elements]))
        if reference_match:
            header_info['reference'] = f"{reference_match.group(1)}/{reference_match.group(2)}"


        # Extract date using regular expression
        date_match = re.search(r'Edité le\s+(\d{2}/\d{2}/\d{4})', ' '.join([elem['text'] for elem in text_elements]))
        if date_match:
            header_info['date'] = date_match.group(1)

        # Process category (around y=290-300)
        category_elements = [elem for elem in text_elements if 290 < elem['y'] < 300]
        for i, elem in enumerate(category_elements):
            if "catégorie" in elem['text'].lower() and i + 1 < len(category_elements):
                header_info['category'] = category_elements[i+1]['text']
        
        # Process value (around y=315-325)
        value_elements = [elem for elem in text_elements if 315 < elem['y'] < 325]
        value_elements.sort(key=lambda x: x['x'])
        for i, elem in enumerate(value_elements):
            if "valeur" in elem['text'].lower():
                # Combine next elements for full value
                if i + 2 < len(value_elements):
                    value_text = value_elements[i+2]['text']
                    if re.match(r'\d+[.,]\d+', value_text):
                        header_info['value'] = value_text
        
        if self.debug:
            print("\nExtracted Header Information:")
            for key, value in header_info.items():
                print(f"{key}: {value}")
        
        return header_info

    def extract_test_results(self, results_text: List[Tuple]) -> List[Dict]:
        """
        Extract test names, values, and units from OCR results where each component
        may be on a separate line but aligned vertically.
        """
        test_results = []
        
        # Sort text by vertical position and create groups of nearby text
        threshold_y = 10  # Adjust this value based on your OCR output spacing
        
        # Sort by Y coordinate first, then X coordinate
        sorted_text = sorted(results_text, key=lambda x: (x[0][0][1], x[0][0][0]))
        
        current_group = []
        current_y = None
        
        if self.debug:
            print("\nProcessing Test Results:")
        
        # Group items that are on roughly the same line
        groups = []
        for text_tuple in sorted_text:
            bbox, text, confidence = text_tuple
            y_pos = bbox[0][1]
            text = text.strip()
            
            # Skip header
            if "EXAMENS BIOCHIMIQUES" in text:
                continue
                
            if current_y is None:
                current_y = y_pos
                current_group.append((text, confidence, bbox[0][0]))  # Include X position
            elif abs(y_pos - current_y) <= threshold_y:
                current_group.append((text, confidence, bbox[0][0]))
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [(text, confidence, bbox[0][0])]
                current_y = y_pos
                
        if current_group:
            groups.append(current_group)
        
        # Process each group to extract test information
        for group in groups:
            if len(group) >= 2:  # We need at least a name and value
                # Sort items in group by X position
                group.sort(key=lambda x: x[2])
                
                # Extract components
                test_name = group[0][0].strip()
                value = None
                unit = None
                
                # Look for the value and unit
                for item in group[1:]:
                    text = item[0].strip()
                    # If it looks like a number
                    if re.match(r'^[\d,\.]+$', text):
                        value = text.replace(',', '.')
                    # If it looks like a unit
                    elif re.match(r'^[a-zA-Z/µ]+$', text):
                        unit = text
                
                if value:  # Only add if we found a value
                    result = {
                        'test_name': test_name,
                        'value': value,
                        'unit': unit if unit else '',
                        'confidence': min(g[1] for g in group)  # Use minimum confidence from group
                    }
                    
                    if self.debug:
                        print(f"Found test result: {result}")
                    
                    test_results.append(result)
        
        return test_results

    def process_report(self, temp_image_path):
        # Read the image from the provided file path
        image = cv2.imread(temp_image_path)
        
        # Check if the image was loaded correctly
        if image is None:
            raise ValueError("The image could not be loaded. Check the file path or format.")

        # Proceed with processing
        height, width = image.shape[:2]

        if self.debug:
            plt.figure(figsize=(15, 10))
            plt.title('Original Document with Regions')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        # Define regions (adjusted proportions)
        header_height = int(height * 0.3)  # Increased header height
        header_img = image[0:header_height, :]
        results_img = image[header_height:, :width // 2]
        
        # Process regions
        header_processed = self.preprocess_image(header_img)
        header_text = self.reader.readtext(header_processed)
        self.visualize_ocr_results(header_img, header_text, "Header Region")

        results_processed = self.preprocess_image(results_img)
        results_text = self.reader.readtext(results_processed)
        self.visualize_ocr_results(results_img, results_text, "Results Region")
    
        # Extract information
        header_info = self.extract_header_info(header_text)
        test_results = self.extract_test_results(results_text)
        
        return {
            'header': header_info,
            'test_results': test_results
        }

    def save_to_json(self, data: Dict, output_path: str):
        """Save the extracted data to a JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def process_medical_report(image_path: str, output_json_path: str):
    processor = MedicalReportProcessor(debug=True)
    
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Could not load image. Please check the file path.")
    
    report_data = processor.process_report(image)  # Pass the image array
    processor.save_to_json(report_data, output_json_path)
    
    return report_data
