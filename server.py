from flask import Flask, request, jsonify, render_template, session, send_file
import os
import io
import base64
from PIL import Image
import json
from datetime import datetime
import uuid  
from flask_session import Session  
import webbrowser
from threading import Timer
import inference_app
import llm_utils 
import db_utils
import sqlite3 # Make sure this is imported if not already in db_utils

# Import WeasyPrint (after installing it: pip install weasyprint)
from weasyprint import HTML

app = Flask(__name__)
app.secret_key = os.urandom(24)  

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'flask_sessions'  
app.config['SESSION_PERMANENT'] = False
Session(app) 

os.makedirs('temp_uploads', exist_ok=True)
os.makedirs('flask_sessions', exist_ok=True)
os.makedirs('temp_images', exist_ok=True) 
os.makedirs('patient_data', exist_ok=True) 

# --- Initialize Database ---
print("Initializing database...")
db_utils.init_db()
print("Database initialized.")

# --- Configuration ---
# Ensure MODEL_DIR in inference_app.py points to the correct absolute path
# e.g., MODEL_DIR = 'e:\\bcrrp\\models'

# --- Load Models on Startup ---
print("Initializing application and loading models...")
if not inference_app.load_all_models():
    print("Critical error: Model loading failed. The application might not work correctly.")
    # In a production app, you might want to exit or prevent the server from starting.
else:
    print("Models loaded successfully.")

def convert_pil_to_base64(pil_image):
    if pil_image is None:
        return None
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def save_permanent_image(pil_image, patient_id, prefix="img"):
    """Save image to permanent storage for a patient"""
    if pil_image is None:
        return None
        
    # Create patient directory if it doesn't exist
    patient_dir = os.path.join('patient_data', patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    
    # Generate a unique filename
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(patient_dir, filename)
    
    # Save the image
    pil_image.save(filepath)
    return filepath

def save_uploaded_file(file, patient_id, prefix="file"):
    """Save an uploaded file to permanent storage for a patient"""
    if file is None:
        return None
        
    # Create patient directory if it doesn't exist
    patient_dir = os.path.join('patient_data', patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    
    # Generate a unique filename with original extension
    original_ext = os.path.splitext(file.filename)[1]
    filename = f"{prefix}_{uuid.uuid4().hex}{original_ext}"
    filepath = os.path.join(patient_dir, filename)
    
    # Save the file
    file.save(filepath)
    return filepath

def get_image_as_base64(filepath):
    """Convert a saved image file to base64"""
    if not filepath or not os.path.exists(filepath):
        return None
        
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patients')
def patients_list():
    """Display a list of all patients"""
    patients = db_utils.get_all_patients()
    return render_template('patients.html', patients=patients)

@app.route('/patient/<patient_id>')
def patient_details(patient_id):
    """Display details for a specific patient"""
    records = db_utils.get_patient_records(patient_id)
    if not records:
        return render_template('error.html', message=f"Patient with ID {patient_id} not found")
    
    return render_template('patient_details.html', records=records)

@app.route('/submit_questionnaire', methods=['POST'])
def submit_questionnaire():
    """Handle patient questionnaire submission"""
    try:
        questionnaire_data = request.json
        
        if not questionnaire_data:
            return jsonify({'error': 'No questionnaire data provided'}), 400
            
        # Store questionnaire data in session
        session['patient_data'] = {
            'questionnaire': questionnaire_data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to database
        patient_id = db_utils.save_patient_data({'questionnaire': questionnaire_data})
        
        return jsonify({
            'success': True, 
            'message': 'Questionnaire data saved successfully',
            'patient_id': patient_id
        })
        
    except Exception as e:
        app.logger.error(f"Error processing questionnaire: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred while processing the questionnaire: {str(e)}'}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image_route():
    app.logger.info(f"--- /predict_image route called ---") # Log route entry
    app.logger.info(f"Session data at start of /predict_image: {session}")
    app.logger.info(f"Request files: {request.files}")
    app.logger.info(f"Request form data: {request.form}") # Log incoming form data

    if 'file' not in request.files:
        app.logger.error("No 'file' part in request.files")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        app.logger.error("No selected file (filename is empty)")
        return jsonify({'error': 'No selected file'}), 400

    # Get patient ID from form data or session
    patient_id = request.form.get('patient_id')
    app.logger.info(f"Patient ID from request.form.get('patient_id'): {patient_id}")

    if not patient_id:
        app.logger.info("Patient ID not in form, trying session.")
        if 'patient_data' in session and session.get('patient_data', {}).get('questionnaire'):
            patient_id = session['patient_data']['questionnaire'].get('patientId')
            app.logger.info(f"Patient ID from session['patient_data']['questionnaire'].get('patientId'): {patient_id}")
        else:
            app.logger.warn(f"Attempted session retrieval: 'patient_data' in session: {'patient_data' in session}")
            if 'patient_data' in session:
                 app.logger.warn(f"'questionnaire' in session['patient_data']: {session.get('patient_data', {}).get('questionnaire') is not None}")


    if not patient_id:
        app.logger.error("Patient ID is missing after checking form and session.")
        return jsonify({'error': 'Patient ID is required. Please complete the questionnaire first.'}), 400
    
    app.logger.info(f"Proceeding with Patient ID: {patient_id}")

    if file:
        try:
            # Save the uploaded file temporarily
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_img_path = os.path.join(temp_dir, file.filename)
            file.save(temp_img_path)

            # Call the prediction function
            original_img_pil, grid_img_pil, result_text, ensemble_prob, model_preds_dict = inference_app.predict_image(temp_img_path)

            # Clean up the temporary file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            
            if "Error:" in result_text and original_img_pil is None:
                 return jsonify({'error': result_text}), 500

            # Save images to permanent storage
            original_img_path = save_permanent_image(original_img_pil, patient_id, "original")
            grid_img_path = save_permanent_image(grid_img_pil, patient_id, "grid")

            # Convert PIL images to base64 for JSON response
            original_b64 = convert_pil_to_base64(original_img_pil)
            visualization_b64 = convert_pil_to_base64(grid_img_pil)

            # Ensure all prediction probabilities are Python floats
            serializable_model_preds = {}
            if model_preds_dict:
                for model_name, prob in model_preds_dict.items():
                    serializable_model_preds[model_name] = float(prob) if prob is not None else None
            
            serializable_ensemble_prob = float(ensemble_prob) if ensemble_prob is not None else None

            # Store the image prediction results in the session
            image_predictions = {
                'result_text': result_text,
                'ensemble_probability': serializable_ensemble_prob,
                'model_predictions': serializable_model_preds,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            session['image_predictions'] = image_predictions
            session['image_data'] = {
                'original_image_path': original_img_path,
                'visualization_image_path': grid_img_path
            }

            # Save to database
            db_utils.save_image_prediction(
                patient_id, 
                image_predictions,
                original_img_path,
                grid_img_path
            )

            response_data = {
                'result_text': result_text,
                'ensemble_probability': serializable_ensemble_prob,
                'model_predictions': serializable_model_preds,
                'original_image_base64': original_b64,
                'visualization_image_base64': visualization_b64,
                'patient_id': patient_id
            }
            return jsonify(response_data)

        except Exception as e:
            app.logger.error(f"Error during prediction: {e}", exc_info=True)
            if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

    return jsonify({'error': 'File processing failed'}), 500

@app.route('/predict_gene', methods=['POST'])
def predict_gene_route():
    app.logger.info(f"--- /predict_gene route called ---")
    app.logger.info(f"Session data at start of /predict_gene: {session}")
    app.logger.info(f"Request files: {request.files}")
    app.logger.info(f"Request form data: {request.form}")

    if 'file' not in request.files:
        app.logger.error("No 'file' part in request.files")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        app.logger.error("No selected file (filename is empty)")
        return jsonify({'error': 'No selected file'}), 400

    # Get patient ID from form data or session
    patient_id = request.form.get('patient_id')
    app.logger.info(f"Patient ID from request.form.get('patient_id'): {patient_id}")

    if not patient_id:
        app.logger.info("Patient ID not in form, trying session.")
        if 'patient_data' in session and session.get('patient_data', {}).get('questionnaire'):
            patient_id = session['patient_data']['questionnaire'].get('patientId')
            app.logger.info(f"Patient ID from session['patient_data']['questionnaire'].get('patientId'): {patient_id}")

    if not patient_id:
        app.logger.error("Patient ID is missing after checking form and session.")
        return jsonify({'error': 'Patient ID is required. Please complete the questionnaire first.'}), 400
    
    app.logger.info(f"Proceeding with Patient ID: {patient_id}")

    if file:
        try:
            # Save the uploaded file temporarily
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, file.filename)
            file.save(temp_file_path)

            # Create a permanent directory for this patient
            patient_dir = os.path.join("patient_data", patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            # Generate a unique filename for permanent storage
            file_uuid = uuid.uuid4().hex
            permanent_file_name = f"gene_data_{file_uuid}.csv"
            permanent_file_path = os.path.join(patient_dir, permanent_file_name)
            
            # Copy the file to permanent storage
            import shutil
            shutil.copy2(temp_file_path, permanent_file_path)

            # Call the prediction function
            predicted_class, probability, error_message = inference_app.predict_gene_expression_data(temp_file_path)

            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            if error_message or predicted_class is None:
                return jsonify({'error': error_message or "Unknown prediction error"}), 500

            # Store the gene prediction results in the session
            gene_predictions = {
                'predicted_class': predicted_class,
                'probability': float(probability) if probability is not None else None,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            session['gene_predictions'] = gene_predictions
            session['gene_data'] = {
                'gene_data_path': permanent_file_path
            }

            # Save to database
            db_utils.save_gene_prediction(
                patient_id, 
                predicted_class,
                float(probability) if probability is not None else None,
                permanent_file_path
            )

            response_data = {
                'predicted_class': predicted_class,
                'probability': float(probability) if probability is not None else None,
                'patient_id': patient_id
            }
            return jsonify(response_data)

        except Exception as e:
            app.logger.error(f"Error during gene prediction: {e}", exc_info=True)
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

    return jsonify({'error': 'File processing failed'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate a comprehensive medical report using LLM"""
    try:
        request_data = request.json or {}
        app.logger.info(f"Received data for /generate_report: {request_data}") # Log received data

        patient_id = None
        questionnaire_data_from_request = request_data.get('questionnaireData')

        if questionnaire_data_from_request:
            patient_id = questionnaire_data_from_request.get('patientId')
            app.logger.info(f"Extracted patient_id from request payload: {patient_id}")

        if not patient_id:
            app.logger.info("Patient ID not in request payload, trying session.")
            if 'patient_data' in session and 'questionnaire' in session['patient_data']:
                patient_id = session['patient_data']['questionnaire'].get('patientId')
                app.logger.info(f"Extracted patient_id from session: {patient_id}")
        
        if not patient_id:
            app.logger.error("Patient ID is missing after checking payload and session for /generate_report.")
            return jsonify({'error': 'Patient ID is required. Please complete the questionnaire first.'}), 400
            
        app.logger.info(f"Proceeding with patient_id: {patient_id} for report generation.")
            
        # Try to get data from database first
        patient_records = db_utils.get_patient_records(patient_id)
        
        if not patient_records:
            app.logger.error(f"No records found for patient ID: {patient_id} in /generate_report")
            return jsonify({'error': f'No records found for patient ID: {patient_id}'}), 404
            
        # Extract data from records (or use what was passed if more up-to-date)
        # For now, let's assume the questionnaire data from the request is what we want to use for the report context
        # if it was passed, otherwise, we fall back to DB.
        # However, the primary patient_id for fetching records is crucial.

        patient_data_for_report = {}
        if questionnaire_data_from_request:
             patient_data_for_report['questionnaire'] = questionnaire_data_from_request
        elif patient_records['patient'] and patient_records['patient']['questionnaire_data']:
            patient_data_for_report['questionnaire'] = patient_records['patient']['questionnaire_data']
        else:
            app.logger.error(f"Questionnaire data missing for patient ID: {patient_id}")
            return jsonify({'error': f'Questionnaire data is missing for patient ID: {patient_id}'}), 400


        # Get image predictions (either from request_data or database)
        image_predictions_for_report = request_data.get('imageResults')
        if not image_predictions_for_report and patient_records['image_predictions']:
            image_predictions_for_report = patient_records['image_predictions'][0] # most recent
        
        # Get gene predictions (either from request_data or database)
        gene_predictions_for_report = request_data.get('geneResults')
        if not gene_predictions_for_report and patient_records['gene_predictions']:
            gene_predictions_for_report = patient_records['gene_predictions'][0] # most recent

        # Generate the report using our LLM utility
        report_text = llm_utils.generate_medical_report(
            patient_data_for_report, # Use the questionnaire data passed or fetched
            image_predictions_for_report,
            gene_predictions_for_report
        )
        
        # Format the report as HTML
        report_html = llm_utils.format_report_as_html(
            report_text,
            patient_data_for_report,
            image_predictions_for_report,
            gene_predictions_for_report
        )
        
        # Save the report to the database
        report_id = db_utils.save_report(
            patient_id,
            report_text,
            report_html
        )
        
        # Get image paths from database and convert to base64 for response
        # These might be different from what was passed if we are relying on DB records
        original_image_base64_response = request_data.get('originalImageBase64')
        visualization_image_base64_response = request_data.get('visualizationImageBase64')

        if not original_image_base64_response and image_predictions_for_report and image_predictions_for_report.get('original_image_path'):
            original_image_base64_response = get_image_as_base64(image_predictions_for_report['original_image_path'])
            
        if not visualization_image_base64_response and image_predictions_for_report and image_predictions_for_report.get('visualization_image_path'):
            visualization_image_base64_response = get_image_as_base64(image_predictions_for_report['visualization_image_path'])
        
        # Create a response with the report and images
        response_data = {
            'report_id': report_id,
            'report': report_text,
            'report_html': report_html,
            'patient_id': patient_id,
            'patient_data': patient_data_for_report,
            'image_predictions': image_predictions_for_report,
            'gene_predictions': gene_predictions_for_report,
            'original_image_base64': original_image_base64_response,
            'visualization_image_base64': visualization_image_base64_response,
            'database_link': f"/patient/{patient_id}"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Error generating report: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred while generating the report: {str(e)}'}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear all session data for a new patient"""
    try:
        session.clear()
        return jsonify({'success': True, 'message': 'Session cleared successfully'})
    except Exception as e:
        app.logger.error(f"Error clearing session: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred while clearing the session: {str(e)}'}), 500

@app.route('/db_stats')
def database_stats():
    """Get statistics about the database"""
    stats = db_utils.get_database_stats()
    return jsonify(stats)

@app.route('/download_report/<patient_id>/<report_id>')
def download_report(patient_id, report_id):
    """Generate and download a PDF report"""
    try:
        conn = sqlite3.connect(db_utils.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT report_html FROM reports WHERE id = ? AND patient_id = ?", (report_id, patient_id))
        report = cursor.fetchone()
        
        if not report:
            app.logger.error(f"Report not found for patient_id: {patient_id}, report_id: {report_id}")
            return render_template('error.html', message="Report not found or does not belong to this patient"), 404
            
        report_html = report['report_html']
        
        # Create a BytesIO object to store the PDF
        pdf_bytes = io.BytesIO()
        
        # Convert HTML to PDF
        HTML(string=report_html).write_pdf(pdf_bytes)
        
        # Seek to the beginning of the BytesIO object
        pdf_bytes.seek(0)
        
        return send_file(
            pdf_bytes,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'medical_report_{patient_id}_{report_id}.pdf'
        )
        
    except Exception as e:
        app.logger.error(f"Error generating or downloading report: {e}", exc_info=True)
        return render_template('error.html', message=f"Error generating report: {str(e)}"), 500

@app.route('/image/<path:filepath>')
def serve_image(filepath):
    """Serve an image file"""
    try:
        return send_file(filepath)
    except Exception as e:
        app.logger.error(f"Error serving image: {e}", exc_info=True)
        return jsonify({'error': f'Image not found: {str(e)}'}), 404

@app.route('/save_patient', methods=['POST'])
def save_patient():
    data = request.get_json()
    if not data or 'questionnaire' not in data:
        return jsonify({'success': False, 'error': 'Invalid data'}), 400
    try:
        patient_id = db_utils.save_patient_data(data) 
        return jsonify({'success': True, 'patient_id': patient_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=False, host='0.0.0.0', port=5000) # Runs on http://localhost:5000