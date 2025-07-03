import sqlite3
import json
import os
from datetime import datetime

# Database configuration
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bcrrp_data.db')

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create patients table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT UNIQUE,
        name TEXT,
        age INTEGER,
        gender TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        questionnaire_data TEXT
    )
    ''')
    
    # Create image_predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS image_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        result_text TEXT,
        ensemble_probability REAL,
        model_predictions TEXT,
        original_image_path TEXT,
        visualization_image_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    ''')
    
    # Create gene_predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS gene_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        predicted_class TEXT,
        probability REAL,
        gene_data_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    ''')
    
    # Create reports table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        report_text TEXT,
        report_html TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized at {DB_PATH}")
    return True

def save_patient_data(patient_data):
    """Save patient questionnaire data to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    questionnaire = patient_data.get('questionnaire', {})
    patient_id = questionnaire.get('patientId', f"P{datetime.now().strftime('%Y%m%d%H%M%S')}")
    name = questionnaire.get('name', 'Anonymous')
    age = questionnaire.get('age', 0)
    gender = questionnaire.get('gender', 'Unknown')
    
    # Check if patient already exists
    cursor.execute("SELECT id FROM patients WHERE patient_id = ?", (patient_id,))
    existing_patient = cursor.fetchone()
    
    if existing_patient:
        # Update existing patient
        cursor.execute('''
        UPDATE patients 
        SET name = ?, age = ?, gender = ?, questionnaire_data = ?
        WHERE patient_id = ?
        ''', (name, age, gender, json.dumps(questionnaire), patient_id))
    else:
        # Insert new patient
        cursor.execute('''
        INSERT INTO patients (patient_id, name, age, gender, questionnaire_data)
        VALUES (?, ?, ?, ?, ?)
        ''', (patient_id, name, age, gender, json.dumps(questionnaire)))
    
    conn.commit()
    conn.close()
    
    return patient_id

def save_image_prediction(patient_id, image_predictions, original_path=None, visualization_path=None):
    """Save image prediction results to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    result_text = image_predictions.get('result_text', 'N/A')
    ensemble_probability = image_predictions.get('ensemble_probability', None)
    model_predictions = json.dumps(image_predictions.get('model_predictions', {}))
    
    cursor.execute('''
    INSERT INTO image_predictions 
    (patient_id, result_text, ensemble_probability, model_predictions, original_image_path, visualization_image_path)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (patient_id, result_text, ensemble_probability, model_predictions, original_path, visualization_path))
    
    conn.commit()
    conn.close()
    
    return cursor.lastrowid

def save_gene_prediction(patient_id, predicted_class, probability, gene_data_path):
    """Save gene prediction results to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO gene_predictions (
            patient_id, predicted_class, probability, gene_data_path, created_at
        ) VALUES (?, ?, ?, ?, ?)
        ''', (
            patient_id, 
            predicted_class, 
            probability, 
            gene_data_path, 
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        return prediction_id
    except Exception as e:
        print(f"Error saving gene prediction: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def save_report(patient_id, report_text, report_html):
    """Save generated report to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO reports 
    (patient_id, report_text, report_html)
    VALUES (?, ?, ?)
    ''', (patient_id, report_text, report_html))
    
    conn.commit()
    conn.close()
    
    return cursor.lastrowid

def get_patient_data(patient_id):
    """Retrieve patient data by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    patient = cursor.fetchone()
    
    if not patient:
        conn.close()
        return None
    
    # Convert to dict
    patient_dict = dict(patient)
    
    # Parse JSON fields
    if patient_dict.get('questionnaire_data'):
        patient_dict['questionnaire_data'] = json.loads(patient_dict['questionnaire_data'])
    
    conn.close()
    return patient_dict

def get_patient_records(patient_id):
    """Get all records for a patient including predictions and reports"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get patient data
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    patient = cursor.fetchone()
    
    if not patient:
        conn.close()
        return None
    
    patient_dict = dict(patient)
    if patient_dict.get('questionnaire_data'):
        patient_dict['questionnaire_data'] = json.loads(patient_dict['questionnaire_data'])
    
    # Get image predictions
    cursor.execute("SELECT * FROM image_predictions WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,))
    image_predictions = [dict(row) for row in cursor.fetchall()]
    
    for pred in image_predictions:
        if pred.get('model_predictions'):
            pred['model_predictions'] = json.loads(pred['model_predictions'])
    
    # Get gene predictions
    cursor.execute("SELECT * FROM gene_predictions WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,))
    gene_predictions = [dict(row) for row in cursor.fetchall()]
    
    # Get reports
    cursor.execute("SELECT * FROM reports WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,))
    reports = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        'patient': patient_dict,
        'image_predictions': image_predictions,
        'gene_predictions': gene_predictions,
        'reports': reports
    }

def get_all_patients():
    """Get a list of all patients in the database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT patient_id, name, age, gender, created_at FROM patients ORDER BY created_at DESC")
    patients = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return patients

def get_database_stats():
    """Get statistics about the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM patients")
    patient_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM image_predictions")
    image_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM gene_predictions")
    gene_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM reports")
    report_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'patient_count': patient_count,
        'image_prediction_count': image_count,
        'gene_prediction_count': gene_count,
        'report_count': report_count,
        'database_path': DB_PATH
    }