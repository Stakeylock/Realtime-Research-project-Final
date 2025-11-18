import os
import json
import time
import subprocess
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def ensure_ollama_running():
    """
    Ensure Ollama server is up. If not running, launch it automatically.
    """
    try:
        requests.get("http://localhost:11434/api/tags", timeout=1)
        return True  
    except:
        pass 

    print("[INFO] Starting Ollama server...")

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False
        )

        for _ in range(40): 
            try:
                requests.get("http://localhost:11434/api/tags", timeout=1)
                print("[INFO] Ollama started successfully.")
                return True
            except:
                time.sleep(0.25)

        print("[ERROR] Ollama failed to start.")
        return False

    except Exception as e:
        print("[ERROR] Failed to start Ollama:", e)
        return False


def ensure_model_exists(model_name="llama3.2"):
    """
    Ensure the required model is installed locally. If not, auto-pull it.
    """
    try:
        tags = requests.get("http://localhost:11434/api/tags").json()
        installed = [m["name"] for m in tags.get("models", [])]

        if model_name not in installed:
            print(f"[INFO] Pulling missing Ollama model: {model_name} ...")
            subprocess.run(["ollama", "pull", model_name])
            print("[INFO] Model pull completed.")

    except Exception as e:
        print("[WARNING] Unable to verify/pull model:", e)



def format_report_as_html(report_text, patient_data, image_predictions, gene_predictions):
    html = "<div class='medical-report'>"

    html += "<div class='report-header'>"
    html += "<h1>Breast Cancer Risk Assessment Report</h1>"
    html += f"<p class='report-date'>Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>"
    html += "</div>"

    paragraphs = report_text.split('\n\n')
    for p in paragraphs:
        if p.startswith('# '):
            html += f"<h2>{p[2:]}</h2>"
        elif p.startswith('## '):
            html += f"<h3>{p[3:]}</h3>"
        elif p.startswith('### '):
            html += f"<h4>{p[4:]}</h4>"
        elif p.startswith('- '):
            items = [item[2:] for item in p.split('\n- ')]
            html += "<ul>"
            for item in items:
                html += f"<li>{item}</li>"
            html += "</ul>"
        elif p.startswith('**'):
            if p.endswith('**'):
                content = p[2:-2]
                html += f"<p class='important'><strong>{content}</strong></p>"
            else:
                html += f"<p>{p}</p>"
        else:
            html += f"<p>{p}</p>"

    html += "</div>"
    return html



def generate_medical_report(patient_data, image_predictions, gene_predictions):
    """
    Generate a comprehensive medical report using local Ollama llama3.2.
    """

    # --- Auto start ollama serve ---
    if not ensure_ollama_running():
        return "Error: Could not start Ollama. Make sure it is installed."

    # --- Ensure model availability ---
    ensure_model_exists("llama3.2")

    questionnaire = patient_data.get("questionnaire", {})

    # ----- Prepare LLM Prompt -----
    prompt = f"""
You are an expert medical AI assistant helping to generate a comprehensive breast cancer risk assessment report.

## PATIENT INFORMATION
Patient ID: {questionnaire.get('patientId', 'Unknown')}
Name: {questionnaire.get('name', 'Anonymous')}
Age: {questionnaire.get('age', 'Unknown')}
Gender: {questionnaire.get('gender', 'Unknown')}
Date: {datetime.now().strftime("%Y-%m-%d")}

## FAMILY HISTORY
Family history of breast cancer: {questionnaire.get('familyHistoryBreastCancer', 'Unknown')}
Family history of other cancers: {questionnaire.get('familyHistoryOtherCancers', 'Unknown')}
Relatives with breast cancer: {questionnaire.get('relativesWithBreastCancer', 'Unknown')}
Age of relatives at diagnosis: {questionnaire.get('relativesDiagnosisAge', 'Unknown')}

## PERSONAL MEDICAL HISTORY
Previous breast biopsies: {questionnaire.get('previousBreastBiopsies', 'Unknown')}
Previous breast cancer: {questionnaire.get('previousBreastCancer', 'Unknown')}
Hormone replacement therapy: {questionnaire.get('hormoneReplacementTherapy', 'Unknown')}
Age at first menstrual period: {questionnaire.get('ageFirstPeriod', 'Unknown')}
Age at first live birth: {questionnaire.get('ageFirstBirth', 'Unknown')}
Menopausal status: {questionnaire.get('menopausalStatus', 'Unknown')}

## LIFESTYLE FACTORS
Alcohol consumption: {questionnaire.get('alcoholConsumption', 'Unknown')}
Smoking status: {questionnaire.get('smokingStatus', 'Unknown')}
Physical activity level: {questionnaire.get('physicalActivity', 'Unknown')}
BMI: {questionnaire.get('bmi', 'Unknown')}

## IMAGE ANALYSIS RESULTS
"""

    if image_predictions:
        prompt += f"""
Overall image prediction: {image_predictions.get('result_text', 'Not available')}
Ensemble probability: {image_predictions.get('ensemble_probability', 'N/A')}

Individual model predictions:
"""
        for model_name, prob in image_predictions.get("model_predictions", {}).items():
            if prob is not None:
                status = "Cancer" if prob > 0.5 else "Normal"
                prompt += f"- {model_name}: {status} (prob: {prob:.4f})\n"
    else:
        prompt += "No image analysis results.\n"

    if gene_predictions:
        prompt += f"""
## GENE EXPRESSION ANALYSIS
Predicted class: {gene_predictions.get('predicted_class', 'N/A')}
Probability: {gene_predictions.get('probability', 'N/A')}
"""
    else:
        prompt += "\n## GENE EXPRESSION ANALYSIS\nNo gene expression data.\n"

    prompt += """
## REPORT GENERATION INSTRUCTIONS
Provide a structured medical report containing:
1. Summary
2. Risk assessment
3. Imaging + genetics interpretation
4. Recommendations
5. Limitations
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code != 200:
            return f"Ollama Error: {response.status_code} - {response.text}"

        report_text = response.json().get("response", "")
        return report_text

    except Exception as e:
        return f"Error communicating with Ollama: {e}"
