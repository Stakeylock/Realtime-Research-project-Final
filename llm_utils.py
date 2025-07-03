import os
import json
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables. LLM features will not work.")

# Add this function to your existing llm_utils.py file

def format_report_as_html(report_text, patient_data, image_predictions, gene_predictions):
    """
    Format the LLM-generated report as HTML for display
    
    Args:
        report_text (str): The raw report text from the LLM
        patient_data (dict): Patient questionnaire data
        image_predictions (dict): Results from image-based models
        gene_predictions (dict): Results from gene expression model
        
    Returns:
        str: HTML-formatted report
    """
    # Basic HTML formatting - convert markdown-like syntax to HTML
    html = "<div class='medical-report'>"
    
    # Add report header
    html += "<div class='report-header'>"
    html += "<h1>Breast Cancer Risk Assessment Report</h1>"
    html += f"<p class='report-date'>Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>"
    html += "</div>"
    
    # Format the report text - convert simple markdown to HTML
    paragraphs = report_text.split('\n\n')
    for p in paragraphs:
        if p.startswith('# '):
            # Main heading
            html += f"<h2>{p[2:]}</h2>"
        elif p.startswith('## '):
            # Subheading
            html += f"<h3>{p[3:]}</h3>"
        elif p.startswith('### '):
            # Sub-subheading
            html += f"<h4>{p[4:]}</h4>"
        elif p.startswith('- '):
            # List item
            items = [item[2:] for item in p.split('\n- ')]
            html += "<ul>"
            for item in items:
                html += f"<li>{item}</li>"
            html += "</ul>"
        elif p.startswith('**'):
            # Bold paragraph (likely important)
            if p.endswith('**'):
                content = p[2:-2]
                html += f"<p class='important'><strong>{content}</strong></p>"
            else:
                html += f"<p>{p}</p>"
        else:
            # Regular paragraph
            html += f"<p>{p}</p>"
    
    html += "</div>"
    return html

def generate_medical_report(patient_data, image_predictions, gene_predictions):
    """
    Generate a comprehensive medical report using Llama 3.3 via Groq API
    
    Args:
        patient_data (dict): Patient questionnaire data
        image_predictions (dict): Results from image-based models
        gene_predictions (dict): Results from gene expression model
        
    Returns:
        str: Formatted medical report
    """
    if not GROQ_API_KEY:
        return "Error: Groq API key not configured. Please check your .env file."
    
    # Format the data for the prompt
    questionnaire = patient_data.get('questionnaire', {})
    
    # Create a structured prompt for the LLM
    prompt = f"""
You are an expert medical AI assistant helping to generate a comprehensive breast cancer risk assessment report.
Please analyze the following patient data and provide a detailed medical report.

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

    # Add image prediction results if available
    if image_predictions:
        prompt += f"""
Overall image prediction: {image_predictions.get('result_text', 'Not available')}
Ensemble model probability: {image_predictions.get('ensemble_probability', 'Not available')}

Individual model predictions:
"""
        model_preds = image_predictions.get('model_predictions', {})
        for model_name, prob in model_preds.items():
            if prob is not None:
                prediction = "Cancer" if prob > 0.5 else "Normal"
                prompt += f"- {model_name}: {prediction} (probability: {prob:.4f})\n"
    else:
        prompt += "No image analysis results available.\n"

    # Add gene prediction results if available
    if gene_predictions:
        prompt += f"""
## GENE EXPRESSION ANALYSIS
Predicted class: {gene_predictions.get('predicted_class', 'Not available')}
Probability: {gene_predictions.get('probability', 'Not available')}
"""
    else:
        prompt += "\n## GENE EXPRESSION ANALYSIS\nNo gene expression data available.\n"

    # Instructions for the report
    prompt += """
## REPORT GENERATION INSTRUCTIONS
Based on the above information, please generate a comprehensive medical report that includes:

1. SUMMARY: A brief summary of the key findings
2. RISK ASSESSMENT: An analysis of the patient's breast cancer risk based on all available data
3. INTERPRETATION: Clinical interpretation of the imaging and genetic findings
4. RECOMMENDATIONS: Suggested next steps, including any follow-up tests or consultations
5. LIMITATIONS: Any limitations in the current assessment that should be considered

The report should be formatted in a professional medical style with appropriate headings and sections.
Use medical terminology but ensure it remains understandable to healthcare providers.
Be thorough but concise, focusing on clinically relevant information.
"""

    # Call the Groq API with Llama 3.3
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3-70b-8192",  # Using Llama 3.3 70B model
            "messages": [
                {"role": "system", "content": "You are an expert medical AI assistant specializing in breast cancer risk assessment and reporting."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Lower temperature for more consistent, factual responses
            "max_tokens": 4000,  # Allow for a detailed report
            "top_p": 0.9
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            report_text = result["choices"][0]["message"]["content"]
            return report_text
        else:
            error_message = f"API Error: {response.status_code} - {response.text}"
            print(error_message)
            return f"Error generating report: {error_message}"
            
    except Exception as e:
        error_message = f"Exception during report generation: {str(e)}"
        print(error_message)
        return error_message