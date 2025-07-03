document.addEventListener('DOMContentLoaded', function () {
    // --- DOM Elements ---
    const patientQuestionnaireForm = document.getElementById('patientQuestionnaireForm');
    const saveQuestionnaireButton = document.getElementById('saveQuestionnaireButton');
    const resetQuestionnaireButton = document.getElementById('resetQuestionnaireButton');

    const imageUpload = document.getElementById('imageUpload');
    const predictButton = document.getElementById('predictButton');
    const originalImagePlaceholder = document.getElementById('originalImagePlaceholder');
    const originalImage = document.getElementById('originalImage');
    const visualizationImagePlaceholder = document.getElementById('visualizationImagePlaceholder');
    const visualizationImage = document.getElementById('visualizationImage');
    const resultText = document.getElementById('resultText');
    const ensembleProbability = document.getElementById('ensembleProbability');
    const modelPredictionsContainer = document.getElementById('modelPredictionsContainer');

    const geneDataUpload = document.getElementById('geneDataUpload');
    const predictGeneButton = document.getElementById('predictGeneButton');
    const genePredictionResultText = document.getElementById('genePredictionResultText');
    const genePredictionProbability = document.getElementById('genePredictionProbability');

    const generateReportButton = document.getElementById('generateReportButton');
    const medicalReportContainer = document.getElementById('medicalReportContainer');
    const reportPlaceholder = document.getElementById('reportPlaceholder');
    const medicalReport = document.getElementById('medicalReport');
    const printReportButton = document.getElementById('printReportButton');
    const downloadReportButton = document.getElementById('downloadReportButton');

    const loadingSpinner = document.getElementById('loadingSpinner');

    // --- State Variables ---
    let currentPatientId = null;
    let currentQuestionnaireData = null;
    let currentImageResults = null;
    let currentGeneResults = null;
    let currentOriginalImageBase64 = null;
    let currentVisualizationImageBase64 = null;


    // --- Helper Functions ---
    function showSpinner() {
        if (loadingSpinner) loadingSpinner.style.display = 'flex';
    }

    function hideSpinner() {
        if (loadingSpinner) loadingSpinner.style.display = 'none';
    }

    function displayMessage(element, message, isError = false) {
        if (element) {
            element.textContent = message;
            element.style.color = isError ? 'red' : 'green';
        }
    }

    function resetImageDisplays() {
        if (originalImage) {
            originalImage.style.display = 'none';
            originalImage.src = '#';
        }
        if (originalImagePlaceholder) originalImagePlaceholder.style.display = 'flex';
        if (visualizationImage) {
            visualizationImage.style.display = 'none';
            visualizationImage.src = '#';
        }
        if (visualizationImagePlaceholder) visualizationImagePlaceholder.style.display = 'flex';
        if (resultText) resultText.textContent = 'N/A';
        if (ensembleProbability) ensembleProbability.textContent = 'N/A';
        if (modelPredictionsContainer) modelPredictionsContainer.innerHTML = '<p class="placeholder-text">No individual predictions yet.</p>';
    }

    function resetGeneResultsDisplay() {
        if (genePredictionResultText) genePredictionResultText.textContent = 'Predicted class malignancy status';
        if (genePredictionProbability) genePredictionProbability.textContent = 'N/A';
    }

    function resetReportDisplay() {
        if (medicalReport) medicalReport.style.display = 'none';
        if (medicalReport) medicalReport.innerHTML = '';
        if (reportPlaceholder) reportPlaceholder.style.display = 'block';
    }

    // --- Event Listeners ---

    // Questionnaire Form
    if (patientQuestionnaireForm) {
        patientQuestionnaireForm.addEventListener('submit', async function (event) {
            event.preventDefault();
            showSpinner();
            const formData = new FormData(patientQuestionnaireForm);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = value;
            });

            // Handle "No children" checkbox for ageFirstBirth
            if (data.noChildren === 'on') {
                data.ageFirstBirth = null; // Or a specific value like -1 if your backend expects it
            }
            delete data.noChildren; // Remove the checkbox value itself


            currentQuestionnaireData = data; // Store for report generation

            try {
                const response = await fetch('/submit_questionnaire', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data),
                });
                const result = await response.json();
                if (response.ok && result.success) {
                    currentPatientId = result.patient_id;
                    // Update Patient ID field if it was auto-generated or for confirmation
                    if (document.getElementById('patientId') && currentPatientId) {
                        document.getElementById('patientId').value = currentPatientId;
                    }
                    alert(`Patient information saved successfully. Patient ID: ${currentPatientId}`);
                    // Enable other sections if needed
                } else {
                    alert('Error saving questionnaire: ' + (result.error || 'Unknown error'));
                }
            } catch (error) {
                console.error('Error submitting questionnaire:', error);
                alert('An error occurred. Please try again.');
            } finally {
                hideSpinner();
            }
        });
    }

    if (resetQuestionnaireButton) {
        resetQuestionnaireButton.addEventListener('click', function() {
            if (patientQuestionnaireForm) patientQuestionnaireForm.reset();
            currentPatientId = null;
            currentQuestionnaireData = null;
            if (document.getElementById('patientId')) {
                 document.getElementById('patientId').value = ''; // Clear patient ID field
            }
            resetImageDisplays();
            resetGeneResultsDisplay();
            resetReportDisplay();
            alert('Questionnaire reset.');
        });
    }

    // Image Prediction
    if (predictButton && imageUpload) {
        predictButton.addEventListener('click', async function () {
            if (!currentPatientId && patientQuestionnaireForm.patientId.value) {
                currentPatientId = patientQuestionnaireForm.patientId.value;
            }

            if (!currentPatientId) {
                alert('Please save patient information or enter a Patient ID before predicting.');
                return;
            }
            if (!imageUpload.files || imageUpload.files.length === 0) {
                alert('Please select an image file.');
                return;
            }

            showSpinner();
            resetImageDisplays(); // Reset previous results

            const formData = new FormData();
            formData.append('file', imageUpload.files[0]);
            formData.append('patient_id', currentPatientId); // Send patient_id

            try {
                const response = await fetch('/predict_image', {
                    method: 'POST',
                    body: formData,
                });
                const result = await response.json();

                if (response.ok) {
                    currentImageResults = result; // Store for report generation
                    currentOriginalImageBase64 = result.original_image_base64;
                    currentVisualizationImageBase64 = result.visualization_image_base64;


                    if (originalImage && result.original_image_base64) {
                        originalImage.src = result.original_image_base64;
                        originalImage.style.display = 'block';
                        if (originalImagePlaceholder) originalImagePlaceholder.style.display = 'none';
                    }
                    if (visualizationImage && result.visualization_image_base64) {
                        visualizationImage.src = result.visualization_image_base64;
                        visualizationImage.style.display = 'block';
                        if (visualizationImagePlaceholder) visualizationImagePlaceholder.style.display = 'none';
                    }
                    if (resultText) resultText.textContent = result.result_text || 'N/A';
                    if (ensembleProbability) ensembleProbability.textContent = result.ensemble_probability !== null ? parseFloat(result.ensemble_probability).toFixed(4) : 'N/A';

                    if (modelPredictionsContainer && result.model_predictions) {
                        modelPredictionsContainer.innerHTML = ''; // Clear placeholder
                        const ul = document.createElement('ul');
                        for (const modelName in result.model_predictions) {
                            const prob = result.model_predictions[modelName];
                            const li = document.createElement('li');
                            li.textContent = `${modelName}: ${prob > 0.5 ? "Cancer" : "Normal"} (${prob !== null ? parseFloat(prob).toFixed(4) : 'N/A'})`;
                            ul.appendChild(li);
                        }
                        modelPredictionsContainer.appendChild(ul);
                    }
                } else {
                    alert('Error predicting image: ' + (result.error || 'Unknown error'));
                    if (resultText) resultText.textContent = `Error: ${result.error || 'Unknown error'}`;
                }
            } catch (error) {
                console.error('Error predicting image:', error);
                alert('An error occurred during image prediction.');
                if (resultText) resultText.textContent = 'Error: An unexpected error occurred.';
            } finally {
                hideSpinner();
            }
        });
    }
    
    if (imageUpload) {
        imageUpload.addEventListener('change', function(event) {
            if (event.target.files && event.target.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    if (originalImage) {
                        originalImage.src = e.target.result;
                        originalImage.style.display = 'block';
                    }
                    if (originalImagePlaceholder) {
                        originalImagePlaceholder.style.display = 'none';
                    }
                }
                reader.readAsDataURL(event.target.files[0]);
            } else {
                 resetImageDisplays(); // If no file is selected (e.g., user cancels dialog)
            }
        });
    }


    // Gene Prediction
    if (predictGeneButton && geneDataUpload) {
        predictGeneButton.addEventListener('click', async function () {
            if (!currentPatientId && patientQuestionnaireForm.patientId.value) {
                currentPatientId = patientQuestionnaireForm.patientId.value;
            }

            if (!currentPatientId) {
                alert('Please save patient information or enter a Patient ID before predicting.');
                return;
            }
            if (!geneDataUpload.files || geneDataUpload.files.length === 0) {
                alert('Please select a gene data file (CSV or TSV).');
                return;
            }

            showSpinner();
            resetGeneResultsDisplay(); // Reset previous results

            const formData = new FormData();
            formData.append('file', geneDataUpload.files[0]);
            formData.append('patient_id', currentPatientId); // Send patient_id

            try {
                const response = await fetch('/predict_gene', {
                    method: 'POST',
                    body: formData,
                });
                const result = await response.json();

                if (response.ok) {
                    currentGeneResults = result; // Store for report generation

                    if (genePredictionResultText) {
                        genePredictionResultText.textContent = result.predicted_class || 'Predicted gene malignancy status';
                    }
                    if (genePredictionProbability) {
                        genePredictionProbability.textContent = result.probability !== null ? 
                            parseFloat(result.probability).toFixed(4) : 'N/A';
                    }
                    
                    alert('Gene prediction completed successfully.');
                } else {
                    alert('Error during gene prediction: ' + (result.error || 'Unknown error'));
                    if (genePredictionResultText) genePredictionResultText.textContent = 'Error';
                    if (genePredictionProbability) genePredictionProbability.textContent = 'N/A';
                }
            } catch (error) {
                console.error('Error during gene prediction:', error);
                alert('An error occurred during gene prediction. Please try again.');
                if (genePredictionResultText) genePredictionResultText.textContent = 'Error';
                if (genePredictionProbability) genePredictionProbability.textContent = 'N/A';
            } finally {
                hideSpinner();
            }
        });
    }

    // Generate Report
    if (generateReportButton) {
        generateReportButton.addEventListener('click', async function () {
            if (!currentPatientId && patientQuestionnaireForm.patientId.value) {
                currentPatientId = patientQuestionnaireForm.patientId.value;
            }
            
            if (!currentPatientId) {
                alert('Please save patient information or enter a Patient ID first.');
                return;
            }
            // Ensure questionnaire data is current if form has been modified without saving
            if (patientQuestionnaireForm) {
                 const formData = new FormData(patientQuestionnaireForm);
                 const data = {};
                 formData.forEach((value, key) => { data[key] = value; });
                 if (data.noChildren === 'on') { data.ageFirstBirth = null; }
                 delete data.noChildren;
                 currentQuestionnaireData = data;
            }


            if (!currentQuestionnaireData) {
                alert('Please complete and save the patient questionnaire first.');
                return;
            }

            showSpinner();
            resetReportDisplay();

            const reportPayload = {
                patientId: currentPatientId, // Make sure this is being sent
                questionnaireData: currentQuestionnaireData,
                imageResults: currentImageResults,
                geneResults: currentGeneResults,
                originalImageBase64: currentOriginalImageBase64, // Send base64 images for report context
                visualizationImageBase64: currentVisualizationImageBase64
            };
            
            console.log("Sending to /generate_report:", JSON.stringify(reportPayload, null, 2));


            try {
                const response = await fetch('/generate_report', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(reportPayload),
                });
                const result = await response.json();
                if (response.ok) {
                    if (medicalReport && result.report_html) {
                        medicalReport.innerHTML = result.report_html; // Display HTML report
                        medicalReport.style.display = 'block';
                        if (reportPlaceholder) reportPlaceholder.style.display = 'none';
                        if (printReportButton) printReportButton.style.display = 'inline-block';
                        if (downloadReportButton) downloadReportButton.style.display = 'inline-block';
                    } else if (medicalReport && result.report) { // Fallback to text if HTML not available
                        medicalReport.innerHTML = `<pre>${result.report}</pre>`;
                        medicalReport.style.display = 'block';
                        if (reportPlaceholder) reportPlaceholder.style.display = 'none';
                         if (printReportButton) printReportButton.style.display = 'inline-block';
                        if (downloadReportButton) downloadReportButton.style.display = 'inline-block';
                    }
                    alert('Medical report generated successfully.');
                } else {
                    alert('Error generating report: ' + (result.error || 'Unknown error'));
                    if (medicalReport) medicalReport.innerHTML = `<p style="color:red;">Error generating report: ${result.error || 'Unknown error'}</p>`;
                    if (medicalReport) medicalReport.style.display = 'block';
                    if (reportPlaceholder) reportPlaceholder.style.display = 'none';
                }
            } catch (error) {
                console.error('Error generating report:', error);
                alert('An error occurred while generating the report.');
                 if (medicalReport) medicalReport.innerHTML = `<p style="color:red;">An unexpected error occurred.</p>`;
                 if (medicalReport) medicalReport.style.display = 'block';
                 if (reportPlaceholder) reportPlaceholder.style.display = 'none';
            } finally {
                hideSpinner();
            }
        });
    }

    // Print Report
    if (printReportButton) {
        printReportButton.addEventListener('click', function() {
            const reportContent = medicalReport ? medicalReport.innerHTML : null;
            if (reportContent) {
                const printWindow = window.open('', '_blank');
                printWindow.document.write('<html><head><title>Medical Report</title>');
                // Optional: Link to your stylesheet for better print formatting
                // printWindow.document.write('<link rel="stylesheet" href="/static/style.css" type="text/css" />');
                // Basic print styles
                printWindow.document.write(`
                    <style>
                        body { font-family: sans-serif; margin: 20px; }
                        h1, h2, h3 { color: #333; }
                        .report-section { margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
                        img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                        pre { white-space: pre-wrap; background-color: #f8f9fa; padding: 10px; border-radius: 4px; }
                    </style>
                `);
                printWindow.document.write('</head><body>');
                printWindow.document.write(reportContent);
                printWindow.document.write('</body></html>');
                printWindow.document.close();
                printWindow.focus(); // Necessary for some browsers
                // Timeout to ensure content is loaded before printing
                setTimeout(() => {
                    printWindow.print();
                    printWindow.close();
                }, 250);

            } else {
                alert('No report content to print.');
            }
        });
    }

    // Download Report (PDF - This will trigger a server endpoint that generates PDF)
    if (downloadReportButton) {
        downloadReportButton.addEventListener('click', function() {
            if (!currentPatientId) {
                alert('Patient ID is not available. Cannot download report.');
                return;
            }
            // This assumes you have an endpoint like '/download_report_pdf/<patient_id>'
            // or you pass the report content to a generic PDF generation endpoint.
            // For simplicity, let's assume the latest report for the patient.
            // The actual PDF generation must happen on the server.
            // This button will just navigate to the download URL.
            
            // Option 1: If report ID is known and server can generate PDF from it
            // const reportId = ...; // if you get a report_id from generate_report response
            // window.location.href = `/download_report_pdf_by_id/${reportId}`;

            // Option 2: If server generates PDF based on patient_id (latest report)
            window.location.href = `/download_report_pdf/${currentPatientId}`;
            
            // Option 3: If you want to send current report HTML to server for PDF conversion
            /*
            const reportHtmlContent = medicalReport ? medicalReport.innerHTML : null;
            if (reportHtmlContent) {
                showSpinner();
                fetch('/generate_pdf_from_html', { // You'd need to create this endpoint
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ html_content: reportHtmlContent, patient_id: currentPatientId })
                })
                .then(response => {
                    if (!response.ok) throw new Error('PDF generation failed');
                    return response.blob();
                })
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = `medical_report_${currentPatientId || 'unknown'}.pdf`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                })
                .catch(error => {
                    console.error('Error downloading PDF:', error);
                    alert('Error downloading PDF: ' + error.message);
                })
                .finally(hideSpinner);
            } else {
                alert('No report content to download.');
            }
            */
        });
    }

    // Initial state for buttons that depend on report generation
    if (printReportButton) printReportButton.style.display = 'none';
    if (downloadReportButton) downloadReportButton.style.display = 'none';

    // Handle "No children" checkbox logic
    const noChildrenCheckbox = document.getElementById('noChildren');
    const ageFirstBirthInput = document.getElementById('ageFirstBirth');

    if (noChildrenCheckbox && ageFirstBirthInput) {
        noChildrenCheckbox.addEventListener('change', function() {
            if (this.checked) {
                ageFirstBirthInput.value = '';
                ageFirstBirthInput.disabled = true;
                ageFirstBirthInput.required = false;
            } else {
                ageFirstBirthInput.disabled = false;
                ageFirstBirthInput.required = true; // Or false, depending on your form validation needs
            }
        });
        // Initial check in case the page loads with it checked (e.g. form repopulation)
        if (noChildrenCheckbox.checked) {
            ageFirstBirthInput.value = '';
            ageFirstBirthInput.disabled = true;
            ageFirstBirthInput.required = false;
        }
    }

});