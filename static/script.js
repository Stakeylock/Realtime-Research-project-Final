document.addEventListener('DOMContentLoaded', function () {
    // --- DOM Elements ---
    const patientQuestionnaireForm = document.getElementById('patientQuestionnaireForm');
    const saveQuestionnaireButton = document.getElementById('saveQuestionnaireButton');
    const resetQuestionnaireButton = document.getElementById('resetQuestionnaireButton');

    const imageUpload = document.getElementById('imageUpload');
    const predictButton = document.getElementById('predictButton');
    const validateButton = document.getElementById('validateButton'); // NEW: Validation button
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

    // NEW: Validation section elements
    const validationSection = document.getElementById('validation-section');
    const trueLabelDisplay = document.getElementById('true-label-display');
    const validationResultsOutput = document.getElementById('validation-results-output');
    const metricsComparisonGraph = document.getElementById('metrics-comparison-graph');


    // --- State Variables ---
    let currentPatientId = null;
    let currentQuestionnaireData = null;
    let currentImageResults = null;
    let currentGeneResults = null;
    let currentOriginalImageBase64 = null;
    let currentVisualizationImageBase64 = null;
    let currentReportId = null; 

    // UPDATED: Static Overall Metrics Baseline (from your provided data)
    const OVERALL_METRICS_BASELINE = {
        'LBP': { // Local Binary Pattern - Original Image
            'accuracy': 0.486423,
            'precision': 0.486423,
            'recall': 0.486423,
            'f1_score': 0.486423
        },
        'LBPN': { // Local Binary Pattern - Negative Transformer
            'accuracy': 0.486423,
            'precision': 0.486423,
            'recall': 0.486423,
            'f1_score': 0.486423
        },
        'LBPAHE': { // Local Binary Pattern - Adaptive Histogram Equalization
            'accuracy': 0.486423,
            'precision': 0.486423,
            'recall': 0.486423,
            'f1_score': 0.486423
        },
        'Sift': { // Scale Invariant Feature Transform (SIFT) - Original Image
            'accuracy': 0.795750,
            'precision': 0.795750,
            'recall': 0.795750,
            'f1_score': 0.795750
        },
        'SiftN': { // Scale Invariant Feature Transform (SIFT) - Negative Transformer
            'accuracy': 0.684770,
            'precision': 0.684770,
            'recall': 0.684770,
            'f1_score': 0.684770
        },
        'SiftAHE': { // Scale Invariant Feature Transform (SIFT) - Adaptive Histogram Equalization
            'accuracy': 0.785124,
            'precision': 0.785124,
            'recall': 0.785124,
            'f1_score': 0.785124
        },
        'Hog': { // Histogram of Oriented Gradients (HOG) - Original Image
            'accuracy': 0.787485,
            'precision': 0.787485,
            'recall': 0.787485,
            'f1_score': 0.787485
        },
        'HogN': { // Histogram of Oriented Gradients (HOG) - Negative Transformer
            'accuracy': 0.775679,
            'precision': 0.775679,
            'recall': 0.775679,
            'f1_score': 0.775679
        },
        'HogAHE': { // Histogram of Oriented Gradients (HOG) - Adaptive Histogram Equalization
            'accuracy': 0.783943,
            'precision': 0.783943,
            'recall': 0.783943,
            'f1_score': 0.783943
        },
        'ResNet': { // ResNet model - Original Image
            'accuracy': 0.931523,
            'precision': 0.931523,
            'recall': 0.931523,
            'f1_score': 0.931523
        },
        'ResNetN': { // ResNet model - Negative Transformer
            'accuracy': 0.958678,
            'precision': 0.958678,
            'recall': 0.958678,
            'f1_score': 0.958678
        },
        'ResNetAHE': { // ResNet model - Adaptive Histogram Equalization
            'accuracy': 0.971665,
            'precision': 0.971665,
            'recall': 0.971665,
            'f1_score': 0.971665
        }
    };


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
        currentReportId = null; 
    }

    // NEW: Function to reset validation display
    function resetValidationDisplay() {
        if (trueLabelDisplay) trueLabelDisplay.textContent = 'Not Checked';
        if (validationResultsOutput) {
            validationResultsOutput.innerHTML = '<p class="placeholder-text">Upload an image with a label (e.g., \'image_benign.png\') and click \'Validate with Label\' to see metrics.</p>';
        }
        if (metricsComparisonGraph) {
            metricsComparisonGraph.innerHTML = '<h3>Performance Comparison (Accuracy)</h3><p class="placeholder-text">Graph will appear here after validation.</p>';
        }
        if (validationSection) {
            validationSection.style.display = 'block'; // Keep validation section visible with placeholder
        }
    }

    // NEW: Function to display validation metrics
    function displayValidationMetrics(validationMetrics) {
        if (!validationSection || !trueLabelDisplay || !validationResultsOutput || !metricsComparisonGraph) return;

        validationSection.style.display = 'block'; // Ensure the section is visible

        if (validationMetrics.true_label_found) {
            trueLabelDisplay.textContent = `${validationMetrics.true_label} (${validationMetrics.true_label === 1 ? 'Malignant' : 'Benign'})`;
            validationResultsOutput.innerHTML = ''; // Clear previous content

            // Create a combined table for Overall and Current Performance
            let combinedPerformanceHtml = '<h3>Performance Metrics (Overall vs. Current)</h3>';
            combinedPerformanceHtml += '<table class="metrics-table"><thead><tr><th>Model</th><th>Metric</th><th>Overall Baseline</th><th>Current Image</th></tr></thead><tbody>';

            const modelsToCompare = Object.keys(validationMetrics.performance);
            // UPDATED: Only display Accuracy in the table
            const metricsToDisplayInTable = ['accuracy']; 

            modelsToCompare.forEach(modelName => {
                const currentMetrics = validationMetrics.performance[modelName];
                const overallMetrics = OVERALL_METRICS_BASELINE[modelName] || {}; // Get baseline or empty object

                metricsToDisplayInTable.forEach((metric, index) => {
                    const overallValue = overallMetrics[metric] !== undefined ? (typeof overallMetrics[metric] === 'number' ? overallMetrics[metric].toFixed(4) : overallMetrics[metric]) : 'N/A';
                    const currentValue = currentMetrics[metric] !== null ? currentMetrics[metric].toFixed(4) : 'N/A';
                    
                    combinedPerformanceHtml += `<tr>
                        ${index === 0 ? `<td rowspan="${metricsToDisplayInTable.length}">${modelName}</td>` : ''}
                        <td>${metric.replace('_', ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</td>
                        <td>${overallValue}</td>
                        <td>${currentValue}</td>
                    </tr>`;
                });
            });
            combinedPerformanceHtml += '</tbody></table>';
            validationResultsOutput.innerHTML += combinedPerformanceHtml;


            // Trustability Metrics Table (unchanged)
            let trustabilityHtml = '<h3>Trustability Metrics</h3>';
            trustabilityHtml += '<table class="metrics-table"><thead><tr><th>Model</th><th>Predicted Class</th><th>Is Correct</th><th>Trust Score</th></tr></thead><tbody>';
            for (const modelName in validationMetrics.trustability) {
                const metrics = validationMetrics.trustability[modelName];
                const predictedClassLabel = metrics.predicted_class === 1 ? 'Malignant' : 'Benign';
                const isCorrectClass = metrics.is_correct ? 'metric-correct' : 'metric-incorrect';
                trustabilityHtml += `<tr>
                    <td>${modelName}</td>
                    <td><span class="${predictedClassLabel.toLowerCase()}">${predictedClassLabel}</span></td>
                    <td class="${isCorrectClass}">${metrics.is_correct ? 'Yes' : 'No'}</td>
                    <td>${metrics.trust_score}</td>
                </tr>`;
            }
            trustabilityHtml += '</tbody></table>';
            validationResultsOutput.innerHTML += trustabilityHtml;

            // Call graph drawing function
            drawComparisonGraph(OVERALL_METRICS_BASELINE, validationMetrics.performance);

        } else {
            trueLabelDisplay.textContent = 'Not Found in Filename';
            validationResultsOutput.innerHTML = '<p class="placeholder-text">No true label could be extracted from the image filename. Performance and trustability metrics cannot be calculated.</p>';
            metricsComparisonGraph.innerHTML = '<h3>Performance Comparison (Accuracy)</h3><p class="placeholder-text">Graph requires a true label to be present in the filename.</p>';
        }
    }

    // NEW: Function to draw a simple comparison bar graph (now only for Accuracy)
    function drawComparisonGraph(overallMetrics, currentMetrics) {
        if (!metricsComparisonGraph) return;

        metricsComparisonGraph.innerHTML = '<h3>Performance Comparison (Accuracy)</h3>'; // Updated title
        const graphContainer = document.createElement('div');
        graphContainer.className = 'bar-chart-container';
        metricsComparisonGraph.appendChild(graphContainer);

        const models = Object.keys(currentMetrics); // Use models that actually had current predictions
        const metricsToGraph = ['accuracy']; // UPDATED: Only graph Accuracy

        models.forEach(modelName => {
            const modelBarGroup = document.createElement('div');
            modelBarGroup.className = 'model-bar-group';
            
            const modelLabel = document.createElement('div');
            modelLabel.className = 'model-label';
            modelLabel.textContent = modelName;
            modelBarGroup.appendChild(modelLabel);

            metricsToGraph.forEach(metric => {
                const overallValue = overallMetrics[modelName] && typeof overallMetrics[modelName][metric] === 'number' ? overallMetrics[modelName][metric] : 0;
                const currentValue = currentMetrics[modelName] && typeof currentMetrics[modelName][metric] === 'number' ? currentMetrics[modelName][metric] : 0;

                const barWrapper = document.createElement('div');
                barWrapper.className = 'bar-wrapper';
                barWrapper.innerHTML = `<div class="metric-label">${metric.replace('_', ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</div>`;

                const overallBar = document.createElement('div');
                overallBar.className = 'bar overall-bar';
                overallBar.style.width = `${(overallValue * 100).toFixed(0)}%`;
                overallBar.textContent = `${(overallValue * 100).toFixed(1)}%`;
                barWrapper.appendChild(overallBar);

                const currentBar = document.createElement('div');
                currentBar.className = 'bar current-bar';
                currentBar.style.width = `${(currentValue * 100).toFixed(0)}%`;
                currentBar.textContent = `${(currentValue * 100).toFixed(1)}%`;
                barWrapper.appendChild(currentBar);

                modelBarGroup.appendChild(barWrapper);
            });
            graphContainer.appendChild(modelBarGroup);
        });

        // Add a legend
        const legend = document.createElement('div');
        legend.className = 'graph-legend';
        legend.innerHTML = `
            <span class="legend-item"><span class="legend-color overall-color"></span> Overall Baseline</span>
            <span class="legend-item"><span class="legend-color current-color"></span> Current Image</span>
        `;
        metricsComparisonGraph.appendChild(legend);
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
            resetValidationDisplay(); // NEW: Reset validation display
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
            resetValidationDisplay(); // NEW: Reset validation display

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
                            const predictedClass = prob > 0.5 ? "Malignant" : "Benign";
                            const classColor = predictedClass === "Malignant" ? "result-cancer" : "result-normal";
                            li.innerHTML = `${modelName}: <span class="${classColor}">${predictedClass}</span> (${prob !== null ? parseFloat(prob).toFixed(4) : 'N/A'})`;
                            ul.appendChild(li);
                        }
                        modelPredictionsContainer.appendChild(ul);
                    }

                    // NEW: Display validation metrics if available
                    if (result.validation_metrics) {
                        displayValidationMetrics(result.validation_metrics);
                    } else {
                        resetValidationDisplay(); // Ensure it's reset if no metrics are returned
                    }

                } else {
                    alert('Error predicting image: ' + (result.error || 'Unknown error'));
                    if (resultText) resultText.textContent = `Error: ${result.error || 'Unknown error'}`;
                    resetValidationDisplay(); // Also reset validation on error
                }
            } catch (error) {
                console.error('Error predicting image:', error);
                alert('An error occurred during image prediction.');
                if (resultText) resultText.textContent = 'Error: An unexpected error occurred.';
                resetValidationDisplay(); // Also reset validation on error
            } finally {
                hideSpinner();
            }
        });
    }

    // NEW: Validate Button Event Listener
    if (validateButton && imageUpload) {
        validateButton.addEventListener('click', async function () {
            if (!imageUpload.files || imageUpload.files.length === 0) {
                alert('Please select an image file to validate.');
                return;
            }

            showSpinner();
            resetImageDisplays(); // Reset previous prediction results
            resetValidationDisplay(); // Always reset validation display for a fresh start

            const formData = new FormData();
            formData.append('mammogram', imageUpload.files[0]); // Use 'mammogram' as the key as per server.py

            try {
                const response = await fetch('/validate', { // Call the new /validate endpoint
                    method: 'POST',
                    body: formData,
                });
                const result = await response.json();

                if (response.ok) {
                    // Display prediction results (optional, but good for context)
                    if (result.predictions) {
                        if (modelPredictionsContainer) {
                            modelPredictionsContainer.innerHTML = ''; // Clear placeholder
                            const ul = document.createElement('ul');
                            for (const modelName in result.predictions) {
                                const prob = result.predictions[modelName];
                                const li = document.createElement('li');
                                const predictedClass = prob > 0.5 ? "Malignant" : "Benign";
                                const classColor = predictedClass === "Malignant" ? "result-cancer" : "result-normal";
                                li.innerHTML = `${modelName}: <span class="${classColor}">${predictedClass}</span> (${prob !== null ? parseFloat(prob).toFixed(4) : 'N/A'})`;
                                ul.appendChild(li);
                            }
                            modelPredictionsContainer.appendChild(ul);
                        }
                    }

                    // Display validation metrics
                    if (result.validation) {
                        displayValidationMetrics(result.validation);
                    } else {
                        resetValidationDisplay(); // Fallback
                    }

                } else {
                    alert('Error during validation: ' + (result.error || 'Unknown error'));
                    resetValidationDisplay(); // Reset on error
                }
            } catch (error) {
                console.error('Error during validation:', error);
                alert('An error occurred during validation. Please try again.');
                resetValidationDisplay(); // Reset on error
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
                 resetValidationDisplay(); // Also reset validation if image is cleared
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
                    currentReportId = result.report_id; 
                    console.log('Generated Report ID:', currentReportId); 
                    console.log('Generated Patient ID:', result.patient_id); 

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
            if (!currentReportId) { 
                alert('Report ID is not available. Please generate the report first.'); 
                return; 
            }

            // This assumes your server.py has the route: @app.route('/download_report/<patient_id>/<report_id>')
            window.location.href = `/download_report/${currentPatientId}/${currentReportId}`; 

            // The commented out "Option 3" is a more complex client-side PDF generation,
            // which you don't seem to be using, so we stick to the server-side download.
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
