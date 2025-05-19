const loadingMessage = document.getElementById('loading-message');
const errorMessageElement = document.getElementById('error-message');
const predictionFormDiv = document.getElementById('prediction-form');
const houseForm = document.getElementById('house-form');
const predictButton = document.getElementById('predict-button');
const resultArea = document.getElementById('result-area');
const predictionOutput = document.getElementById('prediction-output');

let pyodide = null;
let modelBytes = null;
let pythonLogicCode = null;

// Segédfüggvény a hibák kijelzésére
function displayError(message) {
    console.error(message);
    errorMessageElement.innerText = `Hiba: ${message}. Kérjük, ellenőrizd a konzolt (F12) a részletekért.`;
    errorMessageElement.style.display = 'block';
    loadingMessage.style.display = 'none';
    if (predictButton) predictButton.disabled = false;
}

async function setupPyodide() {
    try {
        loadingMessage.innerText = "Pyodide betöltése...";
        pyodide = await loadPyodide();
        loadingMessage.innerText = "Python csomagok (pandas, numpy, scikit-learn, joblib) betöltése...";
        await pyodide.loadPackage(['pandas', 'numpy', 'scikit-learn', 'joblib']);

        loadingMessage.innerText = "Modell fájl letöltése (house_price_model.pkl)...";
        const modelResponse = await fetch('house_price_model.pkl');
        if (!modelResponse.ok) {
            throw new Error(`Nem sikerült letölteni a model.pkl fájlt: ${modelResponse.statusText}`);
        }
        modelBytes = await modelResponse.arrayBuffer();

        loadingMessage.innerText = "Python logika letöltése (python_logic.py)...";
        const pythonLogicResponse = await fetch('python_logic.py');
        if (!pythonLogicResponse.ok) {
            throw new Error(`Nem sikerült letölteni a python_logic.py fájlt: ${pythonLogicResponse.statusText}`);
        }
        pythonLogicCode = await pythonLogicResponse.text();
        
        // Load the Python code
        await pyodide.runPythonAsync(pythonLogicCode);

        // Convert model bytes to Python format
        const modelBytesArray = new Uint8Array(modelBytes);
        pyodide.globals.set('model_bytes_py', modelBytesArray);
        
        // Make sure the bytes are properly stored
        await pyodide.runPythonAsync(`
import io
model_bytes_py = bytes(model_bytes_py)
print("Model bytes loaded, size:", len(model_bytes_py))
        `);

        loadingMessage.style.display = 'none';
        predictionFormDiv.style.display = 'block';
        console.log("Pyodide és a modell sikeresen betöltve.");

    } catch (error) {
        displayError(`Inicializálási hiba: ${error.message}`);
    }
}

houseForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!pyodide || !modelBytes || !pythonLogicCode) {
        displayError("Pyodide vagy a modell még nem töltődött be.");
        return;
    }

    predictButton.disabled = true;
    resultArea.style.display = 'none';
    errorMessageElement.style.display = 'none';
    loadingMessage.innerText = "Predikció feldolgozása...";
    loadingMessage.style.display = 'block';

    try {
        const formData = new FormData(houseForm);
        const inputData = {};
        
        // Process form data
        for (let [key, value] of formData.entries()) {
            // Convert numeric form fields to numbers
            const numericFields = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 
                                  'YrSold', 'FullBath', 'GarageArea', 'LotArea', 'LotFrontage'];
            
            if (value === '' || value === null) {
                // For empty inputs, use null (will be converted to NaN in Python)
                inputData[key] = null;
            } else if (numericFields.includes(key)) {
                // Convert numeric fields to numbers
                inputData[key] = parseFloat(value);
            } else {
                // Keep other values as strings
                inputData[key] = value;
            }
        }
        
        console.log("Input data:", inputData);
        
        // Pass input data to Python
        pyodide.globals.set('input_data_py', pyodide.toPy(inputData));
        
        // First, execute get_prediction and store result in a Python variable
        await pyodide.runPythonAsync(`
try:
    import numpy as np
    prediction_value = get_prediction(input_data_py, model_bytes_py)
    print("Prediction result:", prediction_value, type(prediction_value))
except Exception as e:
    import traceback
    print("Python error:", str(e))
    print("Python traceback:", traceback.format_exc())
    prediction_value = None
`);

        // Now retrieve the prediction value from Python
        let predictionResult;
        try {
            predictionResult = await pyodide.runPythonAsync("float(prediction_value) if prediction_value is not None else None");
            console.log("Python result:", predictionResult);
            
            if (predictionResult === undefined || predictionResult === null) {
                throw new Error("A predikció üres vagy érvénytelen eredményt adott vissza.");
            }
            
            // Display the prediction
            predictionOutput.innerText = predictionResult.toLocaleString('hu-HU', { 
                style: 'currency', 
                currency: 'USD', 
                minimumFractionDigits: 0, 
                maximumFractionDigits: 0 
            });
            
            resultArea.style.display = 'block';
            
        } catch (pyError) {
            console.error("Python execution error:", pyError);
            displayError(`Python végrehajtási hiba: ${pyError.message || pyError}`);
        }

    } catch (error) {
        displayError(`Predikciós hiba: ${error.message}`);
    } finally {
        predictButton.disabled = false;
        loadingMessage.style.display = 'none';
    }
});

// Indítsuk el a Pyodide beállítását az oldal betöltésekor
setupPyodide();