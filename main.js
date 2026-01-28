// -------------------- TAB SWITCHING --------------------
const tabs = document.querySelectorAll('.tab-btn');
const contents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');

    contents.forEach(c => c.classList.remove('active'));
    const target = document.getElementById(tab.dataset.tab);
    target.classList.add('active');
  });
});

// -------------------- IMAGE UPLOAD & PREVIEW --------------------
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('imagePreview');
const predictionBox = document.getElementById('prediction-box');

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Show preview
  const reader = new FileReader();
  reader.onload = (ev) => {
    preview.src = ev.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);

  // Show prediction box and set initial text
  predictionBox.style.display = 'block';
  predictionBox.innerHTML = "Predicting...";

  // Send to API for prediction
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();

    if (data.error) {
      predictionBox.innerHTML = `<span style="color:red">${data.error}</span>`;
    } else {
      // Calculate confidence in the predicted label
      const confidence = data.label === "malignant" 
        ? data.probability * 100 
        : (1 - data.probability) * 100;
      
      // Add class for color-coding
      const cls = data.label === "benign" ? "prediction-benign" : "prediction-malignant";
      predictionBox.className = cls;

      predictionBox.innerHTML = `
        <strong>Label:</strong> ${data.label} <br/>
        <strong>Confidence:</strong> ${confidence.toFixed(1)}%
      `;
    }
  } catch(err) {
    predictionBox.innerHTML = `<span style="color:red">Error: ${err.message}</span>`;
  }
});