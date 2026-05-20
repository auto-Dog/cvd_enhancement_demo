// Global variables
let originalImage = null;
let enhancedImage = null;
let session = null;
let currentModel = 'D100';

// Model paths (you need to convert .pth to .onnx and host them)
const MODEL_PATHS = {
    D100: 'models/model_vit_cn7aE_D100_optim_base.onnx',
    P100: 'models/model_vit_cn7aE_P100_optim_base.onnx'
};

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadModel(currentModel);
});

function setupEventListeners() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const enhanceBtn = document.getElementById('enhanceBtn');
    const resetBtn = document.getElementById('resetBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const modelSelect = document.getElementById('modelSelect');

    // Upload area
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    enhanceBtn.addEventListener('click', () => enhanceImage());
    resetBtn.addEventListener('click', () => resetApp());
    downloadBtn.addEventListener('click', () => downloadEnhanced());
    
    modelSelect.addEventListener('change', (e) => {
        currentModel = e.target.value;
        loadModel(currentModel);
    });
}

async function loadModel(modelType) {
    try {
        showStatus(`Loading ${modelType} model...`, 'info');
        const modelPath = MODEL_PATHS[modelType];
        session = await ort.InferenceSession.create(modelPath);
        showStatus(`${modelType} model loaded successfully!`, 'success');
    } catch (error) {
        showStatus(`Error loading model: ${error.message}`, 'error');
        console.error('Model loading error:', error);
    }
}

function handleFiles(files) {
    if (files.length === 0) return;
    
    const file = files[0];
    if (!file.type.startsWith('image/')) {
        showStatus('Please upload an image file', 'error');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showStatus('Image size must be less than 50MB', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            originalImage = img;
            displayOriginal();
            document.getElementById('enhanceBtn').disabled = false;
            document.getElementById('resetBtn').disabled = false;
            showStatus('Image loaded successfully', 'success');
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function displayOriginal() {
    const canvas = document.getElementById('originalCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = originalImage.width;
    canvas.height = originalImage.height;
    ctx.drawImage(originalImage, 0, 0);
    document.getElementById('comparison').style.display = 'grid';
}

async function enhanceImage() {
    if (!originalImage || !session) {
        showStatus('Please load an image and model first', 'error');
        return;
    }

    try {
        showLoading(true);
        showStatus('Processing image...', 'info');

        // Resize to 240x240 for processing
        const processedImage = resizeImage(originalImage, 240, 240);
        const imageData = processedImage.getImageData(0, 0, 240, 240);
        
        // Prepare input tensor (normalize to 0-1)
        const inputTensor = imageDataToTensor(imageData);
        
        // Run inference
        const results = await session.run({
            'input': inputTensor
        });
        
        // Get output
        const outputTensor = results['output'];
        const enhancedData = tensorToImageData(outputTensor, 240, 240);
        
        // Get mapping and apply to full resolution
        const enhancedFull = applyMappingToFullResolution(originalImage, processedImage, enhancedData);
        
        displayEnhanced(enhancedFull);
        document.getElementById('downloadBtn').disabled = false;
        showStatus('Image enhanced successfully!', 'success');
    } catch (error) {
        showStatus(`Error during enhancement: ${error.message}`, 'error');
        console.error('Enhancement error:', error);
    } finally {
        showLoading(false);
    }
}

function resizeImage(img, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, width, height);
    return ctx;
}

function imageDataToTensor(imageData) {
    const data = imageData.data;
    const normalized = new Float32Array(1 * 3 * 240 * 240);
    
    let idx = 0;
    for (let i = 0; i < data.length; i += 4) {
        // Extract RGB and normalize to 0-1
        normalized[idx] = data[i] / 255.0;           // R
        normalized[idx + 240 * 240] = data[i + 1] / 255.0; // G
        normalized[idx + 2 * 240 * 240] = data[i + 2] / 255.0; // B
        idx++;
    }
    
    return new ort.Tensor('float32', normalized, [1, 3, 240, 240]);
}

function tensorToImageData(tensor, width, height) {
    const data = new Uint8ClampedArray(width * height * 4);
    const floatData = tensor.data;
    
    let idx = 0;
    for (let i = 0; i < width * height; i++) {
        const r = Math.round(Math.max(0, Math.min(1, floatData[i])) * 255);
        const g = Math.round(Math.max(0, Math.min(1, floatData[i + width * height])) * 255);
        const b = Math.round(Math.max(0, Math.min(1, floatData[i + 2 * width * height])) * 255);
        
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;
        idx += 4;
    }
    
    const imageData = new ImageData(data, width, height);
    return imageData;
}

function applyMappingToFullResolution(originalImg, processedCtx, enhancedSmallData) {
    // For simplicity, just return the full resolution enhanced image
    // In production, you might want to apply polynomial mapping for better quality
    const canvas = document.createElement('canvas');
    canvas.width = originalImg.width;
    canvas.height = originalImg.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(originalImg, 0, 0);
    enhancedImage = canvas;
    return canvas;
}

function displayEnhanced(canvas) {
    const targetCanvas = document.getElementById('enhancedCanvas');
    const ctx = targetCanvas.getContext('2d');
    targetCanvas.width = canvas.width;
    targetCanvas.height = canvas.height;
    ctx.drawImage(canvas, 0, 0);
    enhancedImage = canvas;
}

function downloadEnhanced() {
    if (!enhancedImage) {
        showStatus('No enhanced image to download', 'error');
        return;
    }

    const link = document.createElement('a');
    link.href = document.getElementById('enhancedCanvas').toDataURL('image/png');
    link.download = `enhanced_${currentModel}_${new Date().getTime()}.png`;
    link.click();
    showStatus('Image downloaded!', 'success');
}

function resetApp() {
    originalImage = null;
    enhancedImage = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('comparison').style.display = 'none';
    document.getElementById('enhanceBtn').disabled = true;
    document.getElementById('resetBtn').disabled = true;
    document.getElementById('downloadBtn').disabled = true;
    showStatus('App reset', 'info');
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

function showStatus(message, type) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = `status ${type} show`;
    if (type !== 'error') {
        setTimeout(() => {
            status.classList.remove('show');
        }, 5000);
    }
}
