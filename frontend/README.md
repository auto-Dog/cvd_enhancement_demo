# CVD Color Enhancement - Front-End Web App

A **pure front-end web application** for enhancing colors for people with color vision deficiency (CVD). Runs entirely in your browser using ONNX.js - no backend server required!

## 🌟 Features

- ✅ **Client-side processing** - All computation happens in your browser
- ✅ **GPU acceleration** - Uses WebGPU/WASM via ONNX.js
- ✅ **Privacy-first** - Images never leave your device
- ✅ **Responsive design** - Works on desktop, tablet, mobile
- ✅ **Multiple models** - Support for D100 and P100 enhancement models
- ✅ **Easy to deploy** - Static files only, can run on any web server or GitHub Pages

## 📋 Prerequisites

### 1. Convert PyTorch Models to ONNX

First, you need to convert your `.pth` models to ONNX format:

```bash
pip install torch onnx
python convert_pth_to_onnx.py
```

Create `convert_pth_to_onnx.py`:

```python
import torch
import torch.onnx
from colorFilter import colorFilter
import os

def convert_model_to_onnx(pth_path, onnx_path, model_name):
    """Convert PyTorch model to ONNX format"""
    print(f"Converting {model_name}...")
    
    # Load model
    model = colorFilter()
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    # Handle DataParallel wrapper
    if 'module.' in list(checkpoint.keys())[0]:
        new_checkpoint = {}
        for k, v in checkpoint.items():
            new_checkpoint[k.replace('module.', '')] = v
        checkpoint = new_checkpoint
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 240, 240)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'}
        },
        opset_version=12,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✓ Converted to {onnx_path}")

if __name__ == '__main__':
    os.makedirs('frontend/models', exist_ok=True)
    
    # Convert both models
    convert_model_to_onnx(
        'model_vit_cn7aE_D100_optim_base.pth',
        'frontend/models/model_vit_cn7aE_D100_optim_base.onnx',
        'D100 Model'
    )
    
    convert_model_to_onnx(
        'model_vit_cn7aE_P100_optim_base.pth',
        'frontend/models/model_vit_cn7aE_P100_optim_base.onnx',
        'P100 Model'
    )
```

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Install dependencies
pip install torch onnx torchvision

# Convert models
python convert_pth_to_onnx.py

# Start a simple HTTP server
cd frontend
python -m http.server 8000
```

Then open: **http://localhost:8000**

### Option 2: Deploy to GitHub Pages

1. Convert models to ONNX
2. Place `frontend/` contents in your GitHub Pages branch
3. Update `.gitignore` to allow ONNX files (or use Git LFS)
4. Push to GitHub - your site is live!

### Option 3: Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Convert models
RUN python convert_pth_to_onnx.py

# Serve with Python
WORKDIR /app/frontend
CMD ["python", "-m", "http.server", "8000"]
```

Build and run:
```bash
docker build -t cvd-web .
docker run -p 8000:8000 cvd-web
```

## 📁 Project Structure

```
frontend/
├── index.html           # Main HTML page
├── app.js              # Application logic
├── models/             # ONNX model files
│   ├── model_vit_cn7aE_D100_optim_base.onnx
│   └── model_vit_cn7aE_P100_optim_base.onnx
└── README.md           # This file
```

## 🔧 How It Works

1. **Upload Image** → User selects or drags an image
2. **Normalize** → Image is converted to tensor (0-1 range)
3. **Model Inference** → ONNX.js runs the model in browser
4. **Enhanced Output** → Result displayed in real-time
5. **Download** → User downloads the enhanced image

## 📊 Model Details

- **Input**: RGB image (variable size, but processed at 240x240)
- **Output**: Enhanced RGB image (same size as input)
- **Framework**: PyTorch → ONNX
- **Execution**: ONNX Runtime Web (WebGPU/WebAssembly)

## 🌐 Browser Support

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | ✅ | Full support, WebGPU enabled |
| Firefox | ✅ | Full support, WASM fallback |
| Safari | ✅ | WASM fallback |
| Edge | ✅ | Full support |

## ⚙️ Advanced Configuration

### Using with WebGPU (Faster)

Add to `app.js`:

```javascript
const options = {
    executionProviders: ['webgpu', 'wasm']
};
session = await ort.InferenceSession.create(modelPath, options);
```

### Customizing Model Paths

Edit `app.js`:

```javascript
const MODEL_PATHS = {
    D100: 'models/model_D100.onnx',
    P100: 'models/model_P100.onnx'
};
```

## 🐛 Troubleshooting

### Model fails to load

1. Check CORS headers (if loading from different domain)
2. Verify ONNX file exists and is valid
3. Check browser console for errors
4. Ensure opset version is compatible

### Image processing is slow

1. Try enabling WebGPU
2. Reduce image size
3. Use Chrome (best performance)

### ONNX conversion fails

1. Verify PyTorch model structure
2. Check CUDA version compatibility
3. Try different opset version (11, 12, or 13)

## 📝 Conversion Script Issues

If you get DataParallel errors during conversion, the script handles it automatically. If not:

```python
# Manually remove DataParallel wrapper
model = colorFilter()
checkpoint = torch.load('model.pth')

# Remove 'module.' prefix
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(state_dict)
```

## 📦 File Size Considerations

- ONNX models can be large (50-70MB each)
- Consider compressing with `onnxruntime-tools` for production
- Use versioning/caching for better UX

## 🔐 Privacy & Security

- ✅ All processing is local (no data sent to servers)
- ✅ Images are not stored
- ✅ Can be run offline after loading
- ✅ No telemetry or tracking

## 📚 Resources

- [ONNX.js Documentation](https://onnxruntime.ai/docs/get-started/with-web/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [WebGPU Documentation](https://gpuweb.github.io/)

## 📄 License

Same as parent project

## 🤝 Contributing

Contributions welcome! Please ensure models are optimized and well-documented.
