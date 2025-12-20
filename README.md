# Unstructured and Structured Image to Text OCR (Optical Character Recognition)

## Image Text OCR Endpoints

![Image_Text_OCR](OCR_API_Endpoints.png)

- `/healthcheck`: verify that the api is running
- `/ocr/image`: OCR endpoint for single images. Accepts an image file, text prompt and an optional system prompt and provides unstructured output.
- `/ocr/pdf`: OCR endpoint for PDF files with hybrid processing. Extracts text and embedded images, then processes each page and provides unstructured output.
- `/ocr/national_id`: OCR endpoint for National ID cards with structured output. Returns validated JSON matching NationalIDData schema.
- `/ocr/offer_letter`: OCR endpoint for Offer Letters with multi-page support and pre-processing. Returns validated JSON matching OfferLetterData schema.


## Application Architecture


![Image Text OCR](Image-Text-OCR-Architecture.png)


## How to Run

### Prerequsites

- Docker
- Python 3.10+
- Python virtual environment (recommended)

### 1. Clone the repo

```bash
git clone https://github.com/afterSt0rm/image-text-ocr.git
```

### 2. Change directory to `image-text-ocr`

```bash
cd image-text-ocr/
```

### 3. Create directory to store models (can be created manullay as well)

```bash
mkdir -p llama/models/
```

### 4. Download and save the models to `/llama/models/`

- Head to [HuggingFace Qwen3-VL Model Page](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/tree/main)
- Download the following files to `/llama/models/`
  - `Qwen3VL-4B-Instruct-Q4_K_M.gguf`
  - `mmproj-Qwen3VL-4B-Instruct-F16.gguf`

### 5. Create a python virtual environment

```python
python -m venv .venv
```

- Activate virtual environment on linux/macos

```bash
source .venv/bin/activate
```

- Activate virtual environment on windows

```bash
.venv\Scripts\activate 
```

### 6. Install dependencies

```python
pip install -r requirements.txt
```

### 7. Crete .env file

```
BASE_URL=http://127.0.0.1:8081/v1/
```

### 8. Run the `llama.cpp-server`

```bash
docker run -v $(pwd)/llama/models:/models -p 8081:8081 ghcr.io/ggml-org/llama.cpp:server -m /models/Qwen3VL-4B-Instruct-Q4_K_M.gguf --mmproj /models/mmproj-Qwen3VL-4B-Instruct-F16.gguf --port 8081 --host 0.0.0.0 -c 10000
```

### 9. Access the FastAPI Endpoints

```bash
uvicorn api.main:app --reload
```

- ⚠️ `llama-server` must be running

### 10. Access the Streamlit UI

```bash
streamlit run frontend/streamlit_app.py
```

- ⚠️ `fastapi` must be running
