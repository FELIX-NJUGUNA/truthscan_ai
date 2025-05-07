# truthscan_ai Flask API

A lightweight Flask backend integrated with a pre-trained machine learning model saved in `.keras` format. This API exposes endpoints for analyzing input data (e.g. text or numerical) and returning predictions, making it ideal for AI-powered web or mobile applications.

---

## 🚀 Features

- ✅ Flask RESTful API
- ✅ Loads and serves `.keras` machine learning models
- ✅ Supports JSON input
- ✅ Deployed-ready for Render (or other platforms)
- ✅ NLTK tokenization (if using NLP)
- ✅ Cross-Origin Resource Sharing (CORS) support

---

## 🛠️ Setup Instructions

### 1. Clone the Repository


```bash
git clone https://github.com/FELIX-NJUGUNA/truthscan_ai
cd truthscan_ai
```
---
### 2. Install Dependencies
Create a virtual environment and install dependencies:

```bash
    Copy
    Edit
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```
### 3. Download NLTK Resources
```
python -m nltk.downloader -d nltk_data punkt
```
### 4. Run the Flask App
```bash
    Copy
    Edit
    python main.py
```
#🧪 API Usage
Endpoint: /predict/
POST Request (JSON):
```json
Copy
Edit
{
  "text": "Your input text goes here"
}
```

Response (JSON):
```json
Copy
Edit
{
  "prediction": "Your model's output"
}
```


