from flask import Flask, request, jsonify, render_template
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import jwt
import datetime
import os
import re
import tempfile
import logging
import numpy as np
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import PyPDF2
from logging.handlers import RotatingFileHandler

# --- Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'truthscan.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload size (reduced for production)

# Initialize extensions
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(handler)

# --- JWT Config ---
JWT_SECRET = app.config['SECRET_KEY']
JWT_EXP_DELTA_SECONDS = 86400  # 24 hours

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_type = db.Column(db.String(10), nullable=False)  # 'text' or 'pdf'
    input_text = db.Column(db.Text, nullable=True)  # Original text or PDF filename
    prediction = db.Column(db.String(10), nullable=False)  # 'ai' or 'human'
    confidence_ai = db.Column(db.Float, nullable=False)
    confidence_human = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Initialize database
with app.app_context():
    db.create_all()

# --- Save Prediction ---
def save_prediction(user_id, input_type, input_text, prediction, confidence):
    try:
        history_entry = PredictionHistory(
            user_id=user_id,
            input_type=input_type,
            input_text=input_text[:1000],  # Limit length for database storage
            prediction=prediction,
            confidence_ai=confidence['ai'],
            confidence_human=confidence['human']
        )
        db.session.add(history_entry)
        db.session.commit()
        logger.info(f"Prediction saved for user {user_id}")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to save prediction: {str(e)}", exc_info=True)
        raise

# --- Load NLTK ---
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK data: {str(e)}")


# --- Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'truthscan_model.keras')
MODEL_INPUT_NAMES = ['repetitive_words', 'type_token_ratio', 'verb_ratio', 'word_inconclusion_presence', 'word_this_presence']
TRAIN_MEANS = np.array([0.31590983, 0.75191151, 0.13076737, 0.1708592, 0.81912229])
TRAIN_STDS = np.array([0.46488986, 0.10908202, 0.02902084, 0.37639613, 0.38492731])
CLASSIFICATION_THRESHOLD = 0.5

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    model = None

# --- JWT Helper ---
def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def jwt_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        token = auth.replace('Bearer ', '')
        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            request.user_id = decoded['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

# --- Validation ---
def is_valid_email(email): return re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email)
def is_valid_username(username): return re.match(r'^[A-Za-z]+$', username)
def is_strong_password(password): return re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$', password)

# --- Feature Extraction ---
class FeatureExtractor:
    @staticmethod
    def get_features(text):
        try:
            if not text.strip():
                raise ValueError("Empty text provided")
            words = word_tokenize(text)
            if not words:
                raise ValueError("No words tokenized")
            pos_tags = pos_tag(words)
            word_count = len(words)
            return {
                'repetitive_words': 1 if FeatureExtractor._detect_repetitive_words(words) else 0,
                'type_token_ratio': len(set(words)) / word_count if word_count > 0 else 0,
                'verb_ratio': len([w for w, t in pos_tags if t.startswith('VB')]) / word_count if word_count > 0 else 0,
                'word_inconclusion_presence': 1 if "in conclusion" in text.lower() else 0,
                'word_this_presence': 1 if "this" in [w.lower() for w in words] else 0
            }
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            raise

    @staticmethod
    def _detect_repetitive_words(words, freq_threshold=5, count_threshold=5):
        try:
            freq_dist = nltk.FreqDist(words)
            repetitive_words = [w for w, f in freq_dist.items() if f > freq_threshold]
            return len(repetitive_words) > count_threshold
        except Exception as e:
            logger.error(f"Repetitive words detection error: {str(e)}")
            return False

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(file_stream):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_stream.read())
        tmp_path = tmp.name

    try:
        with open(tmp_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
            if not text.strip():
                raise ValueError("No extractable text in PDF")
            logger.info(f"Extracted text length: {len(text)} characters")
            return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}", exc_info=True)
        raise ValueError(f"PDF extraction failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Failed to delete temp file: {str(e)}")

# --- Routes ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not all([username, email, password]):
        return jsonify({'error': 'All fields are required'}), 400
    if not is_valid_username(username):
        return jsonify({'error': 'Invalid username'}), 400
    if not is_valid_email(email):
        return jsonify({'error': 'Invalid email'}), 400
    if not is_strong_password(password):
        return jsonify({'error': 'Weak password'}), 400

    existing = User.query.filter((User.username == username) | (User.email == email)).first()
    if existing:
        return jsonify({'error': 'Username or email already exists'}), 400

    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, email=email, password=hashed)

    try:
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': 'User registered'}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration failed: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password, password):
        token = generate_token(user.id)
        return jsonify({'token': token, 'username': user.username}), 200
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@jwt_required
def predict():
    if model is None:
        logger.error("Model not loaded")
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        input_type = None
        input_text = None
        if 'pdf_file' in request.files:
            file = request.files['pdf_file']
            if not file.filename:
                return jsonify({'error': 'No file selected'}), 400
            input_type = 'pdf'
            input_text = file.filename
            logger.info(f"Processing PDF: {input_text}")
            text = extract_text_from_pdf(file)
        elif request.is_json and 'input_text' in request.json:
            input_type = 'text'
            input_text = request.json['input_text'].strip()
            logger.info(f"Processing text: {input_text[:100]}...")
            text = input_text
        else:
            logger.error("Invalid input: No pdf_file or input_text provided")
            return jsonify({'error': 'Invalid input'}), 400

        logger.info("Preprocessing text")
        text = preprocess_text(text)
        if len(text.split()) < 20:
            logger.error("Text too short after preprocessing")
            return jsonify({'error': 'Text too short'}), 400

        logger.info("Extracting features")
        features = FeatureExtractor.get_features(text)
        logger.info(f"Features extracted: {features}")
        features_array = np.array([features[name] for name in MODEL_INPUT_NAMES])
        features_normalized = (features_array - TRAIN_MEANS) / TRAIN_STDS

        logger.info("Preparing model input")
        input_dict = {name: tf.constant([[value]], dtype=tf.float32)
                      for name, value in zip(MODEL_INPUT_NAMES, features_normalized)}

        logger.info("Running model prediction")
        prediction = model.predict(input_dict)
        probability = float(prediction[0][0])
        result = {
            'prediction': 'ai' if probability > CLASSIFICATION_THRESHOLD else 'human',
            'confidence': {
                'ai': round(probability * 100, 2),
                'human': round((1 - probability) * 100, 2)
            },
            'features': features
        }

        logger.info("Saving prediction to history")
        save_prediction(
            user_id=request.user_id,
            input_type=input_type,
            input_text=input_text,
            prediction=result['prediction'],
            confidence=result['confidence']
        )

        logger.info(f"Prediction successful for user {request.user_id}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/history', methods=['GET'])
@jwt_required
def get_history():
    try:
        logger.info(f"Fetching history for user_id: {request.user_id}")
        history = PredictionHistory.query.filter_by(user_id=request.user_id).order_by(PredictionHistory.created_at.desc()).all()
        logger.info(f"Found {len(history)} history entries")
        return jsonify([{
            'id': entry.id,
            'input_type': entry.input_type,
            'input_text': entry.input_text,
            'prediction': entry.prediction,
            'confidence': {
                'ai': entry.confidence_ai,
                'human': entry.confidence_human
            },
            'created_at': entry.created_at.isoformat()
        } for entry in history]), 200
    except Exception as e:
        logger.error(f"Failed to fetch history: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to fetch history: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not Found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal Server Error'}), 500

# --- Run ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)