# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
import re
import nltk
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import PyPDF2
import pdfreader
import tempfile
import logging
import os
from functools import wraps
from flask_cors import CORS 


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'truthscan.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize extensions
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Create tables (run once)
with app.app_context():
    db.create_all()

# NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'truthscan_model.keras')
CLASSIFICATION_THRESHOLD = 0.5
MODEL_INPUT_NAMES = [
    'repetitive_words',
    'type_token_ratio', 
    'verb_ratio',
    'word_inconclusion_presence',
    'word_this_presence'
]

# Training set statistics
TRAIN_MEANS = np.array([0.31590983, 0.75191151, 0.13076737, 0.1708592, 0.81912229])
TRAIN_STDS = np.array([0.46488986, 0.10908202, 0.02902084, 0.37639613, 0.38492731])

# Load model with verification
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    model = None

class FeatureExtractor:
    @staticmethod
    def get_features(text):
        try:
            words = word_tokenize(text)
            pos_tags = pos_tag(words) if words else []
            
            return {
                'repetitive_words': 1 if FeatureExtractor._detect_repetitive_words(words) else 0,
                'type_token_ratio': len(set(words))/len(words) if words else 0,
                'verb_ratio': len([w for w, t in pos_tags if t.startswith('VB')])/len(words) if words else 0,
                'word_inconclusion_presence': 1 if "in conclusion" in text.lower() else 0,
                'word_this_presence': 1 if "this" in [w.lower() for w in words] else 0
            }
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

    @staticmethod
    def _detect_repetitive_words(words, freq_threshold=5, count_threshold=5):
        if not words:
            return False
        freq_dist = nltk.FreqDist(words)
        repetitive_words = [w for w, f in freq_dist.items() if f > freq_threshold]
        return len(repetitive_words) > count_threshold

def extract_text_from_pdf(file_stream):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file_stream.read())
            tmp_path = tmp.name
        
        # Try PyPDF2 first
        try:
            with open(tmp_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
                if text.strip():
                    return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
        
        # Fallback to pdfreader
        try:
            from pdfreader import SimplePDFViewer
            with open(tmp_path, 'rb') as f:
                viewer = SimplePDFViewer(f)
                text = []
                for canvas in viewer:
                    text.append(" ".join(canvas.strings))
                return " ".join(text)
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise ValueError("Could not extract text from PDF")
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

def preprocess_text(text):
    if not text or not isinstance(text, str):
        raise ValueError("Invalid text input")
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
    return text.strip()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Validation Helpers
def is_strong_password(password):
    return bool(re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$', password))

def is_valid_email(email):
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email))

def is_valid_username(username):
    return bool(re.match(r'^[A-Za-z]+$', username))

#routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        # Check all fields are filled
        if not all([username, email, password]):
            flash('All fields are required', 'danger')
            return redirect(url_for('register'))

        # Username validation
        if not is_valid_username(username):
            flash('Username must contain only letters (A-Z or a-z)', 'danger')
            return redirect(url_for('register'))

        # Email validation
        if not is_valid_email(email):
            flash('Invalid email format', 'danger')
            return redirect(url_for('register'))

        # Password strength validation
        if not is_strong_password(password):
            flash('Password must be at least 8 characters long and include uppercase, lowercase, number, and special character.', 'danger')
            return redirect(url_for('register'))

        # Check uniqueness
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists', 'danger')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration failed: {str(e)}")
            flash('Registration failed. Please try again.', 'danger')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        # Input validation
        if not username or not password:
            flash('Username and password are required', 'danger')
            return redirect(url_for('login'))

        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('username'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        return jsonify(error="Model not loaded"), 500
        
    try:
        if 'pdf_file' in request.files:
            file = request.files['pdf_file']
            if file.filename == '':
                return jsonify(error="No selected file"), 400
            text = extract_text_from_pdf(file)
        elif 'input_text' in request.form:
            text = request.form['input_text'].strip()
        else:
            return jsonify(error="No valid input"), 400

        processed_text = preprocess_text(text)
        if len(processed_text.split()) < 20:
            return jsonify(error="Text too short (minimum 20 words)"), 400

        features = FeatureExtractor.get_features(processed_text)
        features_array = np.array([features[name] for name in MODEL_INPUT_NAMES])
        features_normalized = (features_array - TRAIN_MEANS) / TRAIN_STDS

        input_dict = {
            name: tf.constant([[value]], dtype=tf.float32)
            for name, value in zip(MODEL_INPUT_NAMES, features_normalized)
        }

        prediction = model.predict(input_dict)
        probability = float(prediction[0][0])

        return jsonify({
            'prediction': 'ai' if probability > CLASSIFICATION_THRESHOLD else 'human',
            'confidence': {
                'ai': round(probability * 100, 2),
                'human': round((1 - probability) * 100, 2)
            },
            'features': features
        })

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify(error=str(e)), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
