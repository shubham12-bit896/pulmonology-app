# Enterprise Blueprint for the Pulmonology Department Application
#
# This self-contained application has been architecturally refactored to demonstrate
# best practices in security, interoperability (FHIR), and clinical workflow management.
#
# This version integrates standalone Laboratory and Radiology requesting modules and a Gemini AI assistant.
#
# To run this application:
# 1. Make sure you have Python installed and the required libraries:
#    pip install Flask Flask-SQLAlchemy scikit-learn pandas Flask-Login Werkzeug requests
# 2. IMPORTANT: Set your Google AI API Key. Open your terminal and run this command before starting the app:
#    For Windows (Command Prompt): set GEMINI_API_KEY=YOUR_API_KEY_HERE
#    For Windows (PowerShell):   $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
#    For macOS/Linux:            export GEMINI_API_KEY='YOUR_API_KEY_HERE'
# 3. (OPTIONAL) To connect to a live main hospital app, set its URL:
#    set MAIN_APP_API_URL=http://your-main-hospital-server.com/api/patient
# 4. IMPORTANT: If you ran a previous version, please delete the 'instance/respiratory_clinic.db' file
#    before running this new version to ensure the database schema is updated correctly.
# 5. Save this code as app.py and run it from your terminal: python app.py
# 6. Open your web browser and go to http://127.0.0.1:5000
#
# Login Credentials:
# - Admin: username='admin', password='admin123'
# - Doctor: username='dr_house', password='doctor123'
# - IT Executive (New): Add via Admin panel.

import os
import pickle
import datetime
import pandas as pd
import uuid
import threading
import requests
import time
import json as _json
import secrets

from functools import wraps
from flask import Flask, render_template_string, request, redirect, url_for, flash, get_flashed_messages, jsonify, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.ensemble import RandomForestClassifier
from jinja2 import Environment, BaseLoader, TemplateNotFound
from datetime import timezone, timedelta

# ----------------- Configuration & Secrets Management ----------------- #
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(16))
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance', 'respiratory_clinic.db'))
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # IMPORTANT: The application will look for this environment variable for the API key.
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '') 
    # NEW: Add the URL for the main hospital EMR API
    MAIN_APP_API_URL = os.environ.get('MAIN_APP_API_URL', '') 
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = True


# --- Timezone Configuration ---
IST = timezone(timedelta(hours=5, minutes=30))


# ----------------- SIMULATED DATABASES & TERMINOLOGY SERVICES ----------------- #
SIMULATED_MAIN_APP_DB = {
    "MAIN-APP-123": {"name": "Jane Doe", "age": 45, "gender": "Female", "contact": "555-0101"},
    "MAIN-APP-456": {"name": "Peter Jones", "age": 72, "gender": "Male", "contact": "555-0102"},
    "MAIN-APP-789": {"name": "Sam Wilson", "age": 58, "gender": "Male", "contact": "555-0103"},
}

SIMULATED_SNOMED_DB = {
    "195967001": "Chronic obstructive pulmonary disease (COPD)",
    "71388002": "Asthma",
    "44054006": "Pneumonia",
    "233604007": "Lung cancer"
}

# ----------------- App Setup & Extensions ----------------- #
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'



# Ensure 'downloads' directory exists for Lab/Radiology reports
os.makedirs(os.path.join(basedir, "downloads"), exist_ok=True)

# ----------------- Database Models (FHIR-Inspired) ----------------- #
class Role:
    ADMIN = 'Admin'
    DOCTOR = 'Doctor'
    HEALTH_WORKER = 'Health Worker'
    IT_EXECUTIVE = 'IT Executive'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    role = db.Column(db.String(50), nullable=False, default=Role.HEALTH_WORKER)
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uhid = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    contact = db.Column(db.String(20), nullable=True)
    admission_date = db.Column(db.DateTime, default=lambda: datetime.datetime.now(IST))
    status = db.Column(db.String(20), nullable=False, default='Outpatient')
    smoking_status = db.Column(db.String(20), nullable=False, default='Never Smoked')
    pack_years = db.Column(db.Integer, nullable=False, default=0)
    history_of_asthma = db.Column(db.String(5), nullable=False, default='No')
    observations = db.relationship('Observation', backref='patient', lazy='dynamic', cascade="all, delete-orphan")
    conditions = db.relationship('Condition', backref='patient', lazy='dynamic', cascade="all, delete-orphan")
    clinical_notes = db.relationship('ClinicalNote', backref='patient', lazy='dynamic', cascade="all, delete-orphan")

class Observation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id_fk = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    test_date = db.Column(db.Date, nullable=False, default=datetime.date.today)
    fvc_value = db.Column(db.Float)
    fvc_predicted_value = db.Column(db.Float)
    fev1_value = db.Column(db.Float)
    fev1_predicted_value = db.Column(db.Float)
    post_bronchodilator_fev1 = db.Column(db.Float)
    fev1_fvc_ratio = db.Column(db.Float)

class Condition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id_fk = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    code = db.Column(db.String(50), nullable=False)
    display_text = db.Column(db.String(200), nullable=False)
    onset_date = db.Column(db.Date)
    status = db.Column(db.String(20), default='active')

class ClinicalNote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id_fk = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    note_type = db.Column(db.String(50), default="Pulmonology Consult Note")
    authored_by = db.Column(db.String(100))
    authored_on = db.Column(db.DateTime, default=lambda: datetime.datetime.now(IST))
    subjective = db.Column(db.Text)
    objective = db.Column(db.Text)
    assessment = db.Column(db.Text)
    plan = db.Column(db.Text)

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    username = db.Column(db.String(100), nullable=False)
    action = db.Column(db.String(200), nullable=False)
    target_type = db.Column(db.String(50))
    target_id = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.datetime.now(IST))

# ----------------- User Loader & Decorators ----------------- #
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def role_required(roles):
    if not isinstance(roles, list): roles = [roles]
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role not in roles:
                flash('You do not have permission to access this page.', 'danger')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def log_activity(action_template):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = f(*args, **kwargs)
            try:
                action_str = action_template.format(**kwargs)
                target_type, target_id = None, None
                if 'patient_id' in kwargs:
                    target_type, target_id = 'Patient', kwargs['patient_id']
                elif 'user_id' in kwargs:
                    target_type, target_id = 'User', kwargs['user_id']
                elif 'note_id' in kwargs:
                    target_type, target_id = 'ClinicalNote', kwargs['note_id']

                log_entry = AuditLog(
                    user_id=current_user.id, username=current_user.username,
                    action=action_str, target_type=target_type, target_id=target_id
                )
                db.session.add(log_entry)
                db.session.commit()
            except Exception as e:
                print(f"Error creating audit log: {e}")
            return response
        return decorated_function
    return decorator

# ----------------- Machine Learning Model ----------------- #
MODEL_FILENAME = os.path.join(basedir, 'instance', 'copd_prediction_model.pkl')

def train_and_save_model():
    data = { 'age': [65, 70, 55, 75], 'smoking_status': [2, 2, 0, 1], 'pack_years': [40, 50, 0, 30], 'history_of_asthma': [0, 1, 1, 0], 'fev1_fvc_ratio': [0.65, 0.68, 0.85, 0.72], 'copd_diagnosis': [1, 1, 0, 1] }
    df = pd.DataFrame(data)
    X = df.drop('copd_diagnosis', axis=1)
    y = df['copd_diagnosis']
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    with open(MODEL_FILENAME, 'wb') as f: pickle.dump(model, f)
    print("ML model trained and saved.")

def predict_copd(patient_data):
    if not os.path.exists(MODEL_FILENAME): return None, None
    with open(MODEL_FILENAME, 'rb') as f: model = pickle.load(f)
    smoking_map = {'Never Smoked': 0, 'Former Smoker': 1, 'Current Smoker': 2}
    asthma_map = {'No': 0, 'Yes': 1}
    features = pd.DataFrame([[
        patient_data['age'],
        smoking_map.get(patient_data['smoking_status'], 0),
        patient_data['pack_years'],
        asthma_map.get(patient_data['history_of_asthma'], 0),
        patient_data.get('fev1_fvc_ratio', 0)
    ]], columns=['age', 'smoking_status', 'pack_years', 'history_of_asthma', 'fev1_fvc_ratio'])
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return prediction[0], probability[0][1]


# ----------------- Lab Request System (from test.py) ----------------- #
LAB_DEFAULT_HOST = "http://127.0.0.1:5000"
LAB_SHARED_API_KEY = "hospital_shared_key"
LAB_HISTORY_PATH = os.path.join(basedir, "downloads", "order_history.json")

TEST_CATEGORIES = {
    'biochemistry': {
        'Kidney Function': ['GLU', 'UREA', 'CREATININE'],
        'Liver Function': ['SGOT', 'SGPT', 'ALBUMIN', 'TOTAL_BILIRUBIN'],
        'Thyroid Function': ['TSH', 'T3', 'T4'],
        'Cardiac Markers': ['TROPONIN_I'],
        'Lipid Profile': ['TOTAL_CHOLESTEROL', 'HDL', 'LDL'],
        'Electrolytes': ['SODIUM', 'POTASSIUM']
    },
    'microbiology': {
        'Wet Mount & Staining': ['GRAM_STAIN', 'HANGING_DROP', 'INDIA_INK', 'STOOL_OVA', 'KOH_MOUNT', 'ZN_STAIN'],
        'Culture & Sensitivity': ['BLOOD_CULTURE', 'URINE_CULTURE', 'SPUTUM_CULTURE', 'WOUND_CULTURE', 'THROAT_CULTURE', 'CSF_CULTURE'],
        'Fungal Culture': ['FUNGAL_CULTURE', 'FUNGAL_ID', 'ANTIFUNGAL_SENS'],
        'Serology': ['WIDAL', 'TYPHIDOT', 'DENGUE_NS1', 'MALARIA_AG', 'HIV_ELISA', 'HBSAG']
    },
    'pathology': {
        'Histopathology': ['BIOPSY_HISTOPATHOLOGY', 'SURGICAL_PATHOLOGY'],
        'Hematology': ['CBC', 'PERIPHERAL_SMEAR', 'BONE_MARROW', 'COAGULATION'],
        'Immunohistochemistry': ['IHC_MARKERS', 'SPECIAL_STAINS', 'MOLECULAR_PATH']
    }
}

def lab_load_history(department=None):
    if not os.path.exists(LAB_HISTORY_PATH): return []
    try:
        with open(LAB_HISTORY_PATH, 'r', encoding='utf-8') as f:
            history = _json.load(f) or []
            if department:
                history = [order for order in history if order.get('department') == department]
            return history
    except Exception:
        return []

def lab_save_history(history):
    try:
        with open(LAB_HISTORY_PATH, 'w', encoding='utf-8') as f:
            _json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def lab_record_order(entry):
    history = lab_load_history()
    history.insert(0, entry)
    lab_save_history(history)

def lab_perform_test_request(host, department, uhid, tests, priority='routine', specimen='Blood', clinical_notes=''):
    url = f"{host.rstrip('/')}/api/lab/orders"
    payload = {
        "externalOrderId": f"EXT_{uhid}_{int(time.time())}",
        "priority": priority,
        "patient": {"uhid": uhid, "name": f"Patient {uhid}", "age": 30, "gender": "Not Specified"},
        "clinician": {"name": f"Dr. {department.title()}", "department": department, "contact": "Not Specified"},
        "tests": [{"testCode": test} for test in tests],
        "panels": [],
        "specimen": specimen,
        "clinicalNotes": clinical_notes
    }
    headers = {'Content-Type': 'application/json', 'X-API-Key': LAB_SHARED_API_KEY}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
    except requests.RequestException as e:
        return None, f"Request error: {e}"
    if resp.status_code == 201:
        j = resp.json()
        order_id = j.get('orderId')
        if order_id:
            lab_record_order({
                'orderId': order_id,
                'externalOrderId': payload.get('externalOrderId'),
                'uhid': uhid,
                'department': department,
                'priority': priority,
                'tests': tests,
                'specimen': specimen,
                'createdAt': datetime.datetime.now().isoformat()
            })
            return order_id, None
        else:
            return None, "No order ID received from server"
    return None, f"Server returned error {resp.status_code}: {resp.text[:400]}"


# ----------------- Radiology Request System (from radio.py) ----------------- #
RADIOLOGY_DEFAULT_HOST = "http://127.0.0.1:5000"
RADIOLOGY_REQUEST_QUEUE = []
radiology_queue_lock = threading.Lock()

def radiology_save_stream_to_file(resp_data, out_path):
    with open(out_path, 'wb') as f:
        f.write(resp_data)

def radiology_perform_request(host, department, uhid, scan_type, body_part):
    url = f"{host.rstrip('/')}/api/radiology/v1/get_or_request_scan"
    payload = {
        "department_name": department,
        "uhid": uhid,
        "type_of_scan": scan_type,
        "body_part": body_part
    }
    headers = {'Accept': 'application/json, application/dicom, */*'}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30, stream=True)
        if resp.status_code == 200:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{uhid or 'scan'}_{ts}.dcm"
            out_path = os.path.join(basedir, "downloads", fname)
            radiology_save_stream_to_file(resp.content, out_path)
            return fname, None
       
        if resp.status_code == 202:
            return None, "Scan is being processed. Check the queue for status."

        return None, f"Server returned error {resp.status_code}: {resp.text[:400]}"
    except requests.RequestException as e:
        return None, f"Request error: {e}"

def radiology_process_scan_request_worker(req_id, host, department, uhid, scan_type, body_part):
    with app.app_context():
        print(f"[{req_id}] Starting background processing for UHID: {uhid}")
        time.sleep(5) 
        dicom_file, error = radiology_perform_request(host, department, uhid, scan_type, body_part)

        with radiology_queue_lock:
            for req in RADIOLOGY_REQUEST_QUEUE:
                if req['id'] == req_id:
                    if error:
                        req['status'] = 'Failed'
                        req['error'] = error
                    else:
                        req['status'] = 'Completed'
                        req['filename'] = dicom_file
                    break

with app.app_context():
    db.create_all()  # Create all database tables

    # Check if the User table is empty, then create default users
    if User.query.count() == 0:
        print("Database is empty. Creating default users...")
        admin = User(username='admin', role=Role.ADMIN)
        admin.set_password('admin123')
        doctor = User(username='dr_house', role=Role.DOCTOR)
        doctor.set_password('doctor123')
        db.session.add_all([admin, doctor])
        db.session.commit()
        print("Default users created.")

    # Safely try to create the ML model file without crashing
    try:
        if not os.path.exists(MODEL_FILENAME):
            train_and_save_model()
    except Exception as e:
        print(f"Warning: Could not create ML model file. Error: {e}")

# ----------------- HTML Templates (as strings) ----------------- #
base_template_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Pulmonology Dept.{% endblock %}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #F3F4F6; }
        .header { background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .heading { color: #2d5a2f; }
        .subheading, .icon { color: #3a86d7; }
        .card { background-color: white; border-radius: 0.75rem; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1); }
        .btn { padding: 0.6rem 1.2rem; border-radius: 0.5rem; font-weight: 600; transition: all 0.2s ease-in-out; }
        .btn-primary { background-color: #2d5a2f; color: white; }
        .btn-primary:hover { background-color: #3a86d7; }
        .btn-secondary { background-color: #e5e7eb; color: #1f2937; }
        .btn-secondary:hover { background-color: #d1d5db; }
        .modal-backdrop { background-color: rgba(0, 0, 0, 0.5); transition: opacity 0.3s ease-in-out; }
    </style>
    {% block head_extra %}{% endblock %}
</head>
<body class="flex flex-col min-h-screen">
    <header class="header sticky top-0 z-50 p-4 flex flex-wrap justify-between items-center">
        <div class="flex items-center">
            <i class="fas fa-lungs fa-2x subheading"></i>
            <h1 class="text-xl sm:text-2xl font-bold heading ml-3">Pulmonology Department</h1>
        </div>
        <nav class="flex items-center space-x-2 sm:space-x-4 mt-2 sm:mt-0 text-sm sm:text-base">
            {% if current_user.is_authenticated %}
                <a href="{{ url_for('index') }}" class="text-gray-600 hover:text-[#3a86d7]">Dashboard</a>
                <a href="{{ url_for('patients_list') }}" class="text-gray-600 hover:text-[#3a86d7]">Patients</a>
                <!-- Services Dropdown -->
                <div class="relative" x-data="{ open: false }">
                    <button @click="open = !open" @click.away="open = false" class="text-gray-600 hover:text-[#3a86d7] inline-flex items-center">
                        <span>Services</span>
                        <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </button>
                    <div x-show="open"
                         x-transition:enter="transition ease-out duration-100"
                         x-transition:enter-start="transform opacity-0 scale-95"
                         x-transition:enter-end="transform opacity-100 scale-100"
                         x-transition:leave="transition ease-in duration-75"
                         x-transition:leave-start="transform opacity-100 scale-100"
                         x-transition:leave-end="transform opacity-0 scale-95"
                         class="absolute right-0 mt-2 w-48 origin-top-right bg-white rounded-md shadow-lg z-20"
                         style="display: none;">
                        <div class="py-1">
                            <a href="{{ url_for('lab_request') }}" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Lab Tests</a>
                            <a href="{{ url_for('radiology_request') }}" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Radiology</a>
                        </div>
                    </div>
                </div>
                {% if current_user.role == 'Admin' %}
                <a href="{{ url_for('manage_users') }}" class="text-gray-600 hover:text-[#3a86d7]">Users</a>
                {% endif %}
                {% if current_user.role in [Role.ADMIN, Role.IT_EXECUTIVE] %}
                <a href="{{ url_for('view_audit_log') }}" class="text-gray-600 hover:text-[#3a86d7]">Audit Log</a>
                {% endif %}
                <span class="hidden sm:inline text-gray-500">Welcome, {{ current_user.username }} ({{ current_user.role }})</span>
                <a href="{{ url_for('logout') }}" class="btn btn-secondary text-xs sm:text-sm">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}" class="btn btn-primary">Login</a>
            {% endif %}
        </nav>
    </header>
    <main class="flex-grow p-4 sm:p-6 md:p-10">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                <div class="p-4 mb-6 text-sm rounded-lg shadow-md
                    {% if category == 'success' %} bg-green-100 text-green-800 border-l-4 border-green-500
                    {% else %} bg-red-100 text-red-800 border-l-4 border-red-500 {% endif %}" role="alert">
                    <span class="font-medium">{{ category|capitalize }}:</span> {{ message }}
                </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </main>
</body>
</html>
"""

login_html = """
{% extends 'base.html' %}
{% block title %}Login{% endblock %}
{% block content %}
<div class="flex justify-center items-center mt-10 sm:mt-20">
    <div class="card p-8 w-full max-w-md">
        <h2 class="text-3xl font-bold heading text-center mb-6">Login</h2>
        <form method="POST" action="{{ url_for('login') }}">
            <div class="mb-4">
                <label for="username" class="block text-sm font-medium text-gray-700">Username</label>
                <input type="text" name="username" required class="mt-1 block w-full px-4 py-2 bg-gray-50 border rounded-md">
            </div>
            <div class="mb-6">
                <label for="password" class="block text-sm font-medium text-gray-700">Password</label>
                <input type="password" name="password" required class="mt-1 block w-full px-4 py-2 bg-gray-50 border rounded-md">
            </div>
            <button type="submit" class="btn btn-primary w-full">Login</button>
        </form>
    </div>
</div>
{% endblock %}
"""

index_html = """
{% extends 'base.html' %}
{% block title %}Dashboard{% endblock %}
{% block content %}
<div x-data="{
    showChat: false,
    messages: [],
    userInput: '',
    isLoading: false,
    initChat() {
        this.messages.push({ sender: 'gemini', text: 'Hello! I am SpiroSage, your clinical AI assistant. How can I help you with your healthcare-related questions today?' });
    },
    submitQuery() {
        if (this.isLoading || this.userInput.trim() === '') return;
       
        this.isLoading = true;
        this.messages.push({ sender: 'user', text: this.userInput });
        const userMessage = this.userInput;
        this.userInput = '';

        this.$nextTick(() => { this.$refs.chatMessages.scrollTop = this.$refs.chatMessages.scrollHeight; });

        fetch('{{ url_for('ask_gemini') }}', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompt: userMessage })
        })
        .then(response => response.json())
        .then(data => {
            this.messages.push({ sender: 'gemini', text: data.response || 'Sorry, an error occurred.' });
        })
        .catch(error => {
            this.messages.push({ sender: 'gemini', text: 'An error occurred. Please try again.' });
            console.error('Error:', error);
        })
        .finally(() => {
            this.isLoading = false;
            this.$nextTick(() => { this.$refs.chatMessages.scrollTop = this.$refs.chatMessages.scrollHeight; });
        });
    },
    simpleMarkdownToHTML(text) {
        let html = text
            .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
            .replace(/\\*(.*?)\\*/g, '<em>$1</em>');
       
        html = html.split('\\n').map(line => {
            if (line.trim().startsWith('* ')) {
                return `<ul><li class='ml-4'>${line.trim().substring(2)}</li></ul>`;
            }
            return line;
        }).join('<br>');
       
        html = html.replace(/<br><ul>/g, '<ul>').replace(/<\\/ul><br>/g, '</ul>');
       
        return html;
    }
}" x-init="initChat()">
    <h2 class="text-3xl sm:text-4xl font-bold heading mb-8">Department Dashboard</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
        <div class="card p-6"><p class="text-base font-medium text-gray-500">Total Patients</p><p class="text-4xl font-bold text-gray-800 mt-1">{{ total_patients }}</p></div>
        <div class="card p-6"><p class="text-base font-medium text-gray-500">Current Inpatients</p><p class="text-4xl font-bold text-gray-800 mt-1">{{ current_inpatients }}</p></div>
        <div class="card p-6"><p class="text-base font-medium text-gray-500">Admissions (Last 7 Days)</p><p class="text-4xl font-bold text-gray-800 mt-1">{{ recent_admissions }}</p></div>
    </div>
    <div class="mt-12 card p-8">
        <h3 class="text-2xl font-bold heading mb-6">Quick Actions</h3>
        <div class="flex flex-wrap gap-4">
            <a href="{{ url_for('add_patient') }}" class="btn btn-primary"><i class="fas fa-user-plus mr-2"></i> Add New Patient</a>
        </div>
    </div>

    <!-- AI Assistant -->
    <div @keydown.escape.window="showChat = false" class="fixed bottom-0 right-0 p-4 sm:p-8 z-[999]">
        <!-- Chat Bubble -->
        <button @click="showChat = !showChat" class="chat-bubble w-16 h-16 rounded-full flex items-center justify-center">
            <svg class="chat-bubble-icon w-8 h-8 text-white" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M18 10H6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M12 14H6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M12 6H6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M16 4.9999C16 4.9999 17.5 5.4999 17.5 7.4999C17.5 9.4999 16 9.9999 16 9.9999" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M20 4.9999C20 4.9999 21.5 5.4999 21.5 7.4999C21.5 9.4999 20 9.9999 20 9.9999" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12C2 7.28595 2 4.92893 3.46447 3.46447C4.92893 2 7.28595 2 12 2C16.714 2 19.0711 2 20.5355 3.46447C22 4.92893 22 7.28595 22 12C22 16.714 22 19.0711 20.5355 20.5355C19.0711 22 16.714 22 12 22C7.28595 22 4.92893 22 3.46447 20.5355C2 19.0711 2 16.714 2 12Z" stroke="currentColor" stroke-width="1.5"/>
            </svg>
        </button>
        <!-- Chat Modal -->
        <div x-show="showChat" 
             x-transition:enter="transition ease-out duration-300"
             x-transition:enter-start="opacity-0 translate-y-4"
             x-transition:enter-end="opacity-100 translate-y-0"
             x-transition:leave="transition ease-in duration-200"
             x-transition:leave-start="opacity-100 translate-y-0"
             x-transition:leave-end="opacity-0 translate-y-4"
             class="fixed bottom-24 right-4 sm:right-8 w-[calc(100%-2rem)] max-w-md bg-white rounded-xl shadow-2xl flex flex-col h-[70vh]" 
             style="display: none;">
           
            <div class="bg-gray-700 text-white p-4 rounded-t-xl flex justify-between items-center">
                <h3 class="text-lg font-bold">Clinical AI Assistant</h3>
                <button @click="showChat = false" class="text-gray-300 hover:text-white text-2xl">&times;</button>
            </div>

            <div x-ref="chatMessages" class="flex-grow p-4 overflow-y-auto space-y-4 bg-gray-50 chat-area">
                <template x-for="(message, index) in messages" :key="index">
                    <div class="flex" :class="message.sender === 'user' ? 'justify-end' : 'justify-start'">
                        <div class="prose max-w-xs md:max-w-sm p-3 rounded-lg" :class="message.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'" x-html="simpleMarkdownToHTML(message.text)"></div>
                    </div>
                </template>
                <div x-show="isLoading" class="flex justify-start">
                    <div class="max-w-xs md:max-w-sm p-3 rounded-lg bg-gray-200 text-gray-800">
                        <div class="typing-indicator"><span></span><span></span><span></span></div>
                    </div>
                </div>
            </div>

            <div class="p-4 border-t border-gray-200 bg-white rounded-b-xl">
                <form @submit.prevent="submitQuery" class="flex items-center space-x-2">
                    <input x-model="userInput" type="text" placeholder="Ask a healthcare question..." class="w-full px-4 py-2 bg-gray-100 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500" required>
                    <button type="submit" class="bg-blue-600 text-white rounded-full w-10 h-10 flex-shrink-0 flex items-center justify-center hover:bg-blue-700 transition">
                        <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                    </button>
                </form>
            </div>
        </div>
    </div>
</div>

<style>
.chat-bubble {
    background: linear-gradient(45deg, #3b82f6, #14b8a6);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease-in-out;
}
.chat-bubble:hover {
    transform: scale(1.1) rotate(10deg);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    background: linear-gradient(45deg, #2563eb, #0d9488);
}
.chat-bubble .chat-bubble-icon {
    transition: transform 0.3s ease-in-out;
}
.chat-bubble:hover .chat-bubble-icon {
    transform: rotate(-10deg);
}

.typing-indicator { display: flex; padding: 8px; }
.typing-indicator span { height: 8px; width: 8px; float: left; margin: 0 1px; background-color: #9E9EA1; display: block; border-radius: 50%; opacity: 0.4; animation: 1s blink infinite; }
.typing-indicator span:nth-child(2) { animation-delay: .2s; }
.typing-indicator span:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 50% { opacity: 1; } }

.prose { max-width: none; color: inherit; }
.prose strong { color: inherit; }
.prose p, .prose ul, .prose li { margin: 0; padding: 0; }
.prose ul { list-style-type: disc; margin-left: 1.25rem; }

.chat-area::-webkit-scrollbar { width: 8px; }
.chat-area::-webkit-scrollbar-track { background: #f1f5f9; }
.chat-area::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
.chat-area::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
{% endblock %}
"""

# ... (keep all other html templates: patients_list_html, patient_form_html, etc. as they are)

patients_list_html = """
{% extends 'base.html' %}
{% block title %}Patient Records{% endblock %}
{% block content %}
<div class="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8">
    <h2 class="text-3xl sm:text-4xl font-bold heading mb-4 sm:mb-0">Patient Records</h2>
    <a href="{{ url_for('add_patient') }}" class="btn btn-primary"><i class="fas fa-user-plus mr-2"></i> Add Patient</a>
</div>
<div class="card p-6 mb-8">
    <form method="GET" action="{{ url_for('patients_list') }}">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
                <label for="search_uhid" class="block text-sm font-medium text-gray-700">Search by UHID</label>
                <input type="text" name="search_uhid" id="search_uhid" value="{{ request.args.get('search_uhid', '') }}" class="mt-1 block w-full px-4 py-2 bg-gray-50 border rounded-md">
            </div>
            <div>
                <label for="search_name" class="block text-sm font-medium text-gray-700">Search by Name</label>
                <input type="text" name="search_name" id="search_name" value="{{ request.args.get('search_name', '') }}" class="mt-1 block w-full px-4 py-2 bg-gray-50 border rounded-md">
            </div>
            <div class="flex items-end">
                <button type="submit" class="btn btn-primary w-full sm:w-auto">Search</button>
            </div>
        </div>
    </form>
</div>
<div class="card"><div class="overflow-x-auto"><table class="w-full text-left">
    <thead class="bg-gray-50"><tr>
        <th class="p-5 font-semibold text-gray-600">UHID</th><th class="p-5 font-semibold text-gray-600">Name</th>
        <th class="p-5 font-semibold text-gray-600 hidden md:table-cell">Age</th><th class="p-5 font-semibold text-gray-600">Status</th>
        <th class="p-5 font-semibold text-gray-600 hidden sm:table-cell">Admission Date</th>
        <th class="p-5 font-semibold text-gray-600">Actions</th>
    </tr></thead>
    <tbody class="divide-y divide-gray-200">
    {% for patient in patients %}
        <tr class="hover:bg-gray-50">
            <td class="p-5 text-gray-700">{{ patient.uhid }}</td><td class="p-5 font-medium text-gray-800">{{ patient.name }}</td>
            <td class="p-5 text-gray-700 hidden md:table-cell">{{ patient.age }}</td>
            <td class="p-5">
                {% if patient.status == 'Inpatient' %}
                    <span class="px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800"><i class="fas fa-bed mr-2"></i>Inpatient</span>
                {% else %}
                    <span class="px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800"><i class="fas fa-walking mr-2"></i>Outpatient</span>
                {% endif %}
            </td>
            <td class="p-5 text-gray-700 hidden sm:table-cell">{{ patient.admission_date.strftime('%Y-%m-%d %H:%M') }}</td>
            <td class="p-5"><a href="{{ url_for('patient_detail', patient_id=patient.id) }}" class="text-indigo-600 hover:text-indigo-900 font-semibold"><i class="fas fa-external-link-alt mr-2"></i>View Details</a></td>
        </tr>
    {% else %}
        <tr><td colspan="6" class="p-8 text-center text-gray-500">No patient records found.</td></tr>
    {% endfor %}
    </tbody>
</table></div></div>
{% endblock %}
"""

patient_form_html = """
{% extends 'base.html' %}
{% block title %}Add Patient{% endblock %}
{% block content %}
<h2 class="text-3xl sm:text-4xl font-bold heading mb-8">Add New Patient to Department</h2>
<div class="card p-8">
    <div id="uhid-step" class="max-w-md mx-auto">
        <label for="uhid-input" class="block text-sm font-medium text-gray-700 mb-1">Enter Patient UHID from Main Registry</label>
        <div class="flex items-center space-x-2">
            <input type="text" id="uhid-input" placeholder="e.g., MAIN-APP-123" class="block w-full px-4 py-2 bg-gray-50 border rounded-md">
            <button id="fetch-patient-btn" class="btn btn-primary whitespace-nowrap">Fetch Patient</button>
        </div>
        <div id="fetch-status" class="mt-2 text-sm"></div>
    </div>

    <form method="POST" action="{{ url_for('add_patient') }}" id="patient-form" class="hidden">
        <input type="hidden" name="uhid" id="form-uhid">
        <input type="hidden" name="name" id="form-name">
        <input type="hidden" name="age" id="form-age">
        <input type="hidden" name="gender" id="form-gender">
        <input type="hidden" name="contact" id="form-contact">

        <div id="patient-summary" class="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 class="font-bold text-lg text-blue-800">New Patient: <span id="summary-name"></span></h3>
            <p class="text-gray-600">UHID: <span id="summary-uhid"></span> | Age: <span id="summary-age"></span> | Gender: <span id="summary-gender"></span></p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="md:col-span-3 font-semibold text-xl subheading border-b pb-3 mb-2">Pulmonology-Specific Details</div>
            <div>
                <label for="smoking_status" class="block text-sm font-medium text-gray-700">Smoking Status</label>
                <select name="smoking_status" required class="mt-1 block w-full p-2 bg-gray-50 border rounded-md">
                    <option>Never Smoked</option><option>Former Smoker</option><option>Current Smoker</option>
                </select>
            </div>
            <div>
                <label for="pack_years" class="block text-sm font-medium text-gray-700">Pack Years</label>
                <input type="number" name="pack_years" value="0" required class="mt-1 block w-full p-2 bg-gray-50 border rounded-md">
            </div>
            <div>
                <label for="history_of_asthma" class="block text-sm font-medium text-gray-700">History of Asthma</label>
                <select name="history_of_asthma" required class="mt-1 block w-full p-2 bg-gray-50 border rounded-md">
                    <option>No</option><option>Yes</option>
                </select>
            </div>
        </div>
        <div class="mt-10 flex justify-end space-x-4">
            <a href="{{ url_for('patients_list') }}" class="btn btn-secondary">Cancel</a>
            <button type="submit" class="btn btn-primary">Create Patient Chart</button>
        </div>
    </form>
</div>
<script>
document.addEventListener('DOMContentLoaded', function() {
    const fetchBtn = document.getElementById('fetch-patient-btn');
    const uhidInput = document.getElementById('uhid-input');
    const fetchStatus = document.getElementById('fetch-status');
    const patientForm = document.getElementById('patient-form');
    const uhidStep = document.getElementById('uhid-step');

    fetchBtn.addEventListener('click', function() {
        const uhid = uhidInput.value.trim();
        if (!uhid) {
            fetchStatus.textContent = 'Please enter a UHID.';
            fetchStatus.className = 'mt-2 text-sm text-red-600';
            return;
        }

        fetchStatus.textContent = 'Fetching data...';
        fetchStatus.className = 'mt-2 text-sm text-gray-600';

        fetch(`/api/fetch_main_patient_data/${uhid}`)
            .then(response => response.json())
            .then(data => {
                if (data.patient_exists) {
                    fetchStatus.textContent = 'Patient already exists. Redirecting to their chart...';
                    fetchStatus.className = 'mt-2 text-sm text-blue-600';
                    window.location.href = data.redirect_url;
                } else if (data.error) {
                    fetchStatus.textContent = `Error: ${data.error}`;
                    fetchStatus.className = 'mt-2 text-sm text-red-600';
                } else {
                    document.getElementById('form-uhid').value = data.uhid;
                    document.getElementById('form-name').value = data.name;
                    document.getElementById('form-age').value = data.age;
                    document.getElementById('form-gender').value = data.gender;
                    document.getElementById('form-contact').value = data.contact;

                    document.getElementById('summary-name').textContent = data.name;
                    document.getElementById('summary-uhid').textContent = data.uhid;
                    document.getElementById('summary-age').textContent = data.age;
                    document.getElementById('summary-gender').textContent = data.gender;

                    uhidStep.classList.add('hidden');
                    patientForm.classList.remove('hidden');
                }
            })
            .catch(err => {
                fetchStatus.textContent = 'An error occurred while fetching data.';
                fetchStatus.className = 'mt-2 text-sm text-red-600';
            });
    });
});
</script>
{% endblock %}
"""

patient_detail_html = """
{% extends 'base.html' %}
{% block title %}Patient Details{% endblock %}
{% block content %}
<div class="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8">
    <div>
        <h2 class="text-3xl sm:text-4xl font-bold heading">{{ patient.name }}</h2>
        <p class="text-gray-500">{{ patient.uhid }} | {{ patient.age }} | {{ patient.gender }}</p>
    </div>
    <div class="flex space-x-4 mt-4 sm:mt-0">
        <a href="{{ url_for('edit_patient', patient_id=patient.id) }}" class="btn btn-secondary"><i class="fas fa-edit mr-2"></i>Edit Demographics</a>
    </div>
</div>

<div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
    <div class="lg:col-span-2 space-y-8">
        <!-- Clinical Notes (SOAP) -->
        <div class="card p-6">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-bold subheading">Clinical Notes (SOAP)</h3>
                <a href="{{ url_for('add_note', patient_id=patient.id) }}" class="btn btn-primary text-sm"><i class="fas fa-plus mr-2"></i>New Note</a>
            </div>
            {% for note in patient.clinical_notes.order_by(ClinicalNote.authored_on.desc()).limit(3) %}
            <div class="border-t pt-4 mt-4">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="font-semibold">{{ note.note_type }} on {{ note.authored_on.strftime('%Y-%m-%d %H:%M') }}</p>
                        <p class="text-sm text-gray-500">by {{ note.authored_by }}</p>
                    </div>
                    <a href="{{ url_for('view_note', note_id=note.id) }}" class="btn btn-secondary text-sm">View Full Note</a>
                </div>
                <p class="text-sm mt-3"><b>Assessment:</b> {{ note.assessment | truncate(200) }}</p>
            </div>
            {% else %}
            <p class="text-gray-500">No clinical notes recorded.</p>
            {% endfor %}
        </div>

        <!-- Conditions (Diagnoses) -->
        <div class="card p-6">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-bold subheading">Conditions / Diagnoses</h3>
                <button id="open-condition-modal-btn" class="btn btn-primary text-sm"><i class="fas fa-plus mr-2"></i>Add Diagnosis</a>
            </div>
             <ul class="space-y-2">
             {% for condition in patient.conditions.order_by(Condition.onset_date.desc()) %}
                 <li class="p-2 bg-gray-50 rounded-md flex justify-between items-center">
                     <div>
                         <span class="font-semibold">{{ condition.display_text }}</span>
                         {% if condition.status == 'active' %}
                             <span class="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">Active</span>
                         {% else %}
                             <span class="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-100 text-gray-800">Inactive</span>
                         {% endif %}
                         <br>
                         <span class="text-sm text-gray-500">Onset: {{ condition.onset_date.strftime('%Y-%m-%d') if condition.onset_date else 'N/A' }}</span>
                     </div>
                     <div>
                         <form action="{{ url_for('update_condition_status', condition_id=condition.id) }}" method="POST" class="inline">
                             {% if condition.status == 'active' %}
                                 <input type="hidden" name="status" value="inactive">
                                 <button type="submit" class="btn btn-secondary text-xs">Mark Inactive</button>
                             {% else %}
                                 <input type="hidden" name="status" value="active">
                                 <button type="submit" class="btn btn-secondary text-xs">Mark Active</button>
                             {% endif %}
                         </form>
                     </div>
                 </li>
             {% else %}
                  <p class="text-gray-500">No conditions recorded.</p>
             {% endfor %}
             </ul>
        </div>

        <!-- Observations (PFTs) -->
        <div class="card p-6">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-bold subheading">Pulmonary Function Tests (PFTs)</h3>
                <a href="{{ url_for('add_observation', patient_id=patient.id) }}" class="btn btn-primary text-sm"><i class="fas fa-plus mr-2"></i>Add PFT</a>
            </div>
            <canvas id="pftChart" class="mb-4" style="max-height: 250px;"></canvas>
            <div class="overflow-x-auto">
                <table class="w-full text-sm text-left">
                    <thead class="bg-gray-50"><tr>
                        <th class="p-3 font-semibold text-gray-600">Date</th>
                        <th class="p-3 font-semibold text-gray-600">FVC (L)</th>
                        <th class="p-3 font-semibold text-gray-600">FEV1 (L)</th>
                        <th class="p-3 font-semibold text-gray-600">Ratio</th>
                    </tr></thead>
                    <tbody>
                    {% for pft in patient.observations.order_by(Observation.test_date.desc()) %}
                        <tr class="border-b"><td class="p-3">{{ pft.test_date.strftime('%Y-%m-%d') }}</td><td class="p-3">{{ "%.2f"|format(pft.fvc_value) }}</td><td class="p-3">{{ "%.2f"|format(pft.fev1_value) }}</td><td class="p-3">{{ "%.2f"|format(pft.fev1_fvc_ratio) }}</td></tr>
                    {% else %}
                        <tr><td colspan="4" class="p-3 text-center text-gray-500">No PFT records found.</td></tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    <div class="space-y-8">
        <!-- Patient Status Management -->
        <div class="card p-6">
            <h3 class="text-xl font-bold subheading mb-4">Patient Status</h3>
            <div class="text-center">
                {% if patient.status == 'Inpatient' %}
                    <p class="text-2xl font-bold text-blue-600 mb-4"><i class="fas fa-bed mr-2"></i>Inpatient</p>
                    <form action="{{ url_for('update_patient_status', patient_id=patient.id) }}" method="POST">
                        <input type="hidden" name="status" value="Outpatient">
                        <button type="submit" class="btn btn-primary w-full bg-green-600 hover:bg-green-700">Discharge Patient</button>
                    </form>
                {% else %}
                    <p class="text-2xl font-bold text-green-600 mb-4"><i class="fas fa-walking mr-2"></i>Outpatient</p>
                    <form action="{{ url_for('update_patient_status', patient_id=patient.id) }}" method="POST">
                        <input type="hidden" name="status" value="Inpatient">
                        <button type="submit" class="btn btn-primary w-full bg-blue-600 hover:bg-blue-700">Admit as Inpatient</button>
                    </form>
                {% endif %}
            </div>
        </div>
        <!-- Patient Summary -->
        <div class="card p-6">
             <h3 class="text-xl font-bold subheading mb-4">Patient Summary</h3>
             <p><strong>Smoking:</strong> {{ patient.smoking_status }} ({{ patient.pack_years }} pack-years)</p>
             <p><strong>Asthma History:</strong> {{ patient.history_of_asthma }}</p>
        </div>
        <!-- Record Edit History -->
        <div class="card p-6">
            <h3 class="text-xl font-bold subheading mb-4">Record Edit History</h3>
            <div class="overflow-y-auto max-h-60">
                <ul class="space-y-3">
                {% for log in edit_history %}
                    <li class="text-sm border-l-2 pl-3 {% if 'Created' in log.action %}border-green-500{% elif 'Updated' in log.action %}border-yellow-500{% else %}border-gray-400{% endif %}">
                        <p class="font-semibold text-gray-700">{{ log.action }}</p>
                        <p class="text-xs text-gray-500">by {{ log.username }} on {{ log.timestamp.strftime('%Y-%m-%d %H:%M') }}</p>
                    </li>
                {% else %}
                    <p class="text-gray-500">No edit history found.</p>
                {% endfor %}
                </ul>
            </div>
        </div>
        <!-- COPD Risk Assessment -->
        <div class="card p-6">
            <h3 class="text-xl font-bold subheading mb-4">COPD Risk Assessment</h3>
            <button id="predict-btn" data-patient-id="{{ patient.id }}" class="btn btn-primary w-full">Analyze Risk</button>
            <div id="prediction-result" class="mt-4 text-center"></div>
        </div>
    </div>
</div>

<!-- Modals for Condition Requests -->
<div id="condition-request-modal" class="fixed inset-0 z-50 items-center justify-center hidden">
    <div class="modal-backdrop fixed inset-0"></div>
    <div class="relative card p-8 w-full max-w-lg mx-auto mt-20">
        <h3 class="text-2xl font-bold heading mb-6">Add New Condition / Diagnosis</h3>
        <form id="condition-request-form">
            <div class="space-y-4">
                <div>
                    <label for="req-condition-code" class="block text-sm font-medium text-gray-700">Diagnosis (SNOMED CT)</label>
                    <select id="req-condition-code" class="mt-1 block w-full p-2 bg-white border rounded-md">
                        <option value="">-- Select --</option>
                        {% for code, text in snomed_codes.items() %}
                        <option value="{{ code }}">{{ text }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <label for="req-onset-date" class="block text-sm font-medium text-gray-700">Approximate Onset Date</label>
                    <input type="date" id="req-onset-date" class="mt-1 block w-full p-2 bg-white border rounded-md">
                </div>
            </div>
            <div class="mt-8 flex justify-end space-x-4">
                <button type="button" class="btn btn-secondary close-modal-btn">Cancel</button>
                <button type="submit" class="btn btn-primary">Add Condition</button>
            </div>
        </form>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const patientId = {{ patient.id }};

    // COPD Prediction Logic
    const predictBtn = document.getElementById('predict-btn');
    if(predictBtn) {
        predictBtn.addEventListener('click', function() {
            const resultDiv = document.getElementById('prediction-result');
            resultDiv.innerHTML = '<p class="text-gray-500">Analyzing...</p>';
            fetch(`/predict_copd/${patientId}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultDiv.innerHTML = `<p class="text-red-500 font-bold">${data.error}</p>`;
                } else {
                    let riskLevel = 'Low'; let textColor = 'text-green-600';
                    if (data.probability > 0.7) { riskLevel = 'High'; textColor = 'text-red-600'; }
                    else if (data.probability > 0.4) { riskLevel = 'Moderate'; textColor = 'text-yellow-600'; }
                    resultDiv.innerHTML = `<p class="text-lg font-bold ${textColor}">${riskLevel} Risk</p><p class="text-sm text-gray-600">Prob. of COPD: <strong>${(data.probability * 100).toFixed(1)}%</strong></p><p class="text-xs text-gray-400 mt-2">Based on the most recent PFT.</p>`;
                }
            });
        });
    }

    // PFT Chart Logic
    const pftCtx = document.getElementById('pftChart');
    if (pftCtx) {
        fetch(`/api/patient/${patientId}/pft_data`)
            .then(response => response.json())
            .then(pftData => {
                new Chart(pftCtx.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: pftData.labels,
                        datasets: [
                            { label: 'FVC (L)', data: pftData.fvc, borderColor: '#3b82f6', tension: 0.1 },
                            { label: 'FEV1 (L)', data: pftData.fev1, borderColor: '#ef4444', tension: 0.1 },
                            { label: 'FEV1/FVC Ratio', data: pftData.ratio, borderColor: '#14b8a6', tension: 0.1 }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } }
                });
            });
    }

    // Modal Logic
    const conditionModal = document.getElementById('condition-request-modal');
    document.getElementById('open-condition-modal-btn').addEventListener('click', () => conditionModal.style.display = 'flex');
    document.querySelectorAll('.close-modal-btn').forEach(btn => btn.addEventListener('click', () => {
        conditionModal.style.display = 'none';
    }));

    document.getElementById('condition-request-form').addEventListener('submit', function(e) {
        e.preventDefault();
        const requestData = {
            condition_code: document.getElementById('req-condition-code').value,
            onset_date: document.getElementById('req-onset-date').value
        };
        fetch(`/api/patient/${patientId}/add_condition`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(requestData) })
        .then(res => res.json()).then(data => {
            conditionModal.style.display = 'none';
            if(data.success) {
                window.location.reload();
            }
        });
    });
});
</script>
{% endblock %}
"""

patient_edit_form_html = """
{% extends 'base.html' %}
{% block title %}Edit Patient Record{% endblock %}
{% block content %}
<h2 class="text-3xl font-bold heading mb-8">Edit Record for {{ patient.name }}</h2>
<div class="card p-8">
    <form method="POST" action="">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-x-12 gap-y-8">
            <!-- Left Column: Core Details -->
            <div class="space-y-6">
                <h3 class="text-xl font-bold subheading border-b pb-2">Core Pulmonology Details</h3>
                <div>
                    <label for="smoking_status" class="block text-sm font-medium text-gray-700">Smoking Status</label>
                    <select name="smoking_status" class="mt-1 block w-full p-2 bg-gray-50 border rounded-md">
                        <option {{ 'selected' if patient.smoking_status == 'Never Smoked' }}>Never Smoked</option>
                        <option {{ 'selected' if patient.smoking_status == 'Former Smoker' }}>Former Smoker</option>
                        <option {{ 'selected' if patient.smoking_status == 'Current Smoker' }}>Current Smoker</option>
                    </select>
                </div>
                <div>
                    <label for="pack_years" class="block text-sm font-medium text-gray-700">Pack Years</label>
                    <input type="number" name="pack_years" value="{{ patient.pack_years }}" class="mt-1 block w-full p-2 bg-gray-50 border rounded-md">
                </div>
                <div>
                    <label for="history_of_asthma" class="block text-sm font-medium text-gray-700">History of Asthma</label>
                    <select name="history_of_asthma" class="mt-1 block w-full p-2 bg-gray-50 border rounded-md">
                        <option {{ 'selected' if patient.history_of_asthma == 'No' }}>No</option>
                        <option {{ 'selected' if patient.history_of_asthma == 'Yes' }}>Yes</option>
                    </select>
                </div>
            </div>

            <!-- Right Column: Structured Data -->
            <div class="space-y-6">
                <!-- Add New Condition -->
                <h3 class="text-xl font-bold subheading border-b pb-2">Add New Condition (Diagnosis)</h3>
                <div class="p-4 bg-gray-50 rounded-lg border">
                    <div>
                        <label for="condition_code" class="block text-sm font-medium text-gray-700">Diagnosis (SNOMED CT)</label>
                        <select name="condition_code" class="mt-1 block w-full p-2 bg-white border rounded-md">
                            <option value="">-- Select --</option>
                            {% for code, text in snomed_codes.items() %}
                            <option value="{{ code }}">{{ text }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="mt-4">
                        <label for="onset_date" class="block text-sm font-medium text-gray-700">Approximate Onset Date</label>
                        <input type="date" name="onset_date" class="mt-1 block w-full p-2 bg-white border rounded-md">
                    </div>
                </div>
            </div>
        </div>

        <div class="mt-12 flex justify-end space-x-4">
            <a href="{{ url_for('patient_detail', patient_id=patient.id) }}" class="btn btn-secondary">Cancel</a>
            <button type="submit" class="btn btn-primary">Save Changes</button>
        </div>
    </form>
</div>
{% endblock %}
"""

manage_users_html = """
{% extends 'base.html' %}
{% block title %}Manage Users{% endblock %}
{% block content %}
<h2 class="text-3xl sm:text-4xl font-bold heading mb-8">Manage Users</h2>
<div class="card p-8">
    <h3 class="text-2xl font-bold subheading mb-6">Add New User</h3>
    <form method="POST" action="{{ url_for('manage_users') }}" class="mb-8">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div>
                <label for="username" class="block text-sm font-medium text-gray-700">Username</label>
                <input type="text" id="username" name="username" placeholder="Username" required class="mt-1 w-full px-4 py-2 bg-gray-50 border rounded-md">
            </div>
            <div>
                <label for="password" class="block text-sm font-medium text-gray-700">Password</label>
                <input type="password" id="password" name="password" placeholder="Password" required class="mt-1 w-full px-4 py-2 bg-gray-50 border rounded-md">
            </div>
            <div>
                <label for="role" class="block text-sm font-medium text-gray-700">Role</label>
                <select id="role" name="role" class="mt-1 w-full px-4 py-2 bg-gray-50 border rounded-md">
                    <option value="{{ Role.DOCTOR }}">Doctor</option>
                    <option value="{{ Role.HEALTH_WORKER }}">Health Worker</option>
                    <option value="{{ Role.IT_EXECUTIVE }}">IT Executive</option>
                </select>
            </div>
        </div>
        <button type="submit" class="btn btn-primary mt-4">Add User</button>
    </form>
    <hr class="my-8">
    <h3 class="text-2xl font-bold subheading mb-6">Existing Users</h3>
    <div class="overflow-x-auto">
        <table class="w-full text-left">
            <thead class="bg-gray-50">
                <tr>
                    <th class="p-4">Username</th>
                    <th class="p-4">Role</th>
                    <th class="p-4">Actions</th>
                </tr>
            </thead>
            <tbody>
                {% for user in users %}
                <tr>
                    <td class="p-4">{{ user.username }}</td>
                    <td class="p-4">{{ user.role }}</td>
                    <td class="p-4"><a href="{{ url_for('delete_user', user_id=user.id) }}" class="text-red-500 hover:underline">Delete</a></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>
{% endblock %}
"""

add_observation_html = """
{% extends 'base.html' %}
{% block title %}Add PFT Observation{% endblock %}
{% block content %}
<h2 class="text-3xl font-bold heading mb-8">Add PFT for {{ patient.name }}</h2>
<div class="card p-8">
    <form method="POST" action="">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div><label for="test_date" class="block text-sm font-medium text-gray-700">Test Date</label><input type="date" name="test_date" required class="mt-1 block w-full p-2 bg-gray-50 border rounded-md"></div>
            <div></div> <!-- Spacer -->

            <div class="md:col-span-2 font-semibold text-lg subheading border-b pb-2 mb-2">Pre-Bronchodilator</div>
            <div><label for="fvc_value" class="block text-sm font-medium text-gray-700">FVC (L)</label><input type="number" step="0.01" name="fvc_value" required class="mt-1 block w-full p-2 bg-gray-50 border rounded-md"></div>
            <div><label for="fev1_value" class="block text-sm font-medium text-gray-700">FEV1 (L)</label><input type="number" step="0.01" name="fev1_value" required class="mt-1 block w-full p-2 bg-gray-50 border rounded-md"></div>

            <div class="md:col-span-2 font-semibold text-lg subheading border-b pb-2 mb-2 mt-4">Predicted Values</div>
            <div><label for="fvc_predicted_value" class="block text-sm font-medium text-gray-700">Predicted FVC (L)</label><input type="number" step="0.01" name="fvc_predicted_value" class="mt-1 block w-full p-2 bg-gray-50 border rounded-md"></div>
            <div><label for="fev1_predicted_value" class="block text-sm font-medium text-gray-700">Predicted FEV1 (L)</label><input type="number" step="0.01" name="fev1_predicted_value" class="mt-1 block w-full p-2 bg-gray-50 border rounded-md"></div>

            <div class="md:col-span-2 font-semibold text-lg subheading border-b pb-2 mb-2 mt-4">Post-Bronchodilator</div>
            <div><label for="post_bronchodilator_fev1" class="block text-sm font-medium text-gray-700">Post-Bronchodilator FEV1 (L)</label><input type="number" step="0.01" name="post_bronchodilator_fev1" class="mt-1 block w-full p-2 bg-gray-50 border rounded-md"></div>
        </div>
        <div class="mt-10 flex justify-end space-x-4">
            <a href="{{ url_for('patient_detail', patient_id=patient.id) }}" class="btn btn-secondary">Cancel</a>
            <button type="submit" class="btn btn-primary">Save PFT</button>
        </div>
    </form>
</div>
{% endblock %}
"""

note_form_html = """
{% extends 'base.html' %}
{% block title %}Clinical Note{% endblock %}
{% block content %}
<h2 class="text-3xl font-bold heading mb-8">New Clinical Note for {{ patient.name }}</h2>
<div class="card p-8">
    <form method="POST" action="">
        <div class="space-y-6">
            <div>
                <label for="note_type" class="block text-sm font-medium text-gray-700">Note Type</label>
                <input type="text" name="note_type" value="Pulmonology Consult Note" required class="mt-1 block w-full p-2 bg-gray-50 border rounded-md">
            </div>
            <div>
                <label for="subjective" class="block text-sm font-medium text-gray-700">Subjective</label>
                <textarea name="subjective" rows="4" class="mt-1 block w-full p-2 bg-gray-50 border rounded-md" placeholder="Patient's reported symptoms, history of present illness..."></textarea>
            </div>
            <div>
                <label for="objective" class="block text-sm font-medium text-gray-700">Objective</label>
                <textarea name="objective" rows="4" class="mt-1 block w-full p-2 bg-gray-50 border rounded-md" placeholder="Vital signs, physical exam findings, recent lab/imaging results..."></textarea>
            </div>
            <div>
                <label for="assessment" class="block text-sm font-medium text-gray-700">Assessment</label>
                <textarea name="assessment" rows="3" required class="mt-1 block w-full p-2 bg-gray-50 border rounded-md" placeholder="Summary of the clinical situation and diagnoses..."></textarea>
            </div>
            <div>
                <label for="plan" class="block text-sm font-medium text-gray-700">Plan</label>
                <textarea name="plan" rows="5" required class="mt-1 block w-full p-2 bg-gray-50 border rounded-md" placeholder="Further tests, medication changes, referrals, patient education..."></textarea>
            </div>
        </div>
        <div class="mt-10 flex justify-end space-x-4">
            <a href="{{ url_for('patient_detail', patient_id=patient.id) }}" class="btn btn-secondary">Cancel</a>
            <button type="submit" class="btn btn-primary">Save Note</button>
        </div>
    </form>
</div>
{% endblock %}
"""

note_detail_html = """
{% extends 'base.html' %}
{% block title %}Clinical Note Details{% endblock %}
{% block content %}
<div class="flex justify-between items-start mb-8">
    <div>
        <h2 class="text-3xl font-bold heading">Clinical Note for {{ note.patient.name }}</h2>
        <p class="text-gray-500">Note taken on {{ note.authored_on.strftime('%Y-%m-%d %H:%M') }}</p>
    </div>
    <a href="{{ url_for('patient_detail', patient_id=note.patient.id) }}" class="btn btn-secondary"><i class="fas fa-arrow-left mr-2"></i>Back to Patient Chart</a>
</div>

<div class="card p-8">
    <div class="border-b pb-4 mb-6">
        <h3 class="text-xl font-bold subheading mb-2">Note Details</h3>
        <p><strong>Note Type:</strong> {{ note.note_type }}</p>
        <p><strong>Authored By:</strong> {{ note.authored_by }}</p>
        <p><strong>Authored On:</strong> {{ note.authored_on.strftime('%Y-%m-%d %H:%M:%S') }}</p>
    </div>

    <div class="space-y-6">
        <div>
            <h4 class="font-semibold text-lg text-gray-800 mb-2">Subjective</h4>
            <div class="prose max-w-none p-4 bg-gray-50 rounded-md border text-gray-700">
                <p>{{ note.subjective | nl2br | safe if note.subjective else 'N/A' }}</p>
            </div>
        </div>
        <div>
            <h4 class="font-semibold text-lg text-gray-800 mb-2">Objective</h4>
            <div class="prose max-w-none p-4 bg-gray-50 rounded-md border text-gray-700">
                <p>{{ note.objective | nl2br | safe if note.objective else 'N/A' }}</p>
            </div>
        </div>
        <div>
            <h4 class="font-semibold text-lg text-gray-800 mb-2">Assessment</h4>
            <div class="prose max-w-none p-4 bg-gray-50 rounded-md border text-gray-700">
                <p>{{ note.assessment | nl2br | safe }}</p>
            </div>
        </div>
        <div>
            <h4 class="font-semibold text-lg text-gray-800 mb-2">Plan</h4>
            <div class="prose max-w-none p-4 bg-gray-50 rounded-md border text-gray-700">
                <p>{{ note.plan | nl2br | safe }}</p>
            </div>
        </div>
    </div>
</div>
{% endblock %}
"""

audit_log_html = """
{% extends 'base.html' %}
{% block title %}Audit Log{% endblock %}
{% block content %}
<h2 class="text-3xl font-bold heading mb-8">System Audit Log</h2>
<div class="card p-4">
    <div class="overflow-x-auto">
        <table class="w-full text-left">
            <thead class="bg-gray-50">
                <tr>
                    <th class="p-4 font-semibold text-gray-600">Timestamp (IST)</th>
                    <th class="p-4 font-semibold text-gray-600">User</th>
                    <th class="p-4 font-semibold text-gray-600">Action</th>
                    <th class="p-4 font-semibold text-gray-600">Target</th>
                </tr>
            </thead>
            <tbody class="divide-y">
            {% for log in logs %}
                <tr class="hover:bg-gray-50 text-sm">
                    <td class="p-4">{{ log.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</td>
                    <td class="p-4">{{ log.username }}</td>
                    <td class="p-4">{{ log.action }}</td>
                    <td class="p-4">{{ log.target_type }} #{{log.target_id}}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
</div>
{% endblock %}
"""

# ----------------- NEW: Lab & Radiology HTML Templates ----------------- #
lab_request_html = """
{% extends 'base.html' %}
{% block title %}Laboratory Test Request{% endblock %}

{% block head_extra %}
<script src="https://unpkg.com/feather-icons"></script>
{% endblock %}

{% block content %}
<div class="text-center mb-8">
    <h1 class="text-3xl sm:text-4xl font-bold heading mb-2">Laboratory Test Request</h1>
    <p class="text-gray-600">Request tests from the Central Laboratory System.</p>
</div>

<div class="max-w-4xl mx-auto">
    <div class="card p-8 mb-8">
        <h2 class="text-2xl font-semibold mb-6 subheading">New Test Request</h2>
       
        <form method="POST" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Requesting Department</label>
                    <div class="w-full border border-gray-300 rounded-lg px-4 py-3 bg-gray-50">
                        <span class="font-medium">PULMONOLOGY</span>
                    </div>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Patient UHID</label>
                    <input type="text" name="uhid" class="w-full border border-gray-300 rounded-lg px-4 py-3" placeholder="Enter Patient UHID" required>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Priority</label>
                    <select name="priority" class="w-full border border-gray-300 rounded-lg px-4 py-3" required>
                        <option value="routine">Routine</option>
                        <option value="urgent">Urgent</option>
                        <option value="stat">STAT (Immediate)</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Specimen Type</label>
                    <input type="text" name="specimen" class="w-full border border-gray-300 rounded-lg px-4 py-3" value="Blood" required>
                </div>
            </div>

            <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Clinical Notes</label>
                <textarea name="clinical_notes" rows="3" class="w-full border border-gray-300 rounded-lg px-4 py-3" placeholder="Enter any clinical notes or special instructions"></textarea>
            </div>

            <div>
                <label class="block text-sm font-medium text-gray-700 mb-4">Select Tests</label>
                <div class="mb-6">
                    <h3 class="text-lg font-medium text-gray-800 mb-3 flex items-center"><i data-feather="flask-conical" class="w-5 h-5 mr-2 text-blue-600"></i>Biochemistry</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {% for category, tests in test_categories['biochemistry'].items() %}
                        <div class="border border-gray-200 rounded-lg p-4">
                        <h4 class="font-medium text-gray-700 mb-2">{{category}}</h4>
                        <div class="space-y-2">
                        {% for test in tests %}
                            <label class="flex items-center">
                                <input type="checkbox" name="tests" value="{{test}}" class="rounded border-gray-300 text-indigo-600">
                                <span class="ml-2 text-sm text-gray-600">{{test}}</span>
                            </label>
                        {% endfor %}
                        </div></div>
                    {% endfor %}
                    </div>
                </div>
                 <div class="mb-6">
                    <h3 class="text-lg font-medium text-gray-800 mb-3 flex items-center"><i data-feather="microscope" class="w-5 h-5 mr-2 text-green-600"></i>Microbiology</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {% for category, tests in test_categories['microbiology'].items() %}
                       <div class="border border-gray-200 rounded-lg p-4">
                        <h4 class="font-medium text-gray-700 mb-2">{{category}}</h4>
                        <div class="space-y-2">
                        {% for test in tests %}
                            <label class="flex items-center">
                                <input type="checkbox" name="tests" value="{{test}}" class="rounded border-gray-300 text-indigo-600">
                                <span class="ml-2 text-sm text-gray-600">{{test}}</span>
                            </label>
                        {% endfor %}
                        </div></div>
                    {% endfor %}
                    </div>
                </div>
                 <div class="mb-6">
                    <h3 class="text-lg font-medium text-gray-800 mb-3 flex items-center"><i data-feather="activity" class="w-5 h-5 mr-2 text-red-600"></i>Pathology</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {% for category, tests in test_categories['pathology'].items() %}
                       <div class="border border-gray-200 rounded-lg p-4">
                        <h4 class="font-medium text-gray-700 mb-2">{{category}}</h4>
                        <div class="space-y-2">
                        {% for test in tests %}
                            <label class="flex items-center">
                                <input type="checkbox" name="tests" value="{{test}}" class="rounded border-gray-300 text-indigo-600">
                                <span class="ml-2 text-sm text-gray-600">{{test}}</span>
                            </label>
                        {% endfor %}
                        </div></div>
                    {% endfor %}
                    </div>
                </div>
            </div>
            <div class="flex items-center justify-between pt-4">
                <button type="submit" class="btn btn-primary"><i data-feather="send" class="w-5 h-5 mr-2 inline"></i>Submit Test Request</button>
                <a href="{{ url_for('lab_history') }}" class="font-semibold text-indigo-600 hover:underline">View History</a>
            </div>
        </form>
    </div>

    {% if order_id %}
    <div id="resultsSection" class="card p-8 mb-8">
        <h2 class="text-2xl font-semibold mb-6 subheading">Order Status</h2>
        <div class="border w-full p-4 bg-gray-50 rounded">
            <input type="hidden" id="currentOrderId" value="{{ order_id }}" />
            <div class="flex items-center justify-between">
                <div>
                    <strong>Order ID:</strong> <span id="orderIdDisplay">{{ order_id }}</span><br>
                    <strong>Status:</strong> <span id="currentStatus" class="text-blue-600 font-medium">Queued</span>
                </div>
                <div class="flex space-x-2">
                    <button onclick="checkStatus()" class="btn btn-secondary">Check Status</button>
                    <a id="viewResultsBtn" href="#" class="hidden btn btn-primary">View Results</a>
                </div>
            </div>
            <div id="statusDetails" class="mt-3"></div>
        </div>
    </div>
    {% endif %}

</div>

<script>
    document.addEventListener('DOMContentLoaded', function() {
        feather.replace();
       
        document.querySelector('form').addEventListener('submit', function(e) {
            const selectedTests = document.querySelectorAll('input[name="tests"]:checked');
            if (selectedTests.length === 0) {
                e.preventDefault();
                alert('Please select at least one test.');
                return false;
            }
        });

        // If results section is present, scroll to it
        if (document.getElementById('resultsSection')) {
            document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
        }
    });

    function checkStatus() {
        const orderId = document.getElementById('currentOrderId').value;
        if (!orderId) return;

        const statusButton = event.target;
        statusButton.innerHTML = 'Checking...';
        statusButton.disabled = true;

        fetch(`/api/lab/status/${orderId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error checking status: ' + data.error);
                } else {
                    updateStatusDisplay(data);
                }
            })
            .finally(() => {
                statusButton.innerHTML = 'Check Status';
                statusButton.disabled = false;
            });
    }

    function updateStatusDisplay(data) {
        const statusText = document.getElementById('currentStatus');
        const statusDetails = document.getElementById('statusDetails');
        const viewBtn = document.getElementById('viewResultsBtn');
        const orderId = document.getElementById('currentOrderId').value;
       
        if (data.status === 'completed') {
            statusText.textContent = 'Completed';
            statusText.className = 'text-green-600 font-medium';
            viewBtn.href = `/lab/results/${orderId}`;
            viewBtn.classList.remove('hidden');
        } else {
            statusText.textContent = 'In Progress';
            statusText.className = 'text-yellow-600 font-medium';
            viewBtn.classList.add('hidden');
        }
    }
</script>
{% endblock %}
"""

lab_history_html = """
{% extends 'base.html' %}
{% block title %}Lab Order History{% endblock %}
{% block content %}
<div class="flex items-center justify-between mb-8">
    <div>
        <h1 class="text-3xl font-bold heading">Lab Order History</h1>
        <p class="text-gray-600">Department: <span class="font-medium">PULMONOLOGY</span></p>
    </div>
    <a class="btn btn-secondary" href="{{ url_for('lab_request') }}">New Request</a>
</div>
<div class="card overflow-hidden">
    <table class="w-full text-sm">
        <thead class="bg-gray-50 text-gray-600">
            <tr>
                <th class="text-left px-4 py-3">Order ID</th>
                <th class="text-left px-4 py-3">UHID</th>
                <th class="text-left px-4 py-3">Priority</th>
                <th class="text-left px-4 py-3">Created</th>
                <th class="text-left px-4 py-3">Actions</th>
            </tr>
        </thead>
        <tbody>
        {% for h in history %}
            <tr class="border-t">
                <td class="px-4 py-3 font-medium">{{ h.orderId }}</td>
                <td class="px-4 py-3">{{ h.uhid }}</td>
                <td class="px-4 py-3 uppercase">{{ h.priority }}</td>
                <td class="px-4 py-3">{{ h.createdAt.split('T')[0] }}</td>
                <td class="px-4 py-3">
                     <a class="font-semibold text-indigo-600 hover:underline" href="{{ url_for('lab_results', order_id=h.orderId) }}">View Details</a>
                </td>
            </tr>
        {% else %}
             <tr><td colspan="5" class="text-center p-8 text-gray-500">No orders found in history.</td></tr>
        {% endfor %}
        </tbody>
    </table>
</div>
{% endblock %}
"""

lab_results_html = """
{% extends 'base.html' %}
{% block title %}Lab Results{% endblock %}
{% block content %}
<div class="mb-6 flex items-center justify-between">
    <h1 class="text-3xl font-bold heading">Order Results • {{ data.orderId }}</h1>
    <a href="{{ url_for('lab_history') }}" class="btn btn-secondary">Back to History</a>
</div>
<div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
    <div class="card p-4"><div class="text-sm text-gray-500">Patient</div><div class="font-medium">{{ data.patient.name if data.patient else 'N/A' }}</div></div>
    <div class="card p-4"><div class="text-sm text-gray-500">Priority</div><div class="font-medium">{{ (data.priority or 'routine')|upper }}</div></div>
    <div class="card p-4"><div class="text-sm text-gray-500">Requested</div><div class="font-medium">{{ data.receivedAt.split('T')[0] }}</div></div>
</div>

{% for dept in data.perDepartment %}
<div class="card p-6 mb-6">
    <div class="flex items-center justify-between mb-4">
        <h2 class="text-xl font-semibold subheading">{{ dept.department|title }}</h2>
        <span class="text-sm px-3 py-1 rounded-full {{ 'bg-green-100 text-green-700' if dept.status=='completed' else 'bg-yellow-100 text-yellow-700' }}">{{ dept.status.replace('_',' ') }}</span>
    </div>
    {% if dept.results and dept.results|length > 0 %}
        <div class="overflow-x-auto">
            <table class="w-full text-sm">
                <thead><tr class="text-left border-b"><th class="py-2 pr-4">Test</th><th class="py-2 pr-4">Value</th><th class="py-2 pr-4">Unit</th><th class="py-2 pr-4">Flag</th><th class="py-2">Ref Range</th></tr></thead>
                <tbody>
                {% for r in dept.results %}
                    <tr class="border-b">
                        <td class="py-2 pr-4">{{ r.testCode }}</td>
                        <td class="py-2 pr-4">{{ r.value }}</td>
                        <td class="py-2 pr-4">{{ r.unit }}</td>
                        <td class="py-2 pr-4">{{ r.flag }}</td>
                        <td class="py-2">{{ (r.referenceRange.low if r.referenceRange else '') }} - {{ (r.referenceRange.high if r.referenceRange else '') }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
    {% else %}
        <div class="text-gray-500 text-sm">No results yet.</div>
    {% endif %}
</div>
{% endfor %}
{% endblock %}
"""

radiology_request_html = """
{% extends 'base.html' %}
{% block title %}Radiology Scan Request{% endblock %}
{% block content %}
<div class="grid md:grid-cols-2 gap-8 items-start">
    <div class="card p-6">
        <h1 class="text-2xl font-bold text-gray-800 mb-4">New Scan Request</h1>
        <form method="POST" action="{{ url_for('radiology_request') }}" class="space-y-4">
            <input type="text" name="department" value="PULMONOLOGY" class="hidden">
            <label class="block">
                <div class="text-sm font-medium text-gray-700 mb-1">Patient UHID</div>
                <input type="text" name="uhid" class="w-full border rounded-lg px-3 py-2 bg-gray-50" placeholder="e.g. MAIN-APP-123" required>
            </label>
            <label class="block">
                <div class="text-sm font-medium text-gray-700 mb-1">Scan Type</div>
                <select name="scan_type" class="w-full border rounded-lg px-3 py-2 bg-gray-50" required>
                    <option value="" disabled selected>Select Scan Type</option>
                    <option value="CT">CT</option><option value="MR">MR</option><option value="XRAY">XRAY</option>
                    <option value="US">ULTRASOUND</option><option value="PET">PET</option>
                </select>
            </label>
            <label class="block">
                <div class="text-sm font-medium text-gray-700 mb-1">Body Part</div>
                <input type="text" name="body_part" class="w-full border rounded-lg px-3 py-2 bg-gray-50" placeholder="e.g. CHEST" required oninput="this.value=this.value.toUpperCase()">
            </label>
            <div class="flex items-center space-x-3 pt-2">
                <button type="submit" class="btn btn-primary">Queue Request</button>
                <a href="{{ url_for('radiology_view_queue') }}" class="text-sm text-indigo-600 hover:underline">Open queue</a>
            </div>
        </form>
    </div>
    <div class="space-y-4">
        <div class="card p-6">
            <h3 class="font-semibold text-gray-700 mb-2">Queue Stats</h3>
            <div class="grid grid-cols-3 gap-4 mt-3">
                <div class="p-3 rounded-lg bg-gray-50 text-center"><div class="text-xs text-gray-500">Queued</div><div id="stat-queued" class="text-2xl font-bold">0</div></div>
                <div class="p-3 rounded-lg bg-gray-50 text-center"><div class="text-xs text-gray-500">Completed</div><div id="stat-completed" class="text-2xl font-bold">0</div></div>
                <div class="p-3 rounded-lg bg-gray-50 text-center"><div class="text-xs text-gray-500">Failed</div><div id="stat-failed" class="text-2xl font-bold text-red-500">0</div></div>
            </div>
            <div class="mt-4 text-xs text-gray-500">Note: Queue is in-memory and resets on app restart.</div>
        </div>
    </div>
</div>
<script>
    async function fetchStats() {
      try {
        const res = await fetch("{{ url_for('radiology_queue_status') }}");
        if (!res.ok) return;
        const q = await res.json();
        let queued = 0, completed = 0, failed = 0;
        for (const r of q) {
          if (r.status === 'Pending') queued++;
          else if (r.status === 'Completed') completed++;
          else if (r.status === 'Failed') failed++;
        }
        document.getElementById('stat-queued').textContent = queued;
        document.getElementById('stat-completed').textContent = completed;
        document.getElementById('stat-failed').textContent = failed;
      } catch (e) {}
    }
    fetchStats();
    setInterval(fetchStats, 5000);
</script>
{% endblock %}
"""

radiology_queue_html = """
{% extends 'base.html' %}
{% block title %}Radiology Queue{% endblock %}
{% block head_extra %}
    <style>
        .spinner { border: 3px solid rgba(0,0,0,0.06); width: 18px; height: 18px; border-radius: 999px; border-left-color: #4f46e5; animation: spin 1s linear infinite;}
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
{% endblock %}
{% block content %}
<div class="card p-6">
    <h2 class="text-2xl font-bold mb-4">Radiology Request Queue</h2>
    <div id="request-queue-container" class="space-y-3"></div>
</div>

<div id="viewer-modal" class="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center p-4 hidden z-[100]">
    <div class="bg-white rounded-lg shadow-xl w-full max-w-3xl h-full max-h-[80vh] flex flex-col">
        <div class="p-3 border-b flex justify-between items-center">
            <h3 class="font-bold text-lg">DICOM Viewer</h3>
            <button id="close-modal-btn" class="text-gray-500 hover:text-gray-800 text-2xl">&times;</button>
        </div>
        <div id="dicomImage" class="w-full flex-grow bg-black"></div>
    </div>
</div>
<!-- Cornerstone libs -->
<script src="https://unpkg.com/cornerstone-core@2.3.0/dist/cornerstone.js"></script>
<script src="https://unpkg.com/dicom-parser@1.8.7/dist/dicomParser.js"></script>
<script src="https://unpkg.com/cornerstone-wado-image-loader@3.1.2/dist/cornerstoneWADOImageLoader.js"></script>
<script>
    try {
      cornerstoneWADOImageLoader.webWorkerManager.initialize({
        maxWebWorkers: navigator.hardwareConcurrency || 1,
        startWebWorkersOnDemand: true,
        taskConfiguration: { 'decodeTask': { initializeCodecsOnStartup: false, usePDFJS: false, strict: false } }
      });
      cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
    } catch (e) { console.warn('Cornerstone init failed', e); }

    const queueContainer = document.getElementById('request-queue-container');
    const modal = document.getElementById('viewer-modal');
    const dicomElement = document.getElementById('dicomImage');
    const closeModalBtn = document.getElementById('close-modal-btn');

    function renderQueue(queue) {
      if (!queue || queue.length === 0) {
        queueContainer.innerHTML = '<p class="text-center text-gray-500 py-8">No requests in the queue.</p>';
        return;
      }
      let html = '';
      for (const req of queue) {
        const when = new Date(req.timestamp).toLocaleString();
        const statusBadge = req.status === 'Pending' ? `<span class="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800">${req.status}</span>` :
                              req.status === 'Completed' ? `<span class="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">${req.status}</span>` :
                              `<span class="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800">${req.status}</span>`;
        const spinner = req.status === 'Pending' ? '<div class="spinner"></div>' : '';
        const viewBtn = (req.status === 'Completed' && req.filename) ? `<button class="view-dicom-btn btn btn-primary btn-sm text-xs" data-filename="${req.filename}">View</button>` : '';
        const errMsg = req.error ? `<div class="text-xs text-red-500 mt-1">Error: ${req.error}</div>` : '';
        html += `<div class="p-4 bg-gray-50 rounded-lg flex justify-between items-start">
                    <div>
                        <div class="flex items-center gap-3">
                            ${spinner}
                            <div>
                                <div class="font-medium">UHID: ${req.uhid} <span class="text-xs text-gray-400">• ${when}</span></div>
                                <div class="text-sm text-gray-600">${req.scan_type} — ${req.body_part}</div>
                                ${errMsg}
                            </div>
                        </div>
                    </div>
                    <div class="flex flex-col items-end gap-2">
                        ${statusBadge}
                        ${viewBtn}
                    </div>
                   </div>`;
      }
      queueContainer.innerHTML = html;
    }

    async function fetchQueueStatus() {
      try {
        const res = await fetch("{{ url_for('radiology_queue_status') }}");
        if (!res.ok) return;
        renderQueue(await res.json());
      } catch (e) {}
    }
   
    function showViewer(filename) {
        if (!filename) return;
        const imageId = `wadouri:${window.location.origin}/radiology/dicom/${encodeURIComponent(filename)}`;
        modal.classList.remove('hidden');
        try { cornerstone.enable(dicomElement); } catch(e) {}
        cornerstone.loadAndCacheImage(imageId).then(image => {
            cornerstone.displayImage(dicomElement, image);
        }).catch(err => {
            dicomElement.innerHTML = '<div class="p-4 text-red-400">Failed to load DICOM image.</div>';
        });
    }

    closeModalBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
        try { cornerstone.disable(dicomElement); } catch(e) {}
        dicomElement.innerHTML = '';
    });
   
    queueContainer.addEventListener('click', (ev) => {
        const btn = ev.target.closest('.view-dicom-btn');
        if (btn) showViewer(btn.dataset.filename);
    });

    fetchQueueStatus();
    setInterval(fetchQueueStatus, 5000); // Poll every 5 seconds
</script>
{% endblock %}
"""

# ----------------- Flask Routes ----------------- #

# --- Main/Patient Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            flash('Login successful.', 'success')
            log_entry = AuditLog(user_id=user.id, username=user.username, action="User logged in")
            db.session.add(log_entry)
            db.session.commit()
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template_string("login.html")

@app.route('/logout')
@login_required
def logout():
    log_entry = AuditLog(user_id=current_user.id, username=current_user.username, action="User logged out")
    db.session.add(log_entry)
    db.session.commit()
    logout_user()
    session.pop('department', None) # Clear department from session on logout
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    total_patients = db.session.query(Patient.id).count()
    current_inpatients = db.session.query(Patient.id).filter_by(status='Inpatient').count()
    seven_days_ago = datetime.datetime.now(IST) - datetime.timedelta(days=7)
    recent_admissions = db.session.query(Patient.id).filter(Patient.admission_date >= seven_days_ago).count()
    return render_template_string("index.html", total_patients=total_patients, current_inpatients=current_inpatients, recent_admissions=recent_admissions)

@app.route('/patients')
@login_required
@log_activity("Viewed patient list")
def patients_list():
    query = Patient.query
    search_uhid = request.args.get('search_uhid')
    search_name = request.args.get('search_name')
    if search_uhid: query = query.filter(Patient.uhid.contains(search_uhid))
    if search_name: query = query.filter(Patient.name.contains(search_name))
    patients = query.order_by(Patient.admission_date.desc()).all()
    return render_template_string("patients_list.html", patients=patients)

@app.route('/patient/add', methods=['GET', 'POST'])
@login_required
@role_required([Role.ADMIN, Role.DOCTOR, Role.HEALTH_WORKER])
def add_patient():
    if request.method == 'POST':
        if Patient.query.filter_by(uhid=request.form['uhid']).first():
            flash('A chart for this patient already exists.', 'danger')
            return redirect(url_for('patients_list'))

        new_patient = Patient(
            uhid=request.form['uhid'], name=request.form['name'], age=int(request.form['age']),
            gender=request.form['gender'], contact=request.form.get('contact'),
            smoking_status=request.form['smoking_status'], pack_years=int(request.form['pack_years']),
            history_of_asthma=request.form['history_of_asthma']
        )
        db.session.add(new_patient)
        db.session.commit()

        log_entry = AuditLog(
            user_id=current_user.id, username=current_user.username,
            action=f"Created new patient chart for {new_patient.name}",
            target_type='Patient', target_id=new_patient.id
        )
        db.session.add(log_entry)
        db.session.commit()

        flash('Patient chart created successfully!', 'success')
        return redirect(url_for('patient_detail', patient_id=new_patient.id))
    return render_template_string("patient_form.html")

@app.route('/patient/<int:patient_id>')
@login_required
@log_activity("Viewed patient chart for patient_id={patient_id}")
def patient_detail(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient: return redirect(url_for('patients_list'))
    edit_history = AuditLog.query.filter_by(target_type='Patient', target_id=patient_id).order_by(AuditLog.timestamp.desc()).all()
    return render_template_string("patient_detail.html", patient=patient, ClinicalNote=ClinicalNote, Observation=Observation, Condition=Condition, snomed_codes=SIMULATED_SNOMED_DB, edit_history=edit_history)

@app.route('/patient/<int:patient_id>/update_status', methods=['POST'])
@login_required
@role_required([Role.ADMIN, Role.DOCTOR])
@log_activity("Updated patient status for patient_id={patient_id}")
def update_patient_status(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient: return redirect(url_for('patients_list'))
   
    new_status = request.form.get('status')
    if new_status in ['Inpatient', 'Outpatient']:
        patient.status = new_status
        db.session.commit()
        flash(f'Patient status updated to {new_status}.', 'success')
    else:
        flash('Invalid status provided.', 'danger')
    return redirect(url_for('patient_detail', patient_id=patient_id))

@app.route('/patient/edit/<int:patient_id>', methods=['GET', 'POST'])
@login_required
@role_required([Role.ADMIN, Role.DOCTOR])
@log_activity("Updated patient record for patient_id={patient_id}")
def edit_patient(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient: return redirect(url_for('patients_list'))

    if request.method == 'POST':
        patient.smoking_status = request.form['smoking_status']
        patient.pack_years = int(request.form['pack_years'])
        patient.history_of_asthma = request.form['history_of_asthma']

        condition_code = request.form.get('condition_code')
        if condition_code:
            new_condition = Condition(
                patient_id_fk=patient.id, code=condition_code,
                display_text=SIMULATED_SNOMED_DB.get(condition_code, "Unknown"),
                onset_date=datetime.datetime.strptime(request.form['onset_date'], '%Y-%m-%d').date() if request.form.get('onset_date') else None
            )
            db.session.add(new_condition)
       
        db.session.commit()
        flash('Patient record updated!', 'success')
        return redirect(url_for('patient_detail', patient_id=patient.id))
    return render_template_string("patient_edit_form.html", patient=patient, snomed_codes=SIMULATED_SNOMED_DB)

@app.route('/patient/<int:patient_id>/add_observation', methods=['GET', 'POST'])
@login_required
@role_required([Role.ADMIN, Role.DOCTOR, Role.HEALTH_WORKER])
@log_activity("Added PFT Observation for patient_id={patient_id}")
def add_observation(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient: return redirect(url_for('patients_list'))

    if request.method == 'POST':
        fvc = float(request.form['fvc_value'])
        fev1 = float(request.form['fev1_value'])
        new_obs = Observation(
            patient_id_fk=patient.id,
            test_date=datetime.datetime.strptime(request.form['test_date'], '%Y-%m-%d').date(),
            fvc_value=fvc, fev1_value=fev1, fev1_fvc_ratio=fev1 / fvc if fvc > 0 else 0
        )
        db.session.add(new_obs)
        db.session.commit()
        flash('PFT Observation added!', 'success')
        return redirect(url_for('patient_detail', patient_id=patient.id))
    return render_template_string("add_observation.html", patient=patient)

@app.route('/patient/<int:patient_id>/add_note', methods=['GET', 'POST'])
@login_required
@role_required([Role.ADMIN, Role.DOCTOR])
@log_activity("Added Clinical Note for patient_id={patient_id}")
def add_note(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient: return redirect(url_for('patients_list'))
    if request.method == 'POST':
        new_note = ClinicalNote(
            patient_id_fk=patient.id, note_type=request.form['note_type'],
            authored_by=current_user.username, subjective=request.form['subjective'],
            objective=request.form['objective'], assessment=request.form['assessment'],
            plan=request.form['plan']
        )
        db.session.add(new_note)
        db.session.commit()
        flash('Clinical note saved!', 'success')
        return redirect(url_for('patient_detail', patient_id=patient.id))
    return render_template_string("note_form.html", patient=patient)

@app.route('/note/<int:note_id>')
@login_required
@log_activity("Viewed clinical note_id={note_id}")
def view_note(note_id):
    note = db.session.get(ClinicalNote, note_id)
    if not note: return redirect(url_for('patients_list'))
    return render_template_string("note_detail.html", note=note)

@app.route('/condition/<int:condition_id>/update_status', methods=['POST'])
@login_required
@role_required([Role.ADMIN, Role.DOCTOR])
def update_condition_status(condition_id):
    condition = db.session.get(Condition, condition_id)
    if not condition: return redirect(url_for('patients_list'))
    patient_id = condition.patient_id_fk
    new_status = request.form.get('status')
    if new_status in ['active', 'inactive']:
        condition.status = new_status
        db.session.commit()
        flash(f"Condition status updated.", 'success')
    return redirect(url_for('patient_detail', patient_id=patient_id))

# --- Admin Routes ---
@app.route('/manage_users', methods=['GET', 'POST'])
@login_required
@role_required(Role.ADMIN)
def manage_users():
    if request.method == 'POST':
        username = request.form['username']
        if not User.query.filter_by(username=username).first():
            new_user = User(username=username, role=request.form['role'])
            new_user.set_password(request.form['password'])
            db.session.add(new_user)
            db.session.commit()
            flash('User added!', 'success')
        else:
            flash('Username already exists.', 'danger')
        return redirect(url_for('manage_users'))
    users = User.query.all()
    return render_template_string("manage_users.html", users=users, Role=Role)

@app.route('/delete_user/<int:user_id>')
@login_required
@role_required(Role.ADMIN)
@log_activity("Admin deleted user_id={user_id}")
def delete_user(user_id):
    user = db.session.get(User, user_id)
    if user and user.username != 'admin':
        db.session.delete(user)
        db.session.commit()
        flash('User deleted.', 'success')
    return redirect(url_for('manage_users'))

@app.route('/audit_log')
@login_required
@role_required([Role.ADMIN, Role.IT_EXECUTIVE])
def view_audit_log():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).all()
    return render_template_string("audit_log.html", logs=logs)

# --- Lab Test Routes ---
@app.route('/lab', methods=['GET', 'POST'])
@login_required
def lab_request():
    order_id, error = None, None
    department = "PULMONOLOGY" 
    session['department'] = department

    if request.method == "POST":
        uhid = request.form.get("uhid")
        tests = request.form.getlist("tests")
        if not all([uhid, tests]):
            error = "UHID and at least one test are required fields."
        else:
            order_id, error = lab_perform_test_request(
                host=LAB_DEFAULT_HOST,
                department=department,
                uhid=uhid,
                tests=tests,
                priority=request.form.get("priority"),
                specimen=request.form.get("specimen"),
                clinical_notes=request.form.get("clinical_notes")
            )
    return render_template_string('lab_request.html', order_id=order_id, error=error, department=department, test_categories=TEST_CATEGORIES)


@app.route("/lab/history")
@login_required
def lab_history():
    department = "PULMONOLOGY"
    hist = lab_load_history(department)
    return render_template_string("lab_history.html", history=hist, department=department)

@app.route("/lab/results/<order_id>")
@login_required
def lab_results(order_id):
    try:
        url = f"{LAB_DEFAULT_HOST.rstrip('/')}/api/lab/orders/{order_id}"
        r = requests.get(url, headers={'X-API-Key': LAB_SHARED_API_KEY}, timeout=20)
        if not r.ok:
            flash(f"Could not load results for order {order_id} (status: {r.status_code})", "danger")
            return redirect(url_for('lab_history'))
        return render_template_string("lab_results.html", data=r.json())
    except Exception as e:
        flash(f"Error fetching results: {e}", "danger")
        return redirect(url_for('lab_history'))


# --- Radiology Routes ---
@app.route('/radiology', methods=['GET', 'POST'])
@login_required
def radiology_request():
    if request.method == "POST":
        uhid = request.form.get("uhid")
        scan_type = request.form.get("scan_type")
        body_part = request.form.get("body_part")
        department = "PULMONOLOGY"

        if not all([uhid, scan_type, body_part]):
            flash("All fields are required for a scan request.", "danger")
            return redirect(url_for('radiology_request'))

        new_request = {
            "id": str(uuid.uuid4()), "uhid": uhid, "scan_type": scan_type,
            "body_part": body_part, "status": "Pending", "filename": None,
            "error": None, "timestamp": datetime.datetime.now().isoformat()
        }
        with radiology_queue_lock:
            RADIOLOGY_REQUEST_QUEUE.insert(0, new_request)
       
        thread = threading.Thread(
            target=radiology_process_scan_request_worker,
            args=(new_request['id'], RADIOLOGY_DEFAULT_HOST, department, uhid, scan_type, body_part),
            daemon=True
        )
        thread.start()
        flash(f"Scan request for {uhid} has been queued.", "success")
        return redirect(url_for('radiology_view_queue', created_id=new_request['id'], uhid=uhid, scan_type=scan_type))

    return render_template_string("radiology_request.html")

@app.route("/radiology/view")
@login_required
def radiology_view_queue():
    return render_template_string("radiology_queue.html")


# ----------------- API Routes & External Communication ----------------- #
# (These are called by the main app, other services, or the app's own JS)

@app.route('/predict_copd/<int:patient_id>', methods=['POST'])
@login_required
def get_copd_prediction(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient: return jsonify({'error': 'Patient not found'}), 404
    latest_obs = patient.observations.order_by(Observation.test_date.desc()).first()
    if not latest_obs: return jsonify({'error': 'No PFT data available.'})
    patient_data = {
        'age': patient.age, 'smoking_status': patient.smoking_status, 'pack_years': patient.pack_years,
        'history_of_asthma': patient.history_of_asthma, 'fev1_fvc_ratio': latest_obs.fev1_fvc_ratio
    }
    prediction, probability = predict_copd(patient_data)
    return jsonify({'prediction': int(prediction), 'probability': float(probability)})

@app.route('/api/patient/<int:patient_id>/pft_data')
@login_required
def pft_data(patient_id):
    pfts = Observation.query.filter_by(patient_id_fk=patient_id).order_by(Observation.test_date).all()
    return jsonify({
        'labels': [p.test_date.strftime('%Y-%m-%d') for p in pfts],
        'fvc': [p.fvc_value for p in pfts], 'fev1': [p.fev1_value for p in pfts],
        'ratio': [p.fev1_fvc_ratio for p in pfts]
    })

@app.route('/api/fetch_main_patient_data/<string:uhid>')
@login_required
def fetch_main_patient_data(uhid):
    # Check if a real Main App API URL is configured
    main_app_url = app.config.get('MAIN_APP_API_URL')
    if main_app_url:
        try:
            # Append the UHID to the base URL
            api_url = f"{main_app_url.rstrip('/')}/{uhid}"
            response = requests.get(api_url, timeout=5)
            response.raise_for_status() # Raise an exception for bad status codes
            patient_data = response.json()

            # The real API should return data in the same format as the simulation
            # e.g., {"name": "...", "age": ..., "gender": ...}
            if patient_data:
                patient_data['uhid'] = uhid
                return jsonify(patient_data)
            return jsonify({'error': 'Patient not found in main registry (live).'}), 404

        except requests.RequestException as e:
            print(f"Error calling Main App API: {e}")
            return jsonify({'error': 'Could not connect to the main hospital system.'}), 500
   
    # --- Fallback to Simulated Data if no API URL is set ---
    patient_data = SIMULATED_MAIN_APP_DB.get(uhid)
    if patient_data:
        existing_patient = Patient.query.filter_by(uhid=uhid).first()
        if existing_patient:
            return jsonify({'patient_exists': True, 'redirect_url': url_for('patient_detail', patient_id=existing_patient.id)})
        patient_data['uhid'] = uhid
        return jsonify(patient_data)
   
    return jsonify({'error': 'Patient not found in main registry (simulated).'}), 404

@app.route('/api/patient/<int:patient_id>/add_condition', methods=['POST'])
@login_required
@role_required([Role.ADMIN, Role.DOCTOR])
@log_activity("Added condition for patient_id={patient_id}")
def add_condition(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient: return jsonify({'success': False, 'error': 'Patient not found'}), 404
    data = request.get_json()
    condition_code = data.get('condition_code')
    if not condition_code: return jsonify({'success': False, 'error': 'Code is required'}), 400
    new_condition = Condition(
        patient_id_fk=patient.id, code=condition_code,
        display_text=SIMULATED_SNOMED_DB.get(condition_code, "Unknown"),
        onset_date=datetime.datetime.strptime(data.get('onset_date'), '%Y-%m-%d').date() if data.get('onset_date') else None
    )
    db.session.add(new_condition)
    db.session.commit()
    flash('Condition added!', 'success')
    return jsonify({'success': True})

# --- NEW API Endpoint for Terminal Access ---
@app.route('/api/patient/<string:uhid>')
def get_patient_data_by_uhid(uhid):
    patient = Patient.query.filter_by(uhid=uhid).first()
    if not patient:
        return jsonify({'error': f'Patient with UHID {uhid} not found in the Pulmonology database.'}), 404
    patient_data = {
        'id': patient.id, 
        'admission_date': patient.admission_date.isoformat(), 'status': patient.status,
        'smoking_status': patient.smoking_status, 'pack_years': patient.pack_years,
        'history_of_asthma': patient.history_of_asthma,
        'observations': [{'test_date': obs.test_date.isoformat(), 'fvc_value': obs.fvc_value, 'fev1_value': obs.fev1_value, 'fev1_fvc_ratio': obs.fev1_fvc_ratio} for obs in patient.observations],
        'conditions': [{'code': cond.code, 'display_text': cond.display_text, 'onset_date': cond.onset_date.isoformat() if cond.onset_date else None, 'status': cond.status} for cond in patient.conditions],
        'clinical_notes': [{'authored_on': note.authored_on.isoformat(), 'authored_by': note.authored_by, 'assessment': note.assessment, 'plan': note.plan} for note in patient.clinical_notes]
    }
    return jsonify(patient_data)

# --- Gemini API Route ---
@app.route('/ask_gemini', methods=['POST'])
@login_required
def ask_gemini():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is missing.'}), 400

    api_key = app.config.get('GEMINI_API_KEY')
    if not api_key:
        time.sleep(1.5) 
        simulated_response = (
            "**Simulated Response:** API key not configured.\n\n"
            "As a clinical assistant AI, I can provide information based on established medical knowledge. Here is a summary based on your query:\n\n"
            "**Standard Approach:**\n* **Initial Assessment:** Begin with a thorough patient history and physical examination.\n* **Diagnostic Tests:** Consider relevant imaging (like Chest X-ray or CT) and laboratory tests (such as CBC or sputum culture).\n* **Treatment Plan:** Develop a plan based on diagnosis, which may include medication, therapy, or other interventions.\n\n"
            "**Disclaimer:** This information is for educational purposes only and is not a substitute for professional medical advice."
        )
        return jsonify({'response': simulated_response})

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
   
    system_prompt = (
        "You are a specialized medical AI assistant for healthcare professionals. Your name is SpiroSage. "
        "Your sole purpose is to answer questions related to pulmonology, general medicine, and clinical practices. "
        "Provide concise, accurate, and professional answers using markdown for formatting (e.g., **bolding**, lists with asterisks). "
        "**If a user asks a question that is not related to healthcare, medicine, or clinical topics, you must politely decline to answer and state that you are a medical assistant and can only answer healthcare-related questions.** "
        "Do not provide direct medical advice to patients. Frame your answers for a professional audience (doctors, nurses, etc.)."
    )

    payload = {"contents": [{"parts": [{"text": system_prompt + "\n\nUser Question: " + prompt}]}]}
   
    try:
        response = requests.post(api_url, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        text_response = data['candidates'][0]['content']['parts'][0]['text']
        return jsonify({'response': text_response})
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({'response': 'Error: Failed to communicate with the AI service.'}), 500
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini response: {data}")
        return jsonify({'response': 'Error: Invalid response format from the AI service.'}), 500

# --- Lab System API Endpoints (Simulated) ---
@app.route('/api/lab/orders', methods=['POST'])
def api_lab_create_order():
    if request.headers.get('X-API-Key') != LAB_SHARED_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"orderId": f"LAB-{int(time.time())}", "status": "queued"}), 201

@app.route("/api/lab/orders/<order_id>")
def api_lab_get_order(order_id):
    if request.headers.get('X-API-Key') != LAB_SHARED_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    mock_results = {
        "orderId": order_id, "priority": "routine", "receivedAt": datetime.datetime.now().isoformat(),
        "patient": {"name": "Simulated Patient"},
        "perDepartment": [{"department": "biochemistry", "status": "completed", "results": [{"testCode": "GLU", "value": "110", "unit": "mg/dL", "flag": "H", "referenceRange": {"low": 70, "high": 100}}]}]
    }
    return jsonify(mock_results)

@app.route("/api/lab/status/<order_id>")
def api_lab_check_order_status(order_id):
    try:
        url = f"{LAB_DEFAULT_HOST.rstrip('/')}/api/lab/orders/{order_id}"
        resp = requests.get(url, headers={'X-API-Key': LAB_SHARED_API_KEY}, timeout=15)
        if resp.ok:
            order_data = resp.json()
            per_dept = order_data.get('perDepartment', [])
            completed_depts = [d for d in per_dept if d.get('status') == 'completed']
            return jsonify({
                'orderId': order_id,
                'status': 'completed' if completed_depts else 'in_progress',
                'completedDepartments': completed_depts,
                'allDepartments': per_dept
            })
        return jsonify({'error': f'Failed to fetch order status: {resp.status_code}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Radiology System API Endpoints (Simulated) ---
@app.route('/api/radiology/v1/get_or_request_scan', methods=['POST'])
def api_radiology_get_or_request_scan():
    dummy_dicom_content = b'\x00' * 128 + b'DICM' + b'\x01\x02\x03\x04'
    return dummy_dicom_content, 200, {'Content-Type': 'application/dicom'}

@app.route("/radiology/api/queue_status")
def radiology_queue_status():
    with radiology_queue_lock:
        return jsonify(list(RADIOLOGY_REQUEST_QUEUE))

@app.route("/radiology/dicom/<path:filename>")
def radiology_serve_dicom(filename):
    return send_from_directory(os.path.join(basedir, "downloads"), filename, mimetype="application/octet-stream")

# ----------------- Template Rendering & Main Execution ----------------- #
def render_template_string(template_name, **context):
    class StringLoader(BaseLoader):
        def get_source(self, environment, template):
            # This dictionary now holds all templates for the unified application
            templates = {
                'base.html': base_template_html,
                'login.html': login_html,
                'index.html': index_html,
                'patients_list.html': patients_list_html,
                'patient_form.html': patient_form_html,
                'patient_detail.html': patient_detail_html,
                'patient_edit_form.html': patient_edit_form_html,
                'add_observation.html': add_observation_html,
                'note_form.html': note_form_html,
                'audit_log.html': audit_log_html,
                'manage_users.html': manage_users_html,
                'note_detail.html': note_detail_html,
                # Lab templates
                'lab_request.html': lab_request_html,
                'lab_history.html': lab_history_html,
                'lab_results.html': lab_results_html,
                # Radiology templates
                'radiology_request.html': radiology_request_html,
                'radiology_queue.html': radiology_queue_html,
            }
            if template in templates:
                return templates[template], None, lambda: True
            raise TemplateNotFound(template)

    env = Environment(loader=StringLoader())
    env.filters['nl2br'] = lambda s: s.replace('\n', '<br>') if s else s
    env.globals.update(url_for=url_for, get_flashed_messages=get_flashed_messages,
                           current_user=current_user, request=request, Role=Role,
                           ClinicalNote=ClinicalNote, Observation=Observation, Condition=Condition)
    template = env.get_template(template_name)
    return template.render(**context)

if __name__ == '__main__':
    instance_path = os.path.join(basedir, 'instance')
    if not os.path.exists(instance_path): os.makedirs(instance_path)

    with app.app_context():
        db.create_all()
        if User.query.count() == 0:
            print("Creating default users...")
            admin = User(username='admin', role=Role.ADMIN); admin.set_password('admin123')
            doctor = User(username='dr_house', role=Role.DOCTOR); doctor.set_password('doctor123')
            db.session.add_all([admin, doctor])
            db.session.commit()
            print("Default users created.")

        if not os.path.exists(MODEL_FILENAME):
            train_and_save_model()

    app.run(host='0.0.0.0', debug=True, port=5000)
