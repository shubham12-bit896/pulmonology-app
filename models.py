# models.py
# This file contains all the SQLAlchemy database models for the application.

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

# Initialize the database instance
db = SQLAlchemy()

# ----------------- User Roles & Permissions ----------------- #

class Role:
    ADMIN = 'Admin'
    DOCTOR = 'Doctor'
    HEALTH_WORKER = 'Health Worker'
    IT_EXECUTIVE = 'IT Executive'

# ----------------- Database Models ----------------- #

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(50), nullable=False, default=Role.HEALTH_WORKER)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uhid = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    contact = db.Column(db.String(20), nullable=True)
    admission_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    smoking_status = db.Column(db.String(20), nullable=False, default='Never Smoked')
    pack_years = db.Column(db.Integer, nullable=False, default=0)
    history_of_asthma = db.Column(db.String(5), nullable=False, default='No')

    symptoms = db.Column(db.Text, nullable=False)
    medical_history = db.Column(db.Text, nullable=True)
    medications = db.Column(db.Text, nullable=True)
    preliminary_diagnosis = db.Column(db.Text, nullable=True)
    final_diagnosis = db.Column(db.Text, nullable=True)
    treatment_plan = db.Column(db.Text, nullable=True)
    next_follow_up = db.Column(db.Date, nullable=True)

    history = db.relationship('PatientHistory', backref='patient', lazy=True, cascade="all, delete-orphan")
    pfts = db.relationship('PFT', backref='patient', lazy=True, cascade="all, delete-orphan")
    reminders = db.relationship('Reminder', backref='patient', lazy=True, cascade="all, delete-orphan")

class PFT(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id_fk = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    test_date = db.Column(db.Date, nullable=False, default=datetime.date.today)
    fvc = db.Column(db.Float, nullable=False)
    fev1 = db.Column(db.Float, nullable=False)
    fev1_fvc_ratio = db.Column(db.Float, nullable=False)

class PatientHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    uhid_str = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    final_diagnosis = db.Column(db.Text, nullable=True)
    treatment_plan = db.Column(db.Text, nullable=True)
    edit_timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    edited_by = db.Column(db.String(100), default="System")

class Reminder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id_fk = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    reminder_type = db.Column(db.String(50), nullable=False)
    reminder_date = db.Column(db.DateTime, nullable=False)
    notes = db.Column(db.Text, nullable=True)
    completed = db.Column(db.Boolean, default=False)

class ApiKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    department = db.Column(db.String(100), nullable=False)