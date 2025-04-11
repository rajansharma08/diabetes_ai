from flask import render_template, request, Blueprint
import pickle
import numpy as np

# Create the blueprint directly in views.py
diabetes_bp = Blueprint('diabetes', __name__, template_folder='templates', static_folder='static')

# Load the diabetes model
try:
    with open('diabetes.pkl', 'rb') as model_file:
        diabetes_model = pickle.load(model_file)
except FileNotFoundError:
    # Try alternative path
    with open('diabetes/diabetes.pkl', 'rb') as model_file:
        diabetes_model = pickle.load(model_file)

@diabetes_bp.route('/')
def index():
    return render_template('diabetes.html')

@diabetes_bp.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@diabetes_bp.route('/predict', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            data = [
                int(request.form['a']),
                int(request.form['b']),
                int(request.form['c']),
                int(request.form['d']),
                int(request.form['e']),
                int(request.form['f']),
                int(request.form['g']),
            ]
            prediction = diabetes_model.predict([data])[0]
            return render_template('afterdiabetes.html', data=prediction)
        except Exception as e:
            return str(e)