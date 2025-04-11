from flask import Flask

# Create a Flask app
app = Flask(__name__)

# Import the blueprint directly (no relative import)
from views import diabetes_bp  # Direct import from views.py file

app.register_blueprint(diabetes_bp)

if __name__ == '__main__':
    app.run(debug=True)