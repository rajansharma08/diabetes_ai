# from flask import Blueprint

# diabetes_bp = Blueprint('diabetes', __name__, template_folder='templates', static_folder='static')

# from . import views

from flask import Blueprint

# Create the blueprint
diabetes_bp = Blueprint('diabetes', __name__, template_folder='templates', static_folder='static')

# Import views at the bottom to avoid circular imports
from . import views