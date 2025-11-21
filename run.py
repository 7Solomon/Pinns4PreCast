from flask import Flask

from src.blueprints.api import api_bp
from src.blueprints.routes import routes_bp

app = Flask(__name__)

app.register_blueprint(api_bp)
app.register_blueprint(routes_bp)

if __name__ == "__main__":
    app.run(debug=True)