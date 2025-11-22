from flask import Flask

from src.blueprints.api import api_bp
from src.blueprints.routes import routes_bp
from src.blueprints.info import info_bp

app = Flask(__name__)

app.register_blueprint(api_bp)
app.register_blueprint(routes_bp)
app.register_blueprint(info_bp)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)