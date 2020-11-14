from flask import Flask

app = Flask(__name__)
app.config['SERVER_NAME'] = "127.0.0.1:5000"

from flaskApp import routes