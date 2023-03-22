from flask import Flask, render_template, redirect

flask_app = Flask(__name__)

@flask_app.route('/')
def index():
    return render_template('index.html')

@flask_app.route('/house_prices')
def render_dashboard():
    return redirect('/house_prices/')
