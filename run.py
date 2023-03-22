from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from flask_app import flask_app
from dash_housing_app import app as dash_housing_app

application = DispatcherMiddleware(flask_app, {
    '/house_prices': dash_housing_app.server,
})

if __name__ == '__main__':
    run_simple('localhost', 8050, application) 