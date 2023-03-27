# Housing-App

This is a dash app which predicts house prices for several UK cities. Simply choose the city you are interested in from the dropdown menu, then select
the features you'd like to have. Finally, click on the map to see the model's prediction for that location.

To run the app, clone the repo to your desktop.

### Run the app using Docker (recommended)
```
1. cd Housing-App
2. docker build . -t dash_app
3. docker run -p 8050:8050 dash_app
```
### Run the app without Docker
```
1. cd dash-deployment
2. python -m venv venv
3. venv\Scripts\activate
4. pip install -r requirements.txt
5. python dash_housing_app
```
Then, either click on the location given on the terminal or visit http://localhost:8050/ to view the app.
