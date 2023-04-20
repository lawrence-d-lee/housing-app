# Housing App

This is a dash app which predicts house prices for several UK cities. Simply choose the city you are interested in from the dropdown menu, then select
the features you'd like to have. Finally, click on the map to see the model's prediction for that location.

### Run the app using Docker (recommended)
```
1. Check Docker is installed (https://docs.docker.com/engine/install/) and running
2. docker pull ghcr.io/lawrence-d-lee/text_generator:latest
3. docker run -p 8050:8050 ghcr.io/lawrence-d-lee/text_generator:latest
```
### Run the app without Docker (Linux)
```Bash
1. git clone https://github.com/lawrence-d-lee/housing-app.git
2. cd housing-app
3. python3 -m venv housing_venv
4. source housing_venv/bin/activate
5. pip install -r requirements.txt
6. python3 dash_housing_app.py
```
### Run the app without Docker (Windows)
```
1. git clone https://github.com/lawrence-d-lee/housing-app.git
2. cd .\housing-app\
3. python -m venv housing_venv
4. housing_venv\Scripts\activate
5. pip install -r requirements.txt
6. python dash_housing_app.py
```
Then, either click on the location given on the terminal or visit http://localhost:8050/ to view the app.

The app comes with data about UK house prices which has already been obtained. To get current data, you can simply run the etl.py script.
