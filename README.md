# Housing-App

This is a dash app which predicts house prices for several UK cities. Simply choose the city you are interested in from the dropdown menu, then select
the features you'd like to have. Finally, click on the map to see the model's prediction for that location.

To run the app, clone the repo to your desktop. To run the app using Docker (recommended)

### Commands
```
1. cd Housing-App
2. docker build . -t dash_app
3. docker run -p 8050:8050 dash_app
```
Then, either click on the location given on the terminal or visit http://localhost:8050/ to view the app.

If you don't have Docker on your computer, you can run

### Commands
```
1. cd dash-deployment
2. python dash_housing_app
```
