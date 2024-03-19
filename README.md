## Fish Weight Predictor Flask App


This repository contains the code for a Flask-based web application designed to predict the weight of fish using machine learning. The application utilizes a RandomForestRegressor model trained on various physical measurements of fish, including species, length, height, and width.

## Live Demo
A live demo of the app is available on Heroku: https://fish-weight-predictor-app-03fe977d484b.herokuapp.com/

## Features
Predict the weight of a fish based on its species and measurements.
User-friendly web interface for easy interaction with the prediction model.
Implementation of RandomForestRegressor for robust and accurate predictions.

## Local Setup
To run this application locally, follow these steps:

## Clone the Repository
git clone https://github.com/yourusername/fish-weight-predictor.git
cd fish-weight-predictor

## Create and Activate a Virtual Environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

## Install Dependencies
pip install -r requirements.txt

## Start the Flask App
python app.py  
Visit http://127.0.0.1:5001/ in your web browser to use the application locally.

## Deployment
This app is deployed on Heroku. You can view the live application here: https://fish-weight-predictor-app-03fe977d484b.herokuapp.com/

## Model Information
The machine learning model was trained using a dataset that contains measurements (e.g., length, height, width) for various fish species. RandomForestRegressor from scikit-learn was selected due to its effectiveness in handling both linear and non-linear relationships.

## Technologies
Python,
Flask, 
pandas, 
scikit-learn, 
NumPy, 
Heroku (for deployment)

