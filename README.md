### Delivery ETA Prediction (End-to-End ML + Flask Project)(99.6% Accuracy on 40000+ Rows Dataset)

This is a complete end-to-end machine learning project where I built a system to predict restaurant food delivery time (ETA). The idea is similar to how Swiggy, Zomato, Uber Eats, and DoorDash estimate delivery time for each order. The project includes everything from cleaning the data to deploying a working web application using Flask.

## Overview

Food delivery apps depend heavily on accurate delivery time predictions. In this project, I created a machine learning model that predicts delivery time based on multiple real-world factors such as restaurant and customer locations, traffic, weather, order type, rider details, and pickup delays. The final model gives extremely accurate predictions with an R² score of around 0.996.

## Dataset(https://www.kaggle.com/datasets/changlechangsu/india-food-delivery-time-prediction)

The dataset contains about 40,000+ delivery records with features such as:

Delivery person age and rating

Restaurant and delivery latitude/longitude

Order and pickup time

Weather and traffic conditions

Vehicle type and order type

Delivery time (target variable)

The dataset required cleaning because of inconsistent strings, missing values, and formatting issues.

Data Processing & Feature Engineering

To improve model accuracy, several preprocessing and feature engineering steps were applied:

## Time Features:

Extracted order hour, order minute, pickup hour and minute

Calculated pickup delay (minutes between order and pickup time)

Extracted day, month, weekday, and weekend flag

## Location Features:

Calculated Haversine distance between restaurant and customer

Calculated Manhattan distance

Estimated average speed

## Categorical Encoding:
One-hot encoded traffic, weather, order type, vehicle type, and city.
All encoded columns were stored in columns.pkl to keep the same structure during prediction.

Other Features:

Rush hour flag

Night-time flag

Cleaned and standardized text values

Filled missing data with mean, median, or mode as needed

Model Training

I tested several machine learning models, including:

Random Forest
Extra Trees
Neural Network (ANN)
XGBoost Regressor
XGBoost performed the best, giving:

# R² Score: 0.996
# MAE: 0.24 minutes
# RMSE: 0.46 minutes

Which means the ETA prediction is usually off by less than a minute.

## The final model and required preprocessing objects were saved as:

Model.json (XGBoost model)

scaler.pkl (StandardScaler)

columns.pkl (list of final input feature columns)

Web Application (Flask)

A Flask-based web app was created to allow users to input all required delivery details. The backend recreates the same preprocessing pipeline used during training, including:

Time parsing

Feature engineering

Distance calculation

One-hot encoding

Column alignment

Scaling

After processing the input, the app returns the predicted ETA in minutes. The frontend is a simple HTML form designed for easy data input and quick testing.

Project Structure

project/

app.py

templates/index.html

Model.json

scaler.pkl

columns.pkl

requirements.txt

README.md

How to Run

## Install dependencies:
pip install -r requirements.txt

Run the app:
python app.py

Open in browser:
http://localhost:8000/

Summary

This project demonstrates the full machine learning workflow:

Data cleaning and preprocessing

Feature engineering

Model training and tuning

Deployment using Flask

Real-time prediction

The aim was to create a practical, production-like system that works end-to-end and reflects how real delivery ETA systems are built.