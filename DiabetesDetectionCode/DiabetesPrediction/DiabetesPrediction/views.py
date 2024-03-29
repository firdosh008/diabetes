# import libraries
import pandas as pd
import numpy as np
# for heatmap and stuff
import seaborn as sns
from django.shortcuts import render

from sklearn.model_selection import train_test_split as tts
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def home(request):
    return render(request, "Home.html")


def predict(request):
    return render(request, "predict.html")


def result(request):
    # importing the dataset
    data = pd.read_csv(r"C:/Users/adhik/Desktop/Diabetes Detection/archive/diabetes.csv")

    # Train Test Split
    # Dependent variable
    x = data.drop("Outcome", axis=1)
    # Independent variable
    y = data['Outcome']
    X_Train, X_Test, Y_Train, Y_Test = tts(x, y, test_size=0.20)

    # Preprocessing Step
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(X_Train)
    x_test_scaled = sc.transform(X_Test)

    # MLP Classifier: Neural Network Model with Scaled Data
    mlps = MLPClassifier(random_state=42, max_iter=500)
    mlps.fit(x_train_scaled, Y_Train)

    # Fetching Data From User's Input from form
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    # Scaling the user input
    user_input = sc.transform([[val1, val2, val3, val4, val5, val6, val7, val8]])

    # Prediction
    pred = mlps.predict(user_input)
    res = ""
    if pred == [1]:
        res = "Positive result, You need to see a doctor.."
    else:
        res = "Negative result, You are healthy.. :)"

    return render(request, "predict.html", {"result2": res})
