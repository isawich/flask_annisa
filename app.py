# app.py  —  file utama aplikasi Flask

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model dan scaler saat aplikasi pertama kali dijalankan
with open('model.pkl', 'rb') as f:
    models = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_names = list(models.keys())  # ['Decision Tree', 'SVC']


@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)


@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Pregnancies'            : int(request.form['Pregnancies']),
        'Glucose'                : int(request.form['Glucose']),
        'BloodPressure'          : int(request.form['BloodPressure']),
        'SkinThickness'          : int(request.form['SkinThickness']),
        'Insulin'                : int(request.form['Insulin']),
        'BMI'                    : float(request.form['BMI']),
        'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
        'Age'                    : int(request.form['Age']),
    }

    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)

    selected_model = request.form.get('model', model_names[0])
    clf = models[selected_model]
    y_pred = clf.predict(df_scaled)
    prediction = 'Diabetic' if int(y_pred[0]) == 1 else 'Non-Diabetic'

    return render_template('index.html',
                           model_names=model_names,
                           prediction=prediction,
                           selected_model=selected_model)


if __name__ == '__main__':
    app.run(debug=True)
