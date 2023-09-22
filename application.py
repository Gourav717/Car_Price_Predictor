from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
cors = CORS(app)
model = None
car = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global car
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    global model
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    driven = int(request.form.get('kilo_driven'))
    fuel_type = request.form.get('fuel_type')

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))

    if prediction > 0:
        return str(np.round(prediction[0], 2))
    else:
        return '0'
    


if __name__ == '__main__':
    # Load the trained model
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

    # Load the car data
    car = pd.read_csv('Cleaned_Car_data.csv')

    app.run()
