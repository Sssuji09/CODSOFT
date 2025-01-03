from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load saved models
titanic_model = joblib.load('titanic_model.pkl')
titanic_scaler = joblib.load('titanic_scaler.pkl')
sales_model = joblib.load('polynomial_regression_model.pkl')
sales_poly = joblib.load('polynomial_features_model.pkl')
iris_model = joblib.load('iris_svc_model.pkl')
iris_scaler = joblib.load('iris_scaler.pkl')
iris_species_mapping = joblib.load('iris_species_mapping.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/titanic_form')
def titanic_form():
    """Render the Titanic form."""
    return render_template('titanic_form.html')

@app.route('/form_page', methods=['GET', 'POST'])
def form_page():
    """Handles Titanic survival prediction form."""
    if request.method == 'POST':
        # Get form data
        features = [
            float(request.form['Fare']),
            int(request.form['Sex']),  # 0 = female, 1 = male
            float(request.form['Age']),
            int(request.form['Pclass']),
            int(request.form['FamilySize']),
            int(request.form['IsAlone'])
        ]
        
        # Prepare input data
        feature_names = ['Fare', 'Sex', 'Age', 'Pclass', 'FamilySize', 'IsAlone']
        input_data = pd.DataFrame([features], columns=feature_names)
        
        # Scale numerical features (Fare and Age)
        input_data[['Fare', 'Age']] = titanic_scaler.transform(input_data[['Fare', 'Age']])
        
        # Make prediction
        prediction = titanic_model.predict(input_data)
        survival = "Passenger survived" if prediction[0] == 1 else "Passenger did not survive"

        # Return result
        return render_template('form_page.html', prediction_text=f"{survival}")
    else:
        # GET request - render the form
        return render_template('form_page.html')

@app.route('/iris_form')
def iris_form():
    """Render the Iris form."""
    return render_template('iris_form.html')

@app.route('/form_page1', methods=['GET', 'POST'])
def form_page1():
    """Handle iris flower species prediction."""
    if request.method == 'POST':
        try:
            # Get and validate form data
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
            
            # Prepare and scale input data
            input_data = np.array([features])
            scaled_features = iris_scaler.transform(input_data)
            
            # Make prediction
            prediction = iris_model.predict(scaled_features)[0]
            
            # Convert numerical prediction back to species name
            species_names = {v: k for k, v in iris_species_mapping.items()}
            predicted_species = species_names[prediction]
            
            return render_template('form_page1.html', 
                prediction_text=f"Predicted Species: {predicted_species.capitalize()}")
            
        except ValueError:
            return render_template('form_page1.html', 
                prediction_text="Error: Please enter valid numerical values")
        except Exception as e:
            return render_template('form_page1.html', 
                prediction_text=f"Error: {str(e)}")
    
    return render_template('form_page1.html')

@app.route('/sales_form')
def sales_form():
    """Render the Sales form."""
    return render_template('sales_form.html')

@app.route('/form_page2', methods=['GET', 'POST'])
def form_page2():
    if request.method == 'POST':
        # Get form data
        tv = float(request.form['tv'])
        radio = float(request.form['radio'])
        newspaper = float(request.form['newspaper'])

        # Prepare input for prediction
        input_data = np.array([[tv, radio, newspaper]])
        input_data_poly = sales_poly.transform(input_data)
        prediction = sales_model.predict(input_data_poly)[0]

        return render_template('form_page2.html', prediction=prediction)

    return render_template('form_page2.html')

if __name__ == '__main__':
    app.run(debug=True)
