from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [request.form.get('Species'), float(request.form.get('Length1')), float(request.form.get('Length2')), 
    float(request.form.get('Length3')), 
    float(request.form.get('Height')), float(request.form.get('Width'))] 
    input_df = pd.DataFrame([input_features], columns=['Species', 'Length1', 'Length2', 'Length3' , 'Height', 'Width'])  
    prediction = model.predict(input_df)[0]
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
