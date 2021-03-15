from flask import Flask, request, jsonify, render_template
import os
import pickle
import numpy as np

img_folder = os.path.join('static', 'images')
app = Flask(__name__)
model = pickle.load(open('D:\\py\\titanic.sav', 'rb'))
app.config['img_folder'] = img_folder

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if int(prediction) == 0:
        output = 'not survived'
        full_img_path = os.path.join(app.config['img_folder'], 'sad.jpg')
    else:
        output = 'survived'
        full_img_path = os.path.join(app.config['img_folder'], 'happy.png')
    return render_template('output.html', prediction_text='the passenger has ' + output, emoji=full_img_path)


if __name__ == "__main__":
    app.run(debug=True)