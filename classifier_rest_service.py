from flask import Flask, request
import pickle

import numpy as np
local_classifier = pickle.load(open('classifier.pickle','rb'))
local_scalar = pickle.load(open('sc.pickle','rb'))


app = Flask(__name__)

@app.route("/model",methods=['POST'])
def hello_world():
    request_data = request.get_json(force=True)
    age = request_data['age']
    salary = request_data['salary']
    print(age)
    print(salary)
    prediction = local_classifier.predict(local_scaler.transform(np.array([[age,salary]])))
    
    return "The prediction is {}".format(prediction)

if __name__ == "__main__":
    app.run(port=8002, debug=True)