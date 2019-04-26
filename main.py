
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from flask import request
from datetime import datetime
import sys
import os

import pickle
# Preparing the Classifier
cur_dir = os.path.dirname('__file__')
regressor = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model.pkl'), 'rb'))
model_columns = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columns.pkl'),'rb')) 

# Your API definition
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    #print(request)
    if regressor:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            
            prediction = list(regressor.predict(query).astype("int64"))
            #print(prediction)
            
                
            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    app.run()