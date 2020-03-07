
from flask import Flask, request, jsonify
import ts_model
import json
import os 
import shutil

from authentication.auth import requires_auth
from syntbots_ai_exception import SyntBotsAIException
import logging


app = Flask(__name__)

app_root=os.path.dirname(os.path.abspath(__file__))+"/"

@app.errorhandler(SyntBotsAIException)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/ai/timeseriesstatus', methods = ['GET'])
#@requires_auth
def getHealthstatus():
    return "timeseries is executing"

	
	
@app.route('/ai/get_files', methods = ['POST'])
#@requires_auth
def get_templates():
    response={}
    status=1
    template=[]
    for file in os.listdir(app_root+"anomaly"):
        template.append(file.split('.')[0])
    response['ai_response'] = template
    response['ai_status'] = status
    response=json.dumps(response);
    return response
	
@app.route('/ai/get_filepath', methods = ['POST'])
#@requires_auth
def get_filepath():
    response={}
    req=request.json
    status=1
    try:
        filename = req['filename']
        filepath = app_root+"anomaly/"+filename+'.json'
    except Exception as ae:
           logging.exception(ae)
           status=500
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           raise SyntBotsAIException(err_msg, status_code=status)
    response['ai_response'] = filepath
    response['ai_status'] = status
    response=json.dumps(response);
    return response
 
@app.route('/ai/ts_summarize', methods = ['POST'])
#@requires_auth
def summarize():
    req = request.json; print(req)
    response = {}
    status = 1
    summarizeResp = {}
    try:
        uploaded_file = req['inputFile']
        datefield = req['datefield']
        tgtfield = req['targetfield']
        frequency = req['forecast_frequency']
        summarizeResp = ts_model.summarize(uploaded_file,datefield,tgtfield,frequency)
    except Exception as ae:
           logging.exception(ae)
           status=500
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           raise SyntBotsAIException(err_msg, status_code=status)
    response['ai_response'] = summarizeResp
    response['ai_status'] = status
    response=json.dumps(response); print(response)
    return response    

	
@app.route('/ai/ts_train', methods = ['POST'])
#@requires_auth
def train():
    req = request.json; print(req)
    response = {}
    status = 1
    compareResp = {}
    try:
        uploaded_file = req['inputFile']
        holiday_file = req['holidayFile']
        if holiday_file=='': holiday_file=None
        datefield = req['datefield']
        tgtfield = req['targetfield']
        modelKey = req['modelKey']
        modelName = req['modelName']
        regressors = req['other_regressor']
        if regressors=='': regressors=None
        frequency = req['forecast_frequency']
        #if holiday_file is not None: holiday_file = app_root+'input/'+holiday_file
        compareResp = ts_model.ts_train(modelKey,modelName,uploaded_file,holiday_file,datefield,tgtfield,regressors,frequency)
        print(compareResp)
    except Exception as ae:
           logging.exception(ae)
           status=500
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           raise SyntBotsAIException(err_msg, status_code=status)
    response['ai_response'] = compareResp
    response['ai_status'] = status
    response=json.dumps(response); print(response)
    return response    

	
@app.route('/ai/ts_forecast', methods = ['POST'])
#@requires_auth
def predict():
    req = request.json; print(req)
    response = {}
    status = 1
    compareResp = {}
    try:
        model_name = req['modelName']
        uploaded_file = req['inputFile']
        holiday_file = req['holidayFile']
        if holiday_file=='': holiday_file=None
        datefield = req['datefield']
        regressors = req['other_regressor']
        if regressors=='': regressors=None
        #if holiday_file is not None: holiday_file = app_root+'input/'+holiday_file
        compareResp = ts_model.ts_predict(model_name,uploaded_file,holiday_file,datefield,regressors)
        print(compareResp)
    except Exception as ae:
           logging.exception(ae)
           status=500
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           raise SyntBotsAIException(err_msg, status_code=status)
    response['ai_response'] = compareResp
    response['ai_status'] = status
    response=json.dumps(response); print(response)
    return response   

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5002,debug=True)
