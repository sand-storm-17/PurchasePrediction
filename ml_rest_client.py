import json
import requests

url = 'http://http://127.0.0.1:8002/model'

request_data = json.dumps({'age':40,'salary':20000})
response = requests.post(url,request_data)
print(response.text)
