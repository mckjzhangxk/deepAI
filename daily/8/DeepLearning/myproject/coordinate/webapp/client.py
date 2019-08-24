import requests
import  json

with open('my.json') as fs:
    data=json.load(fs)
f = requests.post('http://127.0.0.1:9999/web/getCoord?width=100&height=100&offset=11',data=json.dumps(data))
result = f.text
print(result)