import requests
import time

url = "http://localhost:9696/predict"

client = {"job": "management", "duration": 400, "poutcome": "success"}
while True:
	time.sleep(0.01)
	response = requests.post(url, json=client).json()
	print(response)
