import requests

url = "http://0.0.0.0:9696/predict"
student = {
    "gender": "female",
    "race/ethnicity":"group B",
    "parental level of education":"master's degree",
    "lunch":"standard",
    "test preparation course":"none"
}
response = requests.post(url, json=student)
predictions = response.json()

print(predictions)