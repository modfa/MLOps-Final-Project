import requests

car_details = {
    "vehicleType" : "suv", 
    "gearbox" : "manual",
    "model" : "xc_reihe", 
    "fuelType": "diesel",
     "brand" : "volvo",
      "notRepairedDamage" : "no",
    "powerPS": 163,
    "kilometer": 150000,
    "Age": 13.50
}


url = 'http://localhost:9696/predict'
response = requests.post(url, json=car_details)
print(response.json())