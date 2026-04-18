import requests

url = "http://127.0.0.1:1234/invocations"

data = {
    "inputs": [[0.5, 0.3, 0.2, 0.1]]
}

response = requests.post(url, json=data)
print(response.json())
