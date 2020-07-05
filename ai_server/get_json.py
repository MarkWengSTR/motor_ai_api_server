import json

with open("RESPONSE.json") as f:
  data = json.load(f)
print(data)