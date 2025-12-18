import requests

# API URL
url = "http://127.0.0.1:8000/predict"

# Sample input data
sample_data = {
    "amount_count": 10,
    "amount_sum": 1000,
    "amount_mean": 100,
    "amount_std": 20,
    "amount_min": 50,
    "amount_max": 150,
    "amount_median": 100,
    "most_common_CurrencyCode": 1,
    "most_common_ProviderId": 2,
    "most_common_ProductId": 3,
    "most_common_ProductCategory": 1,
    "most_common_ChannelId": 1
}


# Send POST request
response = requests.post(url, json=sample_data)

# Print the response
if response.status_code == 200:
    print("Success!")
    print("Response JSON:", response.json())
else:
    print(f"Error {response.status_code}: {response.text}")
