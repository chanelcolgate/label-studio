import base64
import json

import tensorflow as tf
import requests

# Get image of a cute cat.
image_path = tf.keras.utils.get_file(
    "image.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
)

# Read the image from disk as raw bytes and then encode it.
bytes_inputs = tf.io.read_file(image_path)
b64str = base64.urlsafe_b64encode(bytes_inputs.numpy()).decode("utf-8")

# Create the request payload
data = json.dumps({"signature_name": "serving_default", "instances": [b64str]})
headers = {"content-type": "application/json"}
json_response = requests.post(
    "http://localhost:8502/v1/models/vit:predict", data=data, headers=headers
)
print(json.loads(json_response.text))
# {'predictions': [{'label': 'beignets', 'confidences': 0.925145924}]}
