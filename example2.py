import base64

import grpc
import tensorflow as tf
import requests
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# Get image of a cute cat.
image_path = tf.keras.utils.get_file(
    "image.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
)

# Read the image from disk as raw bytes and then encode it.
bytes_inputs = tf.io.read_file(image_path)
b64str = base64.urlsafe_b64encode(bytes_inputs.numpy()).decode("utf-8")

channel = grpc.insecure_channel("localhost:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "vit"
request.model_spec.signature_name = "serving_default"
request.inputs["string_input"].CopyFrom(tf.make_tensor_proto([b64str]))

grpc_predictions = stub.Predict(request, 10.0)  # 10 secs timeout
print(grpc_predictions)
# model_spec {
#   name: "vit"
#   version {
#     value: 1
#   }
#   signature_name: "serving_default"
# }
# outputs {
#   key: "label"
#   value {
#     dtype: DT_STRING
#     tensor_shape {
#       dim {
#         size: 1
#       }
#     }
#     string_val: "beignets"
#   }
# }
# outputs {
#   key: "confidences"
#   value {
#     dtype: DT_FLOAT
#     tensor_shape {
#       dim {
#         size: 1
#       }
#     }
#     float_val: 0.925145924
#   }
# }

print(
    grpc_predictions.outputs["label"].string_val,
    grpc_predictions.outputs["confidences"].float_val,
)
# [b'beignets'] [0.9251459240913391]
