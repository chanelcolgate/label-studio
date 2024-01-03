import logging
import base64
import json

import requests
import tensorflow as tf
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (
    get_image_local_path,
    get_single_tag_keys,
)

logger = logging.getLogger(__name__)


class Vit(LabelStudioMLBase):
    def __init__(self, trainable=False, batch_size=32, epochs=3, **kwargs):
        super().__init__(**kwargs)
        (
            self.from_name,
            self.to_name,
            self.value,
            self.labels_in_config,
        ) = get_single_tag_keys(self.parsed_label_config, "Choices", "Image")

    def predict(self, tasks, **kwargs):
        # image_path = get_image_local_path(tasks[0]["data"][self.value])
        image_path = tasks[0]["data"][self.value]
        # [{'id': 1, 'data': {'image': '/data/upload/1/1fc6bf4d-image.jpg'}, 'meta': {}, 'created_at': '2024-01-02T16:56:53.740560Z', 'updated_at': '2024-01-02T16:56:53.740594Z', 'is_labeled': False, 'overlap': 1, 'inner_id': 1, 'total_annotations': 0, 'cancelled_annotations': 0, 'total_predictions': 0, 'comment_count': 0, 'unresolved_comment_count': 0, 'last_comment_updated_at': None, 'project': 1, 'updated_by': None, 'file_upload': 1, 'comment_authors': [], 'annotations': [], 'predictions': []}]

        # Read the image from disk as raw bytes and then encode it.
        bytes_inputs = tf.io.read_file(image_path)
        b64str = base64.urlsafe_b64encode(bytes_inputs.numpy()).decode("utf-8")

        # Create the request payload
        data = json.dumps({"signature_name": "serving_default", "instances": [b64str]})
        headers = {"content-type": "application/json"}
        json_response = requests.post(
            "http://172.26.0.2:8501/v1/models/vit:predict", data=data, headers=headers
        )
        print(json_response)
        result = json.loads(json_response.text)
        predicted_label = result["predictions"][0]["label"]
        predicted_label_score = result["predictions"][0]["confidences"]
        return [
            {
                "result": [
                    {
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "choices",
                        "value": {"choices": [predicted_label]},
                    }
                ],
                "score": float(predicted_label_score),
            }
        ]
