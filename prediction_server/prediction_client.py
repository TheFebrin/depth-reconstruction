import grpc

import prediction_server_pb2
import prediction_server_pb2_grpc

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ping():
    with grpc.insecure_channel('localhost:50042') as channel:
        stub = prediction_server_pb2_grpc.PredictionServerStub(channel)
        response = stub.ping(prediction_server_pb2.PingRequest(message='test ping'))
    print(f"[Prediction client] [ping] Message from server: {response.message}")


def predict(img: np.ndarray) -> np.ndarray:
    assert img.dtype == np.uint8, f'Wrong dtype: {img.dtype}'

    with grpc.insecure_channel(
            'localhost:50042',
            options=[('grpc.max_send_message_length', 50000000), ('grpc.max_receive_message_length', 50000000)]
    ) as channel:
        stub = prediction_server_pb2_grpc.PredictionServerStub(channel)
        response = stub.predict(prediction_server_pb2.PredictionRequest(
            image=img.tobytes(),
            height=img.shape[0],
            width=img.shape[1],
            channels=img.shape[2],
        ))

    prediction = np.frombuffer(response.image, dtype=response.dtype).reshape(*img.shape)

    print(f"[Prediction client] [predict] got message from server")
    return prediction


def main():
    img = np.array(Image.open('../images/color_production_1.png'))
    prediction = predict(img=img)
    plt.figure(figsize=(15, 15))
    plt.imshow(prediction)
    plt.show()


if __name__ == '__main__':
    main()
