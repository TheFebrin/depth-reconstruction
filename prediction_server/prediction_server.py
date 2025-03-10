"""
Python gRPC prediction server
https://grpc.io/docs/languages/python/quickstart/

To generate proto use:
cd prediction_server
$ python -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/prediction_server.proto
"""

import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)
sys.path.append(parentdir)

import prediction_server_pb2
import prediction_server_pb2_grpc
import grpc
import torch
import argparse

import numpy as np
import albumentations as A

from concurrent import futures

from models.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for prediction server.'
    )
    parser.add_argument(
        '--device', required=True, type=str, help='Which device.', choices=['cpu', 'cuda']
    )
    return parser.parse_args()


class PredictionServer(prediction_server_pb2_grpc.PredictionServerServicer):

    def __init__(
        self,
        model_path: str = '../models/saved_models/best_synthetic_model.pth',
        device: str = 'cpu',
    ):
        print('Start server init ...')
        self._model_path = model_path
        self._device = device
        print('Loading model')
        # Load model
        self._model = UNet(n_channels=3, n_classes=3, bilinear=False)
        self._model.load_state_dict(torch.load(
            f=self._model_path,
            map_location=torch.device(self._device))
        )

        print('Model loaded.')
        print('Server init finished. Waiting for request...')

    def ping(self, request, context):
        print(context, dir(context))
        return prediction_server_pb2.PingResponse(message=f'Pong. Message: {request.message}')

    def predict(self, request, context):
        print(f'Received request image: {request.height} x {request.width} x {request.channels}')

        # Define transforms on the image
        transform = A.Compose([
            A.Resize(width=1280, height=720)
        ])

        # Read and transform the image
        color_img = np.frombuffer(request.image, dtype=np.uint8).reshape(request.height, request.width, request.channels)
        transformed = transform(image=color_img)
        color_img = transformed["image"]
        color_img = torch.from_numpy(color_img.astype(np.float32)).permute(2, 0, 1)

        print('Image read and transformed.')

        with torch.no_grad():
            pred_img = self._model(color_img.view(-1, *color_img.shape))[0].detach().permute(1, 2, 0).numpy()

        print('Prediction ready!')

        return prediction_server_pb2.PredictionResponse(
            image=pred_img.tobytes(),
            dtype=str(pred_img.dtype),
        )


def serve():
    print('Starting prediction server.')
    args = parse_args()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_server_pb2_grpc.add_PredictionServerServicer_to_server(
        PredictionServer(device=args.device), server
    )
    server.add_insecure_port('[::]:50042')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
