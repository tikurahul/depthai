from pathlib import Path

import cv2
import numpy as np

import depthai as dai

if __name__ == "__main__":
    model_path = Path(__file__).parent / 'custom_ops/out'
    pipeline = dai.Pipeline()
    # Source
    camera = pipeline.createColorCamera()
    camera.setPreviewSize(300, 300)
    camera.setCamId(0)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setInterleaved(True)
    # Ops
    detection = pipeline.createNeuralNetwork()
    blob_path = model_path / 'model.blob'
    detection.setBlobPath(f'{blob_path.as_posix()}')
    camera.preview.link(detection.input)
    # Link Outputs for Detection
    x_out = pipeline.createXLinkOut()
    x_out.setStreamName('custom')
    detection.out.link(x_out.input)

    device = dai.Device(pipeline)
    device.startPipeline()

    frame_buffer = device.getOutputQueue(name='custom', maxSize=4)

    while True:
        frame = frame_buffer.get()
        # Returns a list
        layer = frame.getFirstLayerFp16()
        layer = np.array(layer, dtype=np.uint8)
        shape = (300, 300, 1)
        frame_data = layer.reshape(shape)
        cv2.imshow('Image', frame_data)
        if cv2.waitKey(1) == ord('q'):
            break
