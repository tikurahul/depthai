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
    camera.setInterleaved(False)
    # Output
    x_out = pipeline.createXLinkOut()
    x_out.setStreamName('rgb')
    # Link
    camera.preview.link(x_out.input)

    device = dai.Device(pipeline)
    device.startPipeline()

    frame_buffer = device.getOutputQueue(name='rgb', maxSize=4)

    while True:
        frame = frame_buffer.get()
        shape = (3, frame.getHeight(), frame.getWidth())
        # BGR -> RGB
        frame_data = frame.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        cv2.imshow('Image', frame_data)
        if cv2.waitKey(1) == ord('q'):
            break
