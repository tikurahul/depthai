from pathlib import Path

import cv2
import numpy as np
import torch

from grayscale.model import GrayscaleModel


def export():
    output_dir = Path(__file__).parent / 'out'
    output_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(output_dir=output_dir)
    print('Done.')


def export_onnx(output_dir):
    """
    Exports the model to an ONNX file.
    """
    # Channels Last
    shape = (1, 300, 300, 3)
    model = GrayscaleModel(shape=shape, dtype=torch.float)
    X = torch.ones(shape, dtype=torch.float)
    torch.onnx.export(
        model,
        X,
        f'{output_dir.as_posix()}/model.onnx',
        opset_version=9,
        do_constant_folding=True
    )


def preview():
    capture = cv2.VideoCapture(0)
    model = None
    while True:
        _, color = capture.read()
        if not model:
            shape = (1, ) + color.shape
            model = GrayscaleModel(shape=shape, dtype=torch.uint8)

        # Numpy -> torch.Tensor
        color = torch.from_numpy(color)
        color = torch.unsqueeze(color, dim=0)
        out = model(color)[0].numpy()
        # Channels Last
        out = out.astype(np.uint8)
        cv2.imshow('Gray', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    export()
