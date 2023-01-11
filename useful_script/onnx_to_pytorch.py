import argparse
import onnx2pytorch
import onnx
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model", type=str,
                        default="saved_model/onnx_model/resnet56.onnx")
    parser.add_argument("--pytorch_model", type=str,
                        default="saved_model/pytorch_model/resnet56.pth")
    args = parser.parse_args()
    onnx_model = onnx.load(args.onnx_model)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
    torch.save
