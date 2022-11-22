import torch
import torch.nn as nn
import torch.onnx

pt_model_path = 'original_model/resnet56_model.pth'
onnx_model_path = 'onnx_model/resnet56.onnx'
model = torch.load(pt_model_path)
model.eval()
input_tensor = torch.randn(1, 3, 32, 32)  # !shape of input
input_names = ['input']
output_names = ['output']
torch.onnx.export(model, input_tensor, onnx_model_path, verbose=True,
                  input_names=input_names, output_names=output_names)
