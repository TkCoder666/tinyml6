import onnxruntime
from onnxruntime.datasets import get_example
import torch

pth_path = "saved_model/original_model/resnet56_model.pth"
onnx_path = "/sldata/liujunhan/tinyml6/saved_model/onnx_model/resnet56.onnx"

model = torch.load(pth_path)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 测试数据
dummy_input = torch.randn(1, 3, 32, 32, device='cpu')

example_model = get_example(onnx_path)
# netron.start(example_model) 使用 netron python 包可视化网络
sess = onnxruntime.InferenceSession(example_model)
print(sess.get_inputs()[0].name)
onnx_out = sess.run(None, {"input": to_numpy(dummy_input)})
print(onnx_out)

model.eval()
with torch.no_grad():
    # pytorch model 网络输出
    torch_out = model(dummy_input)
    print(torch_out)