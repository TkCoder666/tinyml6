import onnx
import onnxruntime as ort

if __name__ == '__main__':
    model = onnx.load("saved_model/onnx_model/resnet56.onnx")
    onnx.checker.check_model(model)
    