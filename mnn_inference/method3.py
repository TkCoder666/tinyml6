import MNN
import cv2
import numpy as np
import time

def process(image_data, size):
    image_resize = cv2.resize(image_data, size).astype(float)
    input_data = np.array(image_resize)
    input_data = input_data.transpose((2, 0, 1))  # HWC --> CHW
    input_data = np.expand_dims(input_data, 0)
    return input_data


def inference(model_path, input_format, input_data):

    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    # 获取输入维度
    print(input_tensor.getShape())

    # NCHW : MNN.Tensor_DimensionType_Caffe
    # NC4HW : MNN.Tensor_DimensionType_Caffe_C4
    # NHWC : MNN.Tensor_DimensionType_Tensorflow
    tmp_input = MNN.Tensor(input_format, MNN.Halide_Type_Float, input_data, MNN.Tensor_DimensionType_Caffe)
    start_time = time.time()
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    infer_result = interpreter.getSessionOutput(session)
    end_time = time.time()
    print("infer time %s ms " % str((end_time - start_time) * 1000))
    # 获取输出维度
    print(infer_result.getShape())
    output_data = infer_result.getData()
    return output_data


if __name__ == "__main__":
    model_path = "../saved_model/mnn_model/resnet56.mnn"
    image_path = 'test.JPEG'
    resize = (32, 32)
    input_format = (1, 3, 32, 32)
    image_data = cv2.imread(image_path)
    input_data = process(image_data, resize)
    print("input_data", input_data.shape)
    start_time = time.time()
    output_data = inference(model_path, input_format, input_data)
    end_time = time.time()
    print("handle time %s ms " % str((end_time - start_time) * 1000))
    print(output_data)
    print("Done!")
