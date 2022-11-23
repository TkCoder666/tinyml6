# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
from __future__ import print_function
import numpy as np
import MNN
import cv2
import time
from data_provider import get_test_loader
from tqdm import tqdm


def inference(model_path, image):
    """ inference mobilenet_v1 using a specific picture """
    batch_size = 1
    interpreter = MNN.Interpreter(
        model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    # image = np.random.random((batch_size, 3, 32, 32))
    image = image.astype(np.float32)
    tmp_input = MNN.Tensor((batch_size, 3, 32, 32), MNN.Halide_Type_Float,
                           image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    # constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output = MNN.Tensor((batch_size, 100), MNN.Halide_Type_Float, np.ones(
        [batch_size, 100]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    # print("output belong to class: {}".format(np.argmax(tmp_output.getData())))
    # print(len(tmp_output.getData()))
    # print(tmp_output.getData())
    return np.argmax(tmp_output.getData())


def mnn_inference_benchmark(model_path, val_loader):
    total = 0
    right = 0
    begin = time.time()
    for image, data in tqdm(val_loader):
        # print(type(image),image.shape)
        # print(type(data),data.shape)
        result = inference(model_path, image.numpy())
        # result=0
        total += 1
        if result == data[0]:
            right += 1
    end = time.time()
    print("right", right)
    print("total", total)
    print("cost", end-begin)


if __name__ == "__main__":

    model_path = "/home/tk/code/tinyml6/saved_model/quantized_model/quan_4bit/quant_model_1.mnn"
    print(f"model_path is {model_path}")
    test_loader = get_test_loader(1, 16)
    mnn_inference_benchmark(model_path, test_loader)
