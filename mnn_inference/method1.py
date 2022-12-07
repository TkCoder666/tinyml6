# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
from __future__ import print_function
import numpy as np
import MNN
import cv2


def inference():
    """ inference mobilenet_v1 using a specific picture """
    batch_size=1
    interpreter = MNN.Interpreter(
        "../saved_model/mnn_model/resnet56.mnn")
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    image = np.random.random((batch_size, 3, 32, 32))
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
    print("output belong to class: {}".format(np.argmax(tmp_output.getData())))
    # print(len(tmp_output.getData()))
    print(tmp_output.getData())


if __name__ == "__main__":
    inference()
