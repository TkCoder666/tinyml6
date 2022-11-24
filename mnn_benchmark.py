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
from glob import glob


def benchmark(model_path, loops):
    """ inference mobilenet_v1 using a specific picture """
    batch_size = 1
    interpreter = MNN.Interpreter(
        model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    image = np.random.random((batch_size, 3, 32, 32))
    image = image.astype(np.float32)
    tmp_input = MNN.Tensor((batch_size, 3, 32, 32), MNN.Halide_Type_Float,
                           image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    dt_list = []
    for i in range(0, loops):
        t1 = time.time()
        interpreter.runSession(session)
        t2 = time.time()
        dt_list.append(t2-t1)
    print(f"{model_path}'s time benchmark")
    print("max,avg,min")
    print(f"{max(dt_list)*1000:.3f},{sum(dt_list)/len(dt_list)*1000:.3f},{min(dt_list)*1000:.3f}")


def get_all_mnn_model_path():
    dir_list = ["/home/tk/code/tinyml6/saved_model/pruned_model/simdoc", "/home/tk/code/tinyml6/saved_model/pruned_model/sniplevel", "/home/tk/code/tinyml6/saved_model/pruned_model/taylorfochannel",
                "/home/tk/code/tinyml6/saved_model/quantized_model/quan_4bit", "/home/tk/code/tinyml6/saved_model/quantized_model/quan_8bit"]
    mnn_path_list = []
    for dir in dir_list:
        pass


if __name__ == "__main__":

    model_path = "/home/tk/code/tinyml6/saved_model/mnn_model/resnet56.mnn"
    benchmark(model_path, 1000)
