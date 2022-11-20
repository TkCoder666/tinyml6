import MNN.expr as F
import cv2
import torch
import numpy as np
mnn_model_path = './yolo.mnn'
image_path = './input.jpeg'

# 图像预处理
def process(image, size):
    # image_data = cv2.imread(image_path)
    # image_data = cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB)
    image_resize = cv2.resize(image, size).astype(float)
    image_resize /= 255
    input_data = np.array(image_resize)
    input_data = input_data.transpose((2, 0, 1))  # HWC --> CHW
    input_data = np.expand_dims(input_data, 0)
    return input_data

# 推理阶段
def m_yolo_inference(image, m_path):
# m_path 即mnn模型的路径
    vars = F.load_as_dict(m_path)
    inputVar = vars["input"]
    # 这里注意，由于直接MNN模型的数据输入，输入格式默认是caffe格式，数据类型是NC4HW，而我们使用的格式（pytorch）是NCHW，因此需要将格式转换一下。
    if inputVar.data_format == F.NC4HW4:
        inputVar.reorder(F.NCHW)
    inputVar.write(image.tolist())
    # 查看输出结果， 由于MNN很多op操作，会造成多个输出头，所以我们需要借助netron工具查看以下各个输出头的shape，然后return我们需要的输出头即可。
    outputVar0 = vars['output']
    if outputVar0.data_format == F.NC4HW4:
        outputVar0 = F.convert(outputVar0, F.NCHW)

    outputVar1 = vars['187']
    if outputVar1.data_format == F.NC4HW4:
        outputVar1 = F.convert(outputVar1, F.NCHW)

    return torch.from_numpy(outputVar0.read()), torch.from_numpy(outputVar1.read())

def main():
    image = process(image_path, (320, 320))
    out1, out2 = m_yolo_inference(image, mnn_model_path)
    print(type(out1))

if __name__ == "__main__":
    main()