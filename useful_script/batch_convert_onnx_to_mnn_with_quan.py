import subprocess
import os
from tqdm import tqdm
if __name__ == '__main__':
    n_epoch = 6
    bits = 8
    root_dir = f"/home/tk/code/tinyml6/saved_model/quantized_model/LSQquantize_data_{bits}bit"
    dst_dir = f"/home/tk/code/tinyml6/saved_model/quantized_model/quan_{bits}bit"
    python_bin = "/home/tk/miniconda3/envs/ml_stable/bin/python"
    for epoch in tqdm(range(1, n_epoch + 1)):
        command = f"{python_bin} -m MNN.tools.mnnconvert -f ONNX --modelFile {root_dir}/quant_model_{epoch}.onnx --MNNModel {dst_dir}/quant_model_{epoch}.mnn --compressionParamsFile {root_dir}/compress_params_{epoch}.bin --bizCode MNN "
        command_list = command.split(" ")
        subprocess.run(command_list)
