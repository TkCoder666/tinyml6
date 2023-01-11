import subprocess
import os
from tqdm import tqdm
if __name__ == '__main__':
    epoch = 30
    bits = 8
    prune_method = "simdoc"
    root_dir = f"/sldata/liujunhan/tinyml6/saved_model/pruned_model/{prune_method}"
    dst_dir = f"/sldata/liujunhan/tinyml6/saved_model/pruned_model/mnn"
    python_bin = "/sldata/liujunhan/anaconda3/envs/tinyml/bin/python"
    sparsity_list=[0.3, 0.5, 0.8]
    for sparsity in sparsity_list:
        command = f"{python_bin} -m MNN.tools.mnnconvert -f ONNX --modelFile {root_dir}/sparsity_{sparsity}/onnx/pruned_model_best.onnx --MNNModel {dst_dir}/{prune_method}_{sparsity}.mnn --compressionParamsFile {root_dir}/sparsity_{sparsity}/bin/compress_params_best.bin --bizCode MNN --weightQuantBits 8"
        command_list = command.split(" ")
        subprocess.run(command_list)