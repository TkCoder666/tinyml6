import tvm
from tvm.contrib import graph_executor as runtime
import numpy as np

target = "llvm"
input_name = "input"
x = np.load("test.npz")[input_name]

lib = tvm.runtime.load_module("resnet56_rpc.tar")

module = runtime.GraphModule(lib["default"](tvm.device(target, 0)))

module.set_input(input_name, tvm.nd.array(x.astype("float32")))
module.run()
# get output
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.numpy())
print("TVM prediction top-1: {}".format(top1))