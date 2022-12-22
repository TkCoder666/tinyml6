import numpy as np
import onnx
from tqdm import tqdm
import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, graph_executor as runtime
import time
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

from data_provider import *
from benchmarking_utils import AccuracyMetric

# control the local testing or not
local_demo = True
# repeat time
repeat = 10000
# number of loops for one test
timing_number = 1

# prepare the data
testdata = valset[0]
x = np.array(testdata[0])
y = testdata[1]
# np.savez("deploy/test", input=x)

# load the model
onnx_model = onnx.load('saved_model/onnx_model/resnet56.onnx')

# convert to relay
input_name = "input"
shape_dict = {input_name: (1,) + x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(
    func.body), None, func.type_params, func.attrs)

# Build package.

if local_demo:
    target = tvm.target.Target("llvm")
else:
    # target = tvm.target.arm_cpu("rasp3b")
    # The above line is a simple form of
    # target = tvm.target.Target('llvm -device=arm_cpu -model=bcm2837 -mtriple=armv7l-linux-gnueabihf -mattr=+neon')
    # target = tvm.target.Target('llvm -device=arm_cpu -model=bcm2837 -mtriple=aarch64-linux-gnu -mattr=+neon')
    target = tvm.target.Target(
        'llvm -device=arm_cpu -model=bcm2835 -mtriple=aarch64-linux-gnu -mattr=+neon')
    # https://discuss.tvm.apache.org/t/tutorial-errors/7962/9
    # It is still failed to work on my MacBook to QEMU Raspberry Pi.

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target, params=params)

# After `relay.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library.
lib_fname = 'resnet56_rpc.tar'
lib_fpath = 'saved_model/tvm_model/' + lib_fname
lib.export_library(lib_fpath)

# obtain an RPC session from remote device.
if local_demo:
    remote = rpc.LocalSession()
else:
    # The following is my environment, change this to the IP address of your target device
    host = "192.168.109.30"
    port = 9090
    remote = rpc.connect(host, port)

    ## FOR DIRECT RUNNING ON REMOTE MACHINE
    # remote.upload(lib_fpath, "/home/pi/resnet56_rpc.tar")
    # remote.upload('deploy/tvm_on_rasp.py', '/home/pi/tvm_on_rasp.py')
    # remote.upload('deploy/test.npz', '/home/pi/test.npz')

# upload the library to remote device and load it
remote.upload(lib_fpath)
rlib = remote.load_module(lib_fname)

# create the remote runtime module
dev = remote.cpu(0)
module = runtime.GraphModule(rlib["default"](dev))
# set input data
module.set_input(input_name, tvm.nd.array(x.astype("float32")))
# run
module.run()
# get output
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.numpy())
print("TVM prediction top-1: {}, {}".format(top1,valset.classes[top1]))
print("Target is: {}, {}".format(y,valset.classes[y]))

def benchmark_tvm(mod: runtime.GraphModule):
    global input_name
    accuracy_metric = AccuracyMetric(topk=(1,5))
    benchmarktimes = []
    for data in tqdm(valset):
        mod.set_input(input_name, tvm.nd.array(np.array(data[0])))
        start_time = time.time()
        mod.run()
        end_time = time.time()
        benchmarktimes.append(end_time - start_time)
        result = module.get_output(0)
        accuracy_metric.update(torch.Tensor(result.numpy()),torch.Tensor(data[1]))
    print(
        # f'Accuracy of the network on the 10000 test images: {100 * correct // total} %',
        f"top1_acc {accuracy_metric.at(1).rate} ",
        f"top5_acc {accuracy_metric.at(5).rate} ",
        f"mean: {np.mean(benchmarktimes)}, median: {np.mean(benchmarktimes)}, std: {np.std(benchmarktimes)}")


benchmark_tvm(module)

