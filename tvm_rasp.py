import gc

import numpy as np
import onnx
# from tqdm import tqdm
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
local_demo = False
# repeat time
repeat = 10000
# number of loops for one test
timing_number = 1

# The following is my environment, change this to the IP address of your target device
host = "192.168.109.30"
port = 9091

# prepare the data
testdata = valset[2]
x = testdata[0].numpy()
# x = np.array(testdata[0], dtype=float)
# x = np.array(testdata[0]).transpose((2, 0, 1))
y = testdata[1]
# np.savez("deploy/test", input=x)


def tvm_benchmark(onnx_name, input_name="input"):
    global local_demo, repeat, timing_number, host, port

    # load the model
    onnx_model = onnx.load('saved_model/onnx_model/{}.onnx'.format(onnx_name))

    # convert to relay
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
        remote = rpc.connect(host, port)

        # FOR DIRECT RUNNING ON REMOTE MACHINE
        # remote.upload(lib_fpath, "/home/pi/resnet56_rpc.tar")
        # remote.upload('deploy/tvm_on_rasp.py', '/home/pi/tvm_on_rasp.py')
        # remote.upload('deploy/test.npz', '/home/pi/test.npz')

    # upload the library to remote device and load it
    remote.upload(lib_fpath)
    rlib = remote.load_module(lib_fname)

    # create the remote runtime module
    dev = remote.cpu(0)
    module = runtime.GraphModule(rlib["default"](dev))

    # Validation

    # set input data
    module.set_input(input_name, tvm.nd.array(x))
    # run
    module.run()
    # get output
    out = module.get_output(0)
    # get top1 result
    top1 = np.argmax(out.numpy())
    print("TVM prediction top-1: {}, {}".format(top1, valset.classes[top1]))
    print("Target is: {}, {}".format(y, valset.classes[y]))

    # Benchmark

    result = module.benchmark(dev, repeat=repeat, number=timing_number)
    print("*** {} ***".format(onnx_name))
    print("mean (ms) \t median (ms) \t std (ms)\n{} \t {}\t {}\n"
          .format(result.mean * 1000, result.median * 1000, result.std * 1000))

    # The time is not accurate: Network delay is counted.
    #
    # valset = CIFAR100(root="./data", train=False, download=True)
    # accuracy_metric = AccuracyMetric(topk=(1,5))
    # benchmarktimes = []
    # for data in tqdm(valset):
    #     mod.set_input(input_name, tvm.nd.array(np.array(data[0])))
    #     start_time = time.time()
    #     mod.run()
    #     end_time = time.time()
    #     benchmarktimes.append(end_time - start_time)
    #     result = module.get_output(0)
    #     accuracy_metric.update(torch.Tensor(result.numpy()),torch.Tensor(data[1]))
    # print(
    #     # f'Accuracy of the network on the 10000 test images: {100 * correct // total} %',
    #     f"top1_acc {accuracy_metric.at(1).rate} ",
    #     f"top5_acc {accuracy_metric.at(5).rate} ",
    #     f"mean: {np.mean(benchmarktimes)}, median: {np.mean(benchmarktimes)}, std: {np.std(benchmarktimes)}")

    gc.collect()


tvm_benchmark("resnet56", "input")
tvm_benchmark("simdoc_10_0.3", "input.1")
tvm_benchmark("simdoc_10_0.5", "input.1")
tvm_benchmark("simdoc_10_0.8", "input.1")
tvm_benchmark("sniplevel_30_0.3", "input.1")
tvm_benchmark("sniplevel_30_0.5", "input.1")
tvm_benchmark("sniplevel_30_0.8", "input.1")
tvm_benchmark("taylorfochannel_30_0.3", "input.1")
tvm_benchmark("taylorfochannel_30_0.5", "input.1")
tvm_benchmark("quant_model_1", "onnx::Conv_0")

'''
Test on Macbook Raspberry Pi 4b
dense is not optimized for arm cpu.
dense is not optimized for arm cpu.
TVM prediction top-1: 72, seal
Target is: 72, seal
*** resnet56 ***
mean (ms)        median (ms)     std (ms)
16.910491272     16.523597000000002      4.19621338496957

dense is not optimized for arm cpu.
dense is not optimized for arm cpu.
TVM prediction top-1: 55, otter
Target is: 72, seal
*** simdoc_10_0.3 ***
mean (ms)        median (ms)     std (ms)
16.8499222317    16.454292000000002      4.2546969438207345

dense is not optimized for arm cpu.
dense is not optimized for arm cpu.
TVM prediction top-1: 72, seal
Target is: 72, seal
*** simdoc_10_0.5 ***
mean (ms)        median (ms)     std (ms)
16.749172498799997       16.3910445      4.125120827961239

dense is not optimized for arm cpu.
dense is not optimized for arm cpu.
TVM prediction top-1: 93, turtle
Target is: 72, seal
*** simdoc_10_0.8 ***
mean (ms)        median (ms)     std (ms)
17.0440610278    16.6572565      4.420074868477964

dense is not optimized for arm cpu.
dense is not optimized for arm cpu.
TVM prediction top-1: 44, lizard
Target is: 72, seal
*** sniplevel_30_0.3 ***
mean (ms)        median (ms)     std (ms)
16.937319465199998       16.559012499999998      4.229847561365775

dense is not optimized for arm cpu.
dense is not optimized for arm cpu.
TVM prediction top-1: 55, otter
Target is: 72, seal
*** sniplevel_30_0.5 ***
mean (ms)        median (ms)     std (ms)
17.074978281899998       16.687881500000003      4.466113242429598

dense is not optimized for arm cpu.
dense is not optimized for arm cpu.
TVM prediction top-1: 27, crocodile
Target is: 72, seal
*** sniplevel_30_0.8 ***
mean (ms)        median (ms)     std (ms)
17.1457452844    16.766666999999998      4.179864639001043

dense is not optimized for arm cpu.
dense is not optimized for arm cpu.
TVM prediction top-1: 4, beaver
Target is: 72, seal
*** taylorfochannel_30_0.3 ***
mean (ms)        median (ms)     std (ms)
16.985623493700004       16.609406       4.0683502671606595

dense is not optimized for arm cpu.
dense is not optimized for arm cpu.
TVM prediction top-1: 55, otter
Target is: 72, seal
*** taylorfochannel_30_0.5 ***
mean (ms)        median (ms)     std (ms)
17.138288106899996       16.7489545      4.20717198063859

dense is not optimized for arm cpu.
dense is not optimized for arm cpu.
TVM prediction top-1: 72, seal
Target is: 72, seal
*** quant_model_1 ***
mean (ms)        median (ms)     std (ms)
17.046436611700003       16.673063000000003      4.104875497812671
'''
