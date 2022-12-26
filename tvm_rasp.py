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
local_demo = True
# repeat time
repeat = 10000
# number of loops for one test
timing_number = 1

# The following is my environment, change this to the IP address of your target device
host = "192.168.109.30"
port = 9090

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

    ## Validation

    # set input data
    module.set_input(input_name, tvm.nd.array(x))
    # run
    module.run()
    # get output
    out = module.get_output(0)
    # get top1 result
    top1 = np.argmax(out.numpy())
    print("TVM prediction top-1: {}, {}".format(top1,valset.classes[top1]))
    print("Target is: {}, {}".format(y,valset.classes[y]))

    ## Benchmark

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
Test on Macbook M1
2022-12-26 18:37:40.654 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp9zc5de1e/resnet56_rpc.tar
TVM prediction top-1: 72, seal
Target is: 72, seal
*** resnet56 ***
mean (ms) 	 median (ms) 	 std (ms)
14.082631418099998 	 13.015396	 11.839304837501961

2022-12-26 18:40:14.358 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpiszzn7i0/resnet56_rpc.tar
TVM prediction top-1: 72, seal
Target is: 72, seal
*** simdoc_10_0.3 ***
mean (ms) 	 median (ms) 	 std (ms)
14.2675391139 	 12.9646455	 12.226450871361818

2022-12-26 18:42:50.029 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpz9adqjd4/resnet56_rpc.tar
TVM prediction top-1: 72, seal
Target is: 72, seal
*** simdoc_10_0.5 ***
mean (ms) 	 median (ms) 	 std (ms)
12.8628694449 	 10.293958499999999	 14.059292691472391

2022-12-26 18:45:11.187 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmppjyx3y_y/resnet56_rpc.tar
TVM prediction top-1: 93, turtle
Target is: 72, seal
*** simdoc_10_0.8 ***
mean (ms) 	 median (ms) 	 std (ms)
11.359193027200002 	 9.782312000000001	 10.363412836493882

2022-12-26 18:47:17.531 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpjlk2q6w2/resnet56_rpc.tar
TVM prediction top-1: 72, seal
Target is: 72, seal
*** sniplevel_30_0.3 ***
mean (ms) 	 median (ms) 	 std (ms)
10.338644433099999 	 9.6429795	 7.457446302935148

2022-12-26 18:49:13.466 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpgf_z2kn6/resnet56_rpc.tar
TVM prediction top-1: 55, otter
Target is: 72, seal
*** sniplevel_30_0.5 ***
mean (ms) 	 median (ms) 	 std (ms)
10.491052674599999 	 9.687249999999999	 8.719743304577923

2022-12-26 18:51:10.903 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpod7irrtk/resnet56_rpc.tar
TVM prediction top-1: 93, turtle
Target is: 72, seal
*** sniplevel_30_0.8 ***
mean (ms) 	 median (ms) 	 std (ms)
10.619573171600003 	 9.701125000000001	 8.783563552663974

2022-12-26 18:53:09.837 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp0oibaz2p/resnet56_rpc.tar
TVM prediction top-1: 55, otter
Target is: 72, seal
*** taylorfochannel_30_0.3 ***
mean (ms) 	 median (ms) 	 std (ms)
10.469277588699999 	 9.642978999999999	 8.959543339215124

2022-12-26 18:55:07.122 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp19fi5nn3/resnet56_rpc.tar
TVM prediction top-1: 93, turtle
Target is: 72, seal
*** taylorfochannel_30_0.5 ***
mean (ms) 	 median (ms) 	 std (ms)
11.160059567600001 	 9.791770500000002	 10.366580582829103

2022-12-26 18:57:11.484 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp5rc3dcls/resnet56_rpc.tar
TVM prediction top-1: 72, seal
Target is: 72, seal
*** quant_model_1 ***
mean (ms) 	 median (ms) 	 std (ms)
10.7142628161 	 9.770812	 10.224622694416494

'''