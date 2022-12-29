# Compiling TVMC is a painstaking work.
# Try to use high level interface TVMC.
# The low level one gives more possibilities in tuning,
# but it is now for demo use.
import gc
import os
import numpy as np
from data_provider import *
from tvm.driver import tvmc

# local demo or the remote raspberry pi.
local_demo = False

# # Setup the RPC tracker.
# # On the host:
# python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
# # On the client, After compile the runtime:
# python -m tvm.exec.rpc_server --tracker=[IP]:9190 --key=rasp4b --port=9090
# # On the host to get the current status:
# python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190

# RPC key
rpc_key = "rasp4b"
# hostname of the RPC tracker
hostname = "0.0.0.0"
# port of the RPC tracker
port = 9190
# repeat times
repeat_time = 10000


if local_demo:
    target = "llvm"
else:
    # This could potentially get broken on Mac for cross-compilation.
    target = "llvm -device=arm_cpu -model=bcm2835 -mtriple=aarch64-linux-gnu -mattr=+neon"

    # Setup the rpc tracker on the host machine
    # python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=8180
    # Then start the rpc server on the client
    # sudo ifup -a
    # sudo ip route # <- will get [HOST IP].
    # python -m tvm.exec.rpc_server --tracker=[HOST IP]:9190 --key=rasp3b --port=9090

# Prepare the data.

testdata = np.array(valset[0][0])
testdata = np.expand_dims(testdata, axis=0)
testdata = {'input': testdata}

# Try data externally:

# testdata = valset.data
# testdata = testdata.transpose((0, 3, 1, 2))  # HWC -> NCWH, convert back for ONNX
# testdata = testdata[0]
# testdata = np.expand_dims(testdata, axis=0)

# input_dir = "data/tvm_data"
# if not os.path.isdir(input_dir):
#     os.mkdir(input_dir)
# np.savez(input_dir + "/test", input=testdata)

# Tune the model.


def get_package(model_path, id, tuning):
    global local_demo
    global target, rpc_key, hostname, port
    # saving path
    path = "saved_model/tvm_model/resnet56_{}_{}.tar".format(
        id, ("" if local_demo else "rasp_") + tuning)

    # The package is cached.
    if os.path.isfile(path):
        return tvmc.TVMCPackage(package_path=path)

    # model will be loaded cleanly everytime.
    model = tvmc.load(model_path)

    if local_demo:
        if tuning == "old":  # compile only
            pass
        elif tuning == "autotvm":  # fast but worse
            tvmc.tune(model, target=target)
        elif tuning == "autoscheduler":  # slow but better
            tvmc.tune(model, target=target, enable_autoscheduler=True)
    else:
        # Tuning
        if tuning == "old":  # compile only
            pass
        elif tuning == "autotvm":  # fast but worse
            tvmc.tune(model, target=target, rpc_key=rpc_key,
                      hostname=hostname, port=port)
        elif tuning == "autoscheduler":  # slow but better
            tvmc.tune(model, target=target, enable_autoscheduler=True,
                      rpc_key=rpc_key, hostname=hostname, port=port)

    # Compile
    return tvmc.compile(model, target=target, package_path=path, output_format="tar")


def test_package(package):
    global local_demo, repeat_time
    global rpc_key, hostname, port
    # only simulates 10000 pictures from the same picture.
    if local_demo:
        return tvmc.run(package, device="cpu", inputs=testdata, repeat=repeat_time, benchmark=True, number=1)
    else:
        return tvmc.run(package, device="cpu", inputs=testdata, repeat=repeat_time, benchmark=True, number=1, rpc_key=rpc_key, hostname=hostname, port=port)


def print_output(result: tvmc.TVMCResult):
    output_idx = 'output_0'
    print(result)
    print('TVM prediction top-1: {}'.format(np.argmax(result.get_output(output_idx))))


# The model path
onnx_model_path = 'saved_model/onnx_model/resnet56.onnx'

# Compile the old model
old_package = get_package(onnx_model_path, "orig", "old")

# Tuning using AutoTVM
autotvm_package = get_package(onnx_model_path, "orig", "autotvm")

# Tuning using AutoScheduler
# autoscheduler_package = get_package(onnx_model_path, "orig", "autoscheduler")

# Testing:
print("*** Original ***")
old_result = test_package(old_package)
print_output(old_result)
gc.collect()

print("*** AutoTVM ***")
autotvm_result = test_package(autotvm_package)
print_output(autotvm_result)
gc.collect()

# print("*** AutoScheduler ***")
# autoscheduler_result = test_package(autoscheduler_package)
# print_output(autoscheduler_result)

testdata = np.array(valset[0][0])
testdata = np.expand_dims(testdata, axis=0)
testdata = {'input.1': testdata}

print("Training ...")
for method in ["simdoc_10", "sniplevel_30", "taylorfochannel_30"]:
    sparsity_list = [0.3]
    for sparsity in sparsity_list:
        id = "{}_{}".format(method, sparsity)
        onnx_model_path = "saved_model/onnx_model/{}.onnx".format(id)
        old_package = get_package(onnx_model_path, id, "old")
        autotvm_package = get_package(onnx_model_path, id, "autotvm")
gc.collect()


print("Testing ...")
for method in ["simdoc_10", "sniplevel_30", "taylorfochannel_30"]:
    sparsity_list = [0.3]
    for sparsity in sparsity_list:
        id = "{}_{}".format(method, sparsity)
        onnx_model_path = "saved_model/onnx_model/{}.onnx".format(id)
        old_package = get_package(onnx_model_path, id, "old")
        autotvm_package = get_package(onnx_model_path, id, "autotvm")

        print("*** Original {}***".format(id))
        old_result = test_package(old_package)
        print_output(old_result)
        gc.collect()

        print("*** AutoTVM {}***".format(id))
        autotvm_result = test_package(autotvm_package)
        print_output(autotvm_result)
        gc.collect()

testdata = np.array(valset[0][0])
testdata = np.expand_dims(testdata, axis=0)
testdata = {'onnx::Conv_0': testdata}

id = "quant_model_1"
onnx_model_path = "saved_model/onnx_model/{}.onnx".format(id)
old_package = get_package(onnx_model_path, id, "old")
autotvm_package = get_package(onnx_model_path, id, "autotvm")

print("*** Original {}***".format(id))
old_result = test_package(old_package)
print_output(old_result)
gc.collect()

print("*** AutoTVM {}***".format(id))
autotvm_result = test_package(autotvm_package)
print_output(autotvm_result)
gc.collect()

'''
*** Original ***
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
  16.8361      16.3975      97.6158      15.9718       4.1619

Output Names:
 ['output_0']
TVM prediction top-1: 68
*** AutoTVM ***
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
  16.7696      16.4137      88.3263      16.0163       4.0416

Output Names:
 ['output_0']
TVM prediction top-1: 68
Testing ...
*** Original simdoc_10_0.3***
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
  16.7981      16.3593      94.6272      15.9921       4.3394

Output Names:
 ['output_0']
TVM prediction top-1: 90
*** AutoTVM simdoc_10_0.3***
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
  16.9372      16.5540      94.1881      16.1535       3.7096

Output Names:
 ['output_0']
TVM prediction top-1: 90
'''
