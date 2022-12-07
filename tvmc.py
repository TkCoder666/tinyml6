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
# python -m tvm.exec.rpc_server --tracker=10.0.2.2:9190 --key=rasp3b --port=9090
# # On the host to get the current status:
# python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190

# RPC key
rpc_key = "rasp3b"
# hostname of the RPC tracker
hostname = "127.0.0.1"
# port of the RPC tracker
port = 9090
# repeat times
repeat_time = 10


if local_demo:
    target = "llvm"
else:
    # This could potentially get broken on Mac for cross-compilation.
    target = "llvm -device=arm_cpu -model=bcm2837 -mtriple=aarch64-linux-gnu -mattr=+neon"

    ## Setup the rpc tracker on the host machine
    # python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=8180
    ## Then start the rpc server on the client
    # sudo ifup -a
    # sudo ip route # <- will get [HOST IP].
    # python -m tvm.exec.rpc_server --tracker=[HOST IP]:9190 --key=rasp3b --port=9090

## Prepare the data.

testdata = np.array(valset[0][0])
testdata = np.expand_dims(testdata, axis=0)
testdata = {'input': testdata}

## Try data externally:

# testdata = valset.data
# testdata = testdata.transpose((0, 3, 1, 2))  # HWC -> NCWH, convert back for ONNX
# testdata = testdata[0]
# testdata = np.expand_dims(testdata, axis=0)

# input_dir = "data/tvm_data"
# if not os.path.isdir(input_dir):
#     os.mkdir(input_dir)
# np.savez(input_dir + "/test", input=testdata)

## Tune the model.


def get_package(model_path, tuning):
    global local_demo
    global target, rpc_key, hostname, port
    # saving path
    path = "saved_model/tvm_model/resnet56_{}.tar".format(("" if local_demo else "rasp_") + tuning)

    # The package is cached.
    if os.path.isfile(path):
        return tvmc.TVMCPackage(package_path=path)

    # model will be loaded cleanly everytime.
    model = tvmc.load(model_path)

    if local_demo:
        # Tuning
        if tuning == "old":  # compile only
            pass
        elif tuning == "autotvm":  # fast but worse
            tvmc.tune(model, target=target, rpc_key=rpc_key, hostname=hostname, port=port)
        elif tuning == "autoscheduler":  # slow but better
            tvmc.tune(model, target=target, enable_autoscheduler=True, rpc_key=rpc_key, hostname=hostname, port=port)
    else:
        if tuning == "old":  # compile only
            pass
        elif tuning == "autotvm":  # fast but worse
            tvmc.tune(model, target=target, )
        elif tuning == "autoscheduler":  # slow but better
            tvmc.tune(model, target=target, enable_autoscheduler=True)

    # Compile
    return tvmc.compile(model, target=target, package_path=path)


def test_package(package):
    global local_demo, repeat_time
    global rpc_key, hostname, port
    # only simulates 10000 pictures from the same picture.
    if local_demo:
        return tvmc.run(package, device="cpu", inputs=testdata, repeat=repeat_time, benchmark=True, number=1)
    else:
        return tvmc.run(package, device="cpu", inputs=testdata, repeat=repeat_time, benchmark=True, number=1, rpc_key=rpc_key, hostname=hostname, port=port)

# The model path
onnx_model_path = 'saved_model/onnx_model/resnet56.onnx'

# Compile the old model
old_package = get_package(onnx_model_path, "old")

# Tuning using AutoTVM
autotvm_package = get_package(onnx_model_path, "autotvm")

# Tuning using AutoScheduler
autoscheduler_package = get_package(onnx_model_path, "autoscheduler")


def print_output(result: tvmc.TVMCResult):
    output_idx = 'output_0'
    print(result)
    print('TVM prediction top-1: {}'.format(np.argmax(result.get_output(output_idx))))

# Testing:
print("*** Original ***")
old_result = test_package(old_package)
print_output(old_result)
gc.collect()

print("*** AutoTVM ***")
autotvm_result = test_package(autotvm_package)
print_output(autotvm_result)
gc.collect()

print("*** AutoScheduler ***")
autoscheduler_result = test_package(autoscheduler_package)
print_output(autoscheduler_result)

'''
Test Result on MacBook Pro M1 (arm64)

Files already downloaded and verified
Files already downloaded and verified
2022-11-23 22:13:56.104 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp6wjcewqp/mod.so
*** Original ***
Execution time summary:
mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
11.6242      10.3862      284.3319      4.8533      10.0938

Output Names:
['output_0']
2022-11-23 22:15:53.243 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp5d4_souc/mod.so
*** AutoTVM ***
Execution time summary:
mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
10.7392       9.6786      297.6342      4.9580       9.9129

Output Names:
['output_0']
2022-11-23 22:17:41.477 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp_nwhcn0j/mod.so
*** AutoScheduler ***
Execution time summary:
mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
10.4682       9.4239      275.7486      4.8571       9.5825

Output Names:
['output_0']
'''