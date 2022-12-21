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
rpc_key = "rasp4b"
# hostname of the RPC tracker
hostname = "192.168.109.30"
# port of the RPC tracker
port = 9090
# repeat times
repeat_time = 10000


if local_demo:
    target = "llvm"
else:
    # This could potentially get broken on Mac for cross-compilation.
    target = "llvm -device=arm_cpu -model=bcm2835 -mtriple=aarch64-linux-gnu -mattr=+neon"

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


def get_package(model_path, id, tuning):
    global local_demo
    global target, rpc_key, hostname, port
    # saving path
    path = "saved_model/tvm_model/resnet56_{}_{}.tar".format(id, ("" if local_demo else "rasp_") + tuning)

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
            tvmc.tune(model, target=target, rpc_key=rpc_key, hostname=hostname, port=port)
        elif tuning == "autoscheduler":  # slow but better
            tvmc.tune(model, target=target, enable_autoscheduler=True, rpc_key=rpc_key, hostname=hostname, port=port)

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

def print_output(result: tvmc.TVMCResult):
    output_idx = 'output_0'
    print(result)
    print('TVM prediction top-1: {}'.format(np.argmax(result.get_output(output_idx))))


# The model path
onnx_model_path = 'saved_model/onnx_model/resnet56.onnx'

# Compile the old model
old_package = get_package(onnx_model_path, "orig", "old")

'''
# Tuning using AutoTVM
autotvm_package = get_package(onnx_model_path, "orig", "autotvm")

# Tuning using AutoScheduler
autoscheduler_package = get_package(onnx_model_path, "orig", "autoscheduler")
'''

# Testing:
print("*** Original ***")
old_result = test_package(old_package)
print_output(old_result)
gc.collect()

'''
print("*** AutoTVM ***")
autotvm_result = test_package(autotvm_package)
print_output(autotvm_result)
gc.collect()

print("*** AutoScheduler ***")
autoscheduler_result = test_package(autoscheduler_package)
print_output(autoscheduler_result)
'''

'''
*** Original ***
2022-12-08 21:37:32.001 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpsgx2hpdc/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  10.4150       9.8591      261.3819      6.4832       8.1280   
               
Output Names:
 ['output_0']
TVM prediction top-1: 90
*** AutoTVM ***
2022-12-08 21:39:16.760 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpuy86k1hf/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   9.9201       9.4705      278.3681      5.2655       8.4989   
               
Output Names:
 ['output_0']
TVM prediction top-1: 90
*** AutoScheduler ***
2022-12-08 21:40:56.575 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpvielc7ho/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   9.0624       8.6035      261.9947      4.7345       7.0949   
               
Output Names:
 ['output_0']
TVM prediction top-1: 90
'''

'''

testdata = np.array(valset[0][0])
testdata = np.expand_dims(testdata, axis=0)
testdata = {'input.1': testdata}

print("Training ...")
for method in ["simdoc_10", "sniplevel_30", "taylorfochannel_30"]:
    sparsity_list = [0.3, 0.5, 0.8] if method != "taylorfochannel_30" else [0.3, 0.5]
    for sparsity in sparsity_list:
        id = "{}_{}".format(method, sparsity)
        onnx_model_path = "saved_model/onnx_model/{}.onnx".format(id)
        old_package = get_package(onnx_model_path, id, "old")
        autotvm_package = get_package(onnx_model_path, id, "autotvm")
gc.collect()

print("Testing ...")
for method in ["simdoc_10", "sniplevel_30", "taylorfochannel_30"]:
    sparsity_list = [0.3, 0.5, 0.8] if method != "taylorfochannel_30" else [0.3, 0.5]
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

'''

'''
*** Original simdoc_10_0.3***
2022-12-08 21:47:28.776 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpoivwrygo/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  10.4669       9.9134      271.6495      6.0622       9.0804   
               
Output Names:
 ['output_0']
TVM prediction top-1: 68
*** AutoTVM simdoc_10_0.3***
2022-12-08 21:49:14.148 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpxz_2xlh5/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  10.4166       9.9016      274.8900      5.3818       8.8519   
               
Output Names:
 ['output_0']
TVM prediction top-1: 68
*** Original simdoc_10_0.5***
2022-12-08 21:50:58.989 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpzebqb1jx/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   9.3541       8.9240      257.6106      4.8803       7.1912   
               
Output Names:
 ['output_0']
TVM prediction top-1: 71
*** AutoTVM simdoc_10_0.5***
2022-12-08 21:52:33.302 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpstmt1wr0/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   8.8503       8.4532      228.4196      4.7926       6.2059   
               
Output Names:
 ['output_0']
TVM prediction top-1: 71
*** Original simdoc_10_0.8***
2022-12-08 21:54:02.451 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpabodu7u_/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   8.9214       8.4759      220.5746      4.9315       6.2365   
               
Output Names:
 ['output_0']
TVM prediction top-1: 68
*** AutoTVM simdoc_10_0.8***
2022-12-08 21:55:32.323 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmps32qzih8/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   8.8702       8.4550      224.7396      4.8756       6.7185   
               
Output Names:
 ['output_0']
TVM prediction top-1: 68
*** Original sniplevel_30_0.3***
2022-12-08 21:15:36.324 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpz_n6co1u/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   8.8718       8.5305      253.5140      4.8221       6.8228   
               
Output Names:
 ['output_0']
TVM prediction top-1: 68
*** AutoTVM sniplevel_30_0.3***
2022-12-08 21:17:05.722 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp394k7psj/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   8.8881       8.4759      216.0146      4.8541       6.6423   
               
Output Names:
 ['output_0']
TVM prediction top-1: 68
*** Original sniplevel_30_0.5***
2022-12-08 21:18:35.212 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpexveex98/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   9.1957       8.6764      220.8111      4.8663       7.3526   
               
Output Names:
 ['output_0']
TVM prediction top-1: 68
*** AutoTVM sniplevel_30_0.5***
2022-12-08 21:20:07.891 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpfelpaopp/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   9.0330       8.5996      259.6242      4.9404       6.7892   
               
Output Names:
 ['output_0']
TVM prediction top-1: 68
*** Original sniplevel_30_0.8***
2022-12-08 21:21:39.066 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmppmc_kxrc/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   8.9046       8.5537      240.8822      4.9759       6.6121   
               
Output Names:
 ['output_0']
TVM prediction top-1: 46
*** AutoTVM sniplevel_30_0.8***
2022-12-08 21:23:08.753 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp8zq3vprg/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   9.2058       8.7619      257.5680      4.8931       6.9967   
               
Output Names:
 ['output_0']
TVM prediction top-1: 46
*** Original taylorfochannel_30_0.3***
2022-12-08 21:24:41.608 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpk4l5z_r9/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   8.9925       8.5569      242.3417      5.0245       6.8679   
               
Output Names:
 ['output_0']
TVM prediction top-1: 68
*** AutoTVM taylorfochannel_30_0.3***
2022-12-08 21:26:12.237 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp92j424jd/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   8.9280       8.5512      229.3331      4.8591       6.5644   
               
Output Names:
 ['output_0']
TVM prediction top-1: 68
*** Original taylorfochannel_30_0.5***
2022-12-08 21:27:42.143 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpemb9d28v/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   9.1836       8.7751      262.5479      4.4571       7.0899   
               
Output Names:
 ['output_0']
TVM prediction top-1: 12
*** AutoTVM taylorfochannel_30_0.5***
2022-12-08 21:29:14.719 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpjrx4izfq/mod.so
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   9.0752       8.6868      235.2542      4.9802       7.0347   
               
Output Names:
 ['output_0']
TVM prediction top-1: 12
'''

'''

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

'''
*** Original quant_model_1***
2022-12-08 21:42:27.825 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmpcts_gms2/mod.so
Execution time summary:
mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
9.8075       9.3558      263.3517      5.0134       7.8135

Output Names:
['output_0']
TVM prediction top-1: 68
*** AutoTVM quant_model_1***
2022-12-08 21:44:06.473 INFO load_module /var/folders/8b/z2mgqcpj6flcwmnj5jj00j6r0000gn/T/tmp8jtokmso/mod.so
Execution time summary:
mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
9.6083       9.1280      272.7568      4.9825       7.8099

Output Names:
['output_0']
TVM prediction top-1: 68
'''
