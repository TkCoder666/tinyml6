# Compiling TVMC is a painstaking work.
# Try to use high level interface TVMC.
# The low level one gives more possibilities in tuning,
# but it is now for demo use.
import gc
import os
import numpy as np
from data_provider import *
from tvm.driver import tvmc


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
    # saving path
    path = "saved_model/tvm_model/resnet56_{}.tar".format(tuning)

    # The package is cached.
    if os.path.isfile(path):
        return tvmc.TVMCPackage(package_path=path)

    # model will be loaded cleanly everytime.
    model = tvmc.load(model_path)

    # Tuning
    if tuning == "old":  # compile only
        pass
    elif tuning == "autotvm":  # fast but worse
        tvmc.tune(model, target="llvm")
    elif tuning == "autoscheduler":  # slow but better
        tvmc.tune(model, target="llvm", enable_autoscheduler=True)

    # Compile
    return tvmc.compile(model, target="llvm", package_path=path)


def test_package(package):
    # only simulates 10000 pictures from the same picture.
    return tvmc.run(package, device="cpu", inputs=testdata, repeat=10000, benchmark=True, number=1)


# The model path
onnx_model_path = 'saved_model/onnx_model/resnet56.onnx'

# Compile the old model
old_package = get_package(onnx_model_path, "old")

# Tuning using AutoTVM
autotvm_package = get_package(onnx_model_path, "autotvm")

# Tuning using AutoScheduler
autoscheduler_package = get_package(onnx_model_path, "autoscheduler")

# Testing:
print("*** Original ***\n", test_package(old_package))
gc.collect()
print("*** AutoTVM ***\n", test_package(autotvm_package))
gc.collect()
print("*** AutoScheduler ***\n", test_package(autoscheduler_package))

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