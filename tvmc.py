# Compiling TVMC is a painstaking work.
# Try use high level interface TVMC.
# The low level one gives more possibilities in tuning,
# but it is now for demo use.

from tvm.driver import tvmc

# Step 1: Load the model
model = tvmc.load('saved_model/onnx_model/resnet56.onnx')

# Step 2: Compile (Before)
old_package = tvmc.compile(model, target="llvm", package_path="saved_model/tvm_model/resnet56_before.tar")

# Step 3: Run (Before)
result = tvmc.run(old_package, device="cpu")
print(result)

# Step 4: Tuning
tvmc.tune(model, target="llvm", enable_autoscheduler=True)

# Step 2: Compile (After)
new_package = tvmc.compile(model, target="llvm", package_path="saved_model/tvm_model/resnet56_after.tar")

# Step 3: Run (After)
result = tvmc.run(new_package, device="cpu")
print(result)

