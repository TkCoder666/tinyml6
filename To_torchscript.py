import torch
pt_model_path = 'saved_model/original_model/resnet56_model.pth'
# ...
#  model is exported model
model = torch.load(pt_model_path)
model.eval()
# trace
# model_trace = torch.jit.trace(model, torch.rand(1, 3, 1200, 1200))
# model_trace.save('model_trace.pt')
# script
model_script = torch.jit.script(model)
model_script.save('model_script.pt')