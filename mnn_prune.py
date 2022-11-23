from mnncompress.pytorch.SNIP_level_pruner import SNIPLevelPruner
from mnncompress.pytorch.SIMD_OC_pruner import SIMDOCPruner
from mnncompress.pytorch.TaylorFO_channel_pruner import TaylorFOChannelPruner
import torch.nn.functional as F
import torch.nn as nn
Pruner = SIMDOCPruner
import torch
from data_provider import get_test_loader, get_train_loader
from benchmarking_utils import AccuracyMetric

# 你的模型代码
# class Net(nn.Module):
#     pass

pth_path = "saved_model/original_model/resnet56_model.pth"

model = torch.load(pth_path)
device = torch.device( "cpu")
model.to(device)
# 加载已经训练好的模型
# model.load_state_dict(torch.load("ori_model.pt"))
# model.to(device)


# 将模型进行转换，并使用转换后的模型进行训练，测试
# 更多配置请看API部分
learning_rate=0.001
# pruner = SNIPLevelPruner(model, total_pruning_iterations=1, sparsity=0.9, debug_info=False, 
# prune_finetune_iterations=9,max_prune_ratio=0.99)
# pruner = SIMDOCPruner(model, total_pruning_iterations=1, sparsity=0.9, debug_info=False, 
# prune_finetune_iterations=9,max_prune_ratio=0.99)
pruner = TaylorFOChannelPruner(model, total_pruning_iterations=1, sparsity=0.7, debug_info=False, 
prune_finetune_iterations=9,max_prune_ratio=0.99,
 align_channels=4 )
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

def train(model, data, optimizer, pruner):
    model.train()
    for d, t in data:
        optimizer.zero_grad()
        output = model(d)
        loss = F.nll_loss(output, t)
        loss.backward()
        optimizer.step()
        
        
        # step之后调用pruner的剪枝方法
        pruner.do_pruning()
    
    
    # 获取当前剪枝比例
    print(pruner.current_prune_ratios())

def test(model, data):
    model.eval()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    with torch.no_grad():
        for d, t in data:
            # optimizer.zero_grad()
            output = model(d)
            accuracy_metric.update(output, t)
    print(
        # f'Accuracy of the network on the 10000 test images: {100 * correct // total} %',
        f"top1_acc {accuracy_metric.at(1).rate} ",
        f"top5_acc {accuracy_metric.at(5).rate}")

    
    # 获取当前剪枝比例
    # print(pruner.current_prune_ratios())

epochs = 1 # 发现训练下降
val_loader=get_test_loader(256,16)
for epoch in range(1, epochs + 1):
    train(model, val_loader, optimizer, pruner)
    test(model, val_loader)


# 保存模型
model.eval()
torch.save(model.state_dict(), "pruned_model.pt")
x = torch.randn(1,3,32,32).to(device)
torch.onnx.export(model, x, "pruned_model.onnx")

# 保存MNN模型压缩参数文件，应在剪枝完毕之后进行，建议在保存模型时调用
pruner.save_compress_params("compress_params.bin", append=False)