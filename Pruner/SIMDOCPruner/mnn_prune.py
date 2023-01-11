from torch import nn
from context import *
from mnncompress.pytorch.SNIP_level_pruner import SNIPLevelPruner
from mnncompress.pytorch.SIMD_OC_pruner import SIMDOCPruner
from mnncompress.pytorch.TaylorFO_channel_pruner import TaylorFOChannelPruner
import torch
from tqdm import tqdm
import sys


from data_provider import get_train_loader, get_test_loader


def train(model, train_loader, optimizer, device, criterion):
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

        pruner.do_pruning()

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    return train_loss, train_acc


def test(model, testloader, device, criterion):
    model.eval()
    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(testloader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    return valid_loss, valid_acc


if __name__ == '__main__':
    usecuda=True
    model = torch.load("saved_model/original_model/resnet56_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() and usecuda else "cpu")
    model.to(device)
    sparsity=0.8
    pruner = SIMDOCPruner(model, total_pruning_iterations=5, sparsity=sparsity, debug_info=False, prune_finetune_iterations=99,max_prune_ratio=0.99)

    prune_method="simdoc"
    if not os.path.exists(f'saved_model/pruned_model/{prune_method}/sparsity_{sparsity}'): #判断所在目录下是否有该文件名的文件夹
        os.makedirs(f'saved_model/pruned_model/{prune_method}/sparsity_{sparsity}')
        os.makedirs(f'saved_model/pruned_model/{prune_method}/sparsity_{sparsity}/pth')
        os.makedirs(f'saved_model/pruned_model/{prune_method}/sparsity_{sparsity}/onnx')
        os.makedirs(f'saved_model/pruned_model/{prune_method}/sparsity_{sparsity}/bin')
    #!set epoch here,do you know
    epochs = 10
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_valid_acc = 0
    for epoch in range(1, epochs + 1):
        # 每次训练之前加上这一句，准备好量化训练图

        train_loss, train_acc = train(
            model, get_train_loader(128, 16), optimizer, device, criterion)
        print(
            f"[ Train | {epoch :03d}/{epochs + 1:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        valid_loss, valid_acc = test(
            model, get_test_loader(128, 16), device, criterion)
        print(
            f"[ Valid | {epoch:03d}/{epochs + 1:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        print(pruner.current_prune_ratios())
        model.eval()
        
        torch.save(model.state_dict(), f"saved_model/pruned_model/{prune_method}/sparsity_{sparsity}/pth/pruned_model_{epoch}.pth")
        x = torch.randn(1,3,32,32).to(device)
        torch.onnx.export(model, x, f"saved_model/pruned_model/{prune_method}/sparsity_{sparsity}/onnx/pruned_model_{epoch}.onnx")

        # 保存MNN模型压缩参数文件，应在剪枝完毕之后进行，建议在保存模型时调用
        pruner.save_compress_params(f"saved_model/pruned_model/{prune_method}/sparsity_{sparsity}/bin/compress_params_{epoch}.bin", append=False)
        if valid_acc>best_valid_acc:
            best_valid_acc=valid_acc
            torch.save(model.state_dict(), f"saved_model/pruned_model/{prune_method}/sparsity_{sparsity}/pth/pruned_model_best.pth")
            x = torch.randn(1,3,32,32).to(device)
            torch.onnx.export(model, x, f"saved_model/pruned_model/{prune_method}/sparsity_{sparsity}/onnx/pruned_model_best.onnx")

            # 保存MNN模型压缩参数文件，应在剪枝完毕之后进行，建议在保存模型时调用
            pruner.save_compress_params(f"saved_model/pruned_model/{prune_method}/sparsity_{sparsity}/bin/compress_params_best.bin", append=False)