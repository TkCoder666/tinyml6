from torch import nn
from mnncompress.pytorch import LSQQuantizer
import torch
from tqdm import tqdm
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
    Quantizer = LSQQuantizer
    model = torch.load("saved_model/original_model/resnet56_model.pth")
    bits = 8
    quantizer = Quantizer(model, retain_sparsity=False, bits=bits)
    quant_model = quantizer.convert()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_model.to(device)

    #!set epoch here,do you know
    epochs = 6
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # 每次训练之前加上这一句，准备好量化训练图
        quantizer.resume_qat_graph()

        train_loss, train_acc = train(
            quant_model, get_train_loader(64, 16), optimizer, device, criterion)
        print(
            f"[ Train | {epoch :03d}/{epochs + 1:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        valid_loss, valid_acc = test(
            quant_model, get_test_loader(64, 16), device, criterion)
        print(
            f"[ Valid | {epoch:03d}/{epochs + 1:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        # 保存模型之前去掉插入的节点，恢复原模型结构

        quantizer.strip_qat_ops()

        # 保存模型，注意index，即模型和保存MNN模型压缩参数文件是一一对应的
        quant_model.eval()
        torch.save(quant_model,
                   f"saved_model/quantized_model/LSQquantize_data_{bits}bit/quant_model_{epoch}.pth")
        x = torch.randn(1, 3, 32, 32).to(device)  # !fuck
        torch.onnx.export(
            quant_model, x, f"saved_model/quantized_model/LSQquantize_data_{bits}bit/quant_model_{epoch}.onnx")

        # 保存MNN模型压缩参数文件，必须要保存这个文件！
        # 如果进行量化的模型有剪枝，请将剪枝时生成的MNN模型压缩参数文件 "compress_params.bin" 文件名 在下方传入，并将 append 设置为True
        # append表示追加，如果此模型仅进行量化，append设为False即可！
        quantizer.save_compress_params(
            f"saved_model/quantized_model/LSQquantize_data_{bits}bit/quant_model_{epoch}.onnx", f"saved_model/quantized_model/LSQquantize_data_{bits}bit/compress_params_{epoch}.bin", append=False)
