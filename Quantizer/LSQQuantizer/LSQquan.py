from torch import nn
from mnncompress.pytorch import LSQQuantizer
import torch
from tqdm import tqdm
from data_provider import train_loader, val_loader


def train(model, train_loader, optimizer, device):
    for batch in tqdm(train_loader):
        imgs, labels = batch
        model = model.to(device)
        logits = model(imgs.to(device))


def test(moedel, testloader, device):
    for batch in tqdm(testloader):
        imgs, labels = batch
        model = model.to(device)
        logits = model(imgs.to(device))


if __name__ == '__main__':
    Quantizer = LSQQuantizer
    model = torch.load("saved_model/original_model/resnet56_model.pth")
    quantizer = Quantizer(model, retain_sparsity=False)
    quant_model = quantizer.convert()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1

    for epoch in range(1, epochs + 1):
        # 每次训练之前加上这一句，准备好量化训练图
        quantizer.resume_qat_graph()

        train(quant_model, train_loader, None, device)
        test(quant_model, val_loader, device)

        if True:
            # 保存模型之前去掉插入的节点，恢复原模型结构
            quantizer.strip_qat_ops()

            # 保存模型，注意index，即模型和保存MNN模型压缩参数文件是一一对应的
            quant_model.eval()
            torch.save(quant_model.state_dict(), "quant_model_index.pt")
            x = torch.randn(1, 3, 28, 28).to(device)
            torch.onnx.export(quant_model, x, "quant_model_index.onnx")

            # 保存MNN模型压缩参数文件，必须要保存这个文件！
            # 如果进行量化的模型有剪枝，请将剪枝时生成的MNN模型压缩参数文件 "compress_params.bin" 文件名 在下方传入，并将 append 设置为True
            # append表示追加，如果此模型仅进行量化，append设为False即可！
            quantizer.save_compress_params(
                "quant_model_index.onnx", "compress_params_index.bin", append=False)
