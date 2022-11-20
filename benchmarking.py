import benchmarking_utils
from data_provider import val_loader
import torch
if __name__ == '__main__':
    model_path = "original_model/resnet56_model.pth"
    net = torch.load(model_path)
    print(f"net is from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmarking_utils.inference_benchmark(net, val_loader, device)
