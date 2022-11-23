import benchmarking_utils
from data_provider import get_test_loader, get_train_loader
import torch
if __name__ == '__main__':
    model_path = "saved_model/original_model/resnet56_model.pth"
    net = torch.load(model_path)
    print(f"net is from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_test_loader(128, 16)
    benchmarking_utils.inference_benchmark(net, test_loader, device)
