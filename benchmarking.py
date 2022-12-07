import benchmarking_utils
from data_provider import get_test_loader
import torch
if __name__ == '__main__':
    model_path = "saved_model/original_model/resnet56_model.pth"
    net = torch.load(model_path)

    # for name, module in net._modules.items():
    #     print (name," : ",module)
    # print ("**********")
    val_loader=get_test_loader(256,16)
    print(f"net is from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmarking_utils.inference_benchmark(net, val_loader, device)
