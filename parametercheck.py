import benchmarking_utils
# from data_provider import get_test_loader
import torch

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_prunning_parameter_number(net):
    total_sum = 0
    for module_name, module in net.named_modules():
        if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
            # print (module_name," : ",module)
            for n, p in module.named_parameters():
                if 'weight' == n:
                    total_sum += p.numel()
    return total_sum


if __name__ == '__main__':
    model_path = "saved_model/original_model/resnet56_model.pth"
    net = torch.load(model_path)
    for name, module in net._modules.items():
        print (name," : ",module)
    print ("**********")
    for module_name, module in net.named_modules():
        
        if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
            print (module_name," : ",module)
            for n, p in module.named_parameters():
                if 'weight' == n:
                    pname = module_name + "." + n
                    # print(n)
                    if p.grad is None:
                        continue
    print(get_parameter_number(net))
    print(get_prunning_parameter_number(net))
    # val_loader=get_test_loader(256,16)
    # print(f"net is from {model_path}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # benchmarking_utils.inference_benchmark(net, val_loader, device)
