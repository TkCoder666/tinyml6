import typing
import torch
import os
import time


import torch.distributed as dist


def _all_reduce(*args, reduction="sum"):
    t = torch.tensor(args, dtype=torch.float).cuda()
    dist.all_reduce(t, op=_str_2_reduceop[reduction])
    rev = t.tolist()
    if reduction == "mean":
        world_size = dist.get_world_size()
        rev = [item/world_size for item in rev]
    return rev


_str_2_reduceop = dict(
    sum=dist.ReduceOp.SUM,
    mean=dist.ReduceOp.SUM,
    product=dist.ReduceOp.PRODUCT,
    min=dist.ReduceOp.MIN,
    max=dist.ReduceOp.MAX,
)


class Accuracy(object):
    def __init__(self):
        self._is_distributed = dist.is_available() and dist.is_initialized()
        self.reset()

    def reset(self):
        self._n_correct = 0.0
        self._n_total = 0.0
        self._reset_buffer()

    @property
    def rate(self):
        self.sync()
        return self._n_correct / (self._n_total+1e-8)

    @property
    def n_correct(self):
        self.sync()
        return self._n_correct

    @property
    def n_total(self):
        self.sync()
        return self._n_total

    def _reset_buffer(self):
        self._n_correct_since_last_sync = 0.0
        self._n_total_since_last_sync = 0.0
        self._is_synced = True

    def update(self,  n_correct, n_total):
        self._n_correct_since_last_sync += n_correct
        self._n_total_since_last_sync += n_total
        self._is_synced = False

    def sync(self):
        if self._is_synced:
            return
        n_correct = self._n_correct_since_last_sync
        n_total = self._n_total_since_last_sync
        if self._is_distributed:
            n_correct, n_total = _all_reduce(
                n_correct, n_total, reduction="sum")

        self._n_correct += n_correct
        self._n_total += n_total

        self._reset_buffer()


class AccuracyMetric(object):
    def __init__(self, topk: typing.Iterable[int] = (1,),):
        self.topk = sorted(list(topk))
        self.reset()

    def reset(self) -> None:
        self.accuracies = [Accuracy() for _ in self.topk]

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        maxk = max(self.topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))

        for accuracy, k in zip(self.accuracies, self.topk):
            correct_k = correct[:k].sum().item()
            accuracy.update(correct_k, batch_size)

    def at(self, topk: int) -> Accuracy:
        if topk not in self.topk:
            raise ValueError(
                f"topk={topk} is not in registered topks={self.topk}")
        accuracy = self.accuracies[self.topk.index(topk)]
        accuracy.sync()
        return accuracy

    def __str__(self):
        items = [f"top{k}-acc={self.at(k).rate*100:.2f}%" for k in self.topk]
        return ", ".join(items)


def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


import time
def inference_benchmark(net, testloader,device):
    correct = 0
    total = 0
    t1 = time.time()
    net = net.to(device)
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            accuracy_metric.update(outputs, labels)
    t2 = time.time()
    print(f"time use is {t2 - t1}")
    print(
        # f'Accuracy of the network on the 10000 test images: {100 * correct // total} %',
        f"top1_acc {accuracy_metric.at(1).rate} ",
        f"top5_acc {accuracy_metric.at(5).rate}")
