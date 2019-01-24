import torch


def metric_accuracy(output, target):
    with torch.no_grad():
        output = output[0]
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def metric_accuracy_top_k(output, target, k=3):
    with torch.no_grad():
        output = output[0]
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def metric_accuracy_per_ring(output, target):
    with torch.no_grad():
        x = output[1] # batchsize x n_ring x n_classes
        x = torch.transpose(x, 1,0)
        accuracy_all = []
        for i in range(x.shape[0]):
            x_ring = x[i] # batchsize x n_classes
            accuracy_ring = metric_accuracy((x_ring,0),target)
            accuracy_all.append(accuracy_ring)

    return accuracy_all