import math
import torch
import pickle
import numpy as np
from sklearn.metrics import classification_report

comp_pkl = open('./ckpt/p_10_e21_test_results.pkl', 'rb')

score_comp = pickle.load(comp_pkl)

output_1 = torch.from_numpy(score_comp[0])

video_label = torch.from_numpy(score_comp[1])
target_names = [str(i) for i in range(174)]
output = output_1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return pred, res


pred, res = accuracy(output.cpu(), video_label, topk=(1, 5))
print('acc1:', res[0], 'acc5:', res[1])
cc = classification_report(video_label, pred[0], target_names=target_names, digits=3)
# print(cc)
