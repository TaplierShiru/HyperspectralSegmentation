import torch.nn as nn


bce_loss = nn.CrossEntropyLoss()

def muti_bce_loss_fusion(labels_v, *preds):
	loss = 0.0
	for p in preds:
		loss = loss + bce_loss(p, labels_v)

	return loss