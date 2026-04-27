import torch


def _binarize_from_logits(y_pred, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    return (y_pred > threshold).float()


def dice_coefficient(y_pred, y_true, threshold=0.5, epsilon=1e-7):
    y_pred = _binarize_from_logits(y_pred, threshold)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    return (2.0 * intersection + epsilon) / (union + epsilon)


def iou_score(y_pred, y_true, threshold=0.5, epsilon=1e-6):
    y_pred = _binarize_from_logits(y_pred, threshold)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + epsilon) / (union + epsilon)


def precision_score(y_pred, y_true, threshold=0.5, epsilon=1e-6):
    y_pred = _binarize_from_logits(y_pred, threshold)
    true_positive = (y_pred * y_true).sum()
    false_positive = (y_pred * (1 - y_true)).sum()
    return (true_positive + epsilon) / (true_positive + false_positive + epsilon)


def accuracy_score(y_pred, y_true, threshold=0.5):
    y_pred = _binarize_from_logits(y_pred, threshold)
    correct = (y_pred == y_true).float().sum()
    return correct / y_true.numel()
