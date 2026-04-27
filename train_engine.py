import torch
from tqdm import tqdm


def _get_main_logits(outputs):
    return outputs["logits"] if isinstance(outputs, dict) else outputs


def _main_loss_only(criterion, logits, masks):
    fake_outputs = {"logits": logits}
    loss, _ = criterion(fake_outputs, masks)
    return loss


def _update_binary_stats(logits, masks, threshold, totals, epsilon=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    totals["tp"] += float((preds * masks).sum().item())
    totals["fp"] += float((preds * (1.0 - masks)).sum().item())
    totals["fn"] += float(((1.0 - preds) * masks).sum().item())
    totals["tn"] += float(((1.0 - preds) * (1.0 - masks)).sum().item())
    totals["eps"] = epsilon


def _finalize_binary_stats(totals):
    tp = totals["tp"]
    fp = totals["fp"]
    fn = totals["fn"]
    tn = totals["tn"]
    eps = totals["eps"]

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "accuracy": accuracy,
    }


def run_one_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None,
    device="cuda",
    threshold=0.5,
    grad_clip_norm=None,
):
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    count = 0
    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "eps": 1e-7}

    pbar = tqdm(loader, total=len(loader))
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None and device.startswith("cuda"))):
            outputs = model(images)
            logits = _get_main_logits(outputs)
            loss, _ = criterion(outputs, masks)

        if is_train:
            if scaler is not None and device.startswith("cuda"):
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        _update_binary_stats(logits.detach(), masks.detach(), threshold=threshold, totals=totals)

        running_loss += float(loss.detach().item())
        count += 1
        stat_now = _finalize_binary_stats(totals)
        pbar.set_description(
            f"{'train' if is_train else 'valid'} "
            f"loss:{running_loss / count:.4f} "
            f"dice:{stat_now['dice']:.4f} "
            f"iou:{stat_now['iou']:.4f}"
        )

    final_stats = _finalize_binary_stats(totals)
    final_stats["loss"] = running_loss / max(count, 1)
    return final_stats


@torch.no_grad()
def evaluate(model, loader, criterion, device="cuda", threshold=0.5):
    return run_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=None,
        scaler=None,
        device=device,
        threshold=threshold,
        grad_clip_norm=None,
    )


@torch.no_grad()
def predict_with_tta(model, images):
    logits_list = []

    out = model(images)
    logits_list.append(_get_main_logits(out))

    img_h = torch.flip(images, dims=[3])
    out_h = model(img_h)
    logits_list.append(torch.flip(_get_main_logits(out_h), dims=[3]))

    img_v = torch.flip(images, dims=[2])
    out_v = model(img_v)
    logits_list.append(torch.flip(_get_main_logits(out_v), dims=[2]))

    img_hv = torch.flip(images, dims=[2, 3])
    out_hv = model(img_hv)
    logits_list.append(torch.flip(_get_main_logits(out_hv), dims=[2, 3]))

    mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return mean_logits


@torch.no_grad()
def evaluate_with_tta(model, loader, criterion, device="cuda", threshold=0.5):
    model.eval()

    running_loss = 0.0
    count = 0
    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "eps": 1e-7}

    pbar = tqdm(loader, total=len(loader))
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = predict_with_tta(model, images)
        loss = _main_loss_only(criterion, logits, masks)

        _update_binary_stats(logits, masks, threshold=threshold, totals=totals)
        running_loss += float(loss.detach().item())
        count += 1

        stat_now = _finalize_binary_stats(totals)
        pbar.set_description(
            f"tta valid "
            f"loss:{running_loss / count:.4f} "
            f"dice:{stat_now['dice']:.4f} "
            f"iou:{stat_now['iou']:.4f}"
        )

    final_stats = _finalize_binary_stats(totals)
    final_stats["loss"] = running_loss / max(count, 1)
    return final_stats


@torch.no_grad()
def search_best_threshold(model, loader, criterion, device="cuda", thresholds=None, use_tta=False):
    if thresholds is None:
        thresholds = [round(x, 2) for x in torch.arange(0.25, 0.61, 0.01).tolist()]

    best_thr = thresholds[0]
    best_stats = None
    best_dice = -1.0

    for thr in thresholds:
        if use_tta:
            stats = evaluate_with_tta(model, loader, criterion, device=device, threshold=thr)
        else:
            stats = evaluate(model, loader, criterion, device=device, threshold=thr)

        if stats["dice"] > best_dice:
            best_dice = stats["dice"]
            best_thr = thr
            best_stats = stats

    if len(thresholds) >= 2:
        fine_left = max(0.10, best_thr - 0.03)
        fine_right = min(0.90, best_thr + 0.03)
        fine_thresholds = [round(x, 3) for x in torch.arange(fine_left, fine_right + 1e-8, 0.005).tolist()]
        for thr in fine_thresholds:
            if use_tta:
                stats = evaluate_with_tta(model, loader, criterion, device=device, threshold=thr)
            else:
                stats = evaluate(model, loader, criterion, device=device, threshold=thr)

            if stats["dice"] > best_dice:
                best_dice = stats["dice"]
                best_thr = thr
                best_stats = stats

    return best_thr, best_stats
