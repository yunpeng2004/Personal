import os
import torch
from config import TrainConfig
from loaders import create_dataloaders, load_datasets
from losses import SegmentationCriterion
from model import WKSPBRegionKANNet
from train_engine import evaluate, evaluate_with_tta, run_one_epoch, search_best_threshold
from utils import ensure_dir, set_seed


def build_model(cfg):
    return WKSPBRegionKANNet(
        use_pretrained_backbone=getattr(cfg, "use_pretrained_backbone", True),
        hidden_channels=getattr(cfg, "hidden_channels", 128),
        state_channels=getattr(cfg, "state_channels", 16),
        kan_groups=getattr(cfg, "kan_groups", 4),
        kan_bases=getattr(cfg, "kan_bases", 6),
    )


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset = load_datasets(
        img_size=cfg.image_size,
        seed=cfg.seed,
        use_train_augmentation=cfg.use_train_augmentation,
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg.batch_size,
        train_dataset,
        val_dataset,
        test_dataset,
        num_workers=cfg.num_workers,
    )

    model = build_model(cfg).to(device)

    criterion = SegmentationCriterion(
        aux_region_weight=cfg.aux_region_weight,
        aux_boundary_weight=cfg.aux_boundary_weight,
        aux_confidence_weight=cfg.aux_confidence_weight,
        boundary_emphasis=cfg.boundary_emphasis,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.use_amp and device.startswith("cuda")),
    )

    best_dice = -1.0
    best_epoch = 0
    no_improve_count = 0
    best_path = os.path.join(cfg.save_dir, "wkspb_v2_best.pt")
    min_delta = 1e-4

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\nEpoch [{epoch}/{cfg.num_epochs}]")

        train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            threshold=0.5,
            grad_clip_norm=cfg.grad_clip_norm,
        )

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=0.5,
        )

        scheduler.step(val_stats["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"train loss={train_stats['loss']:.4f} dice={train_stats['dice']:.4f} | "
            f"val loss={val_stats['loss']:.4f} dice={val_stats['dice']:.4f} | "
            f"lr={current_lr:.6e}"
        )

        if val_stats["dice"] > best_dice + min_delta:
            best_dice = val_stats["dice"]
            best_epoch = epoch
            no_improve_count = 0

            torch.save(
                {
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                best_path,
            )
            print(f"Saved best model at epoch {epoch} (val dice = {best_dice:.4f})")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epoch(s).")

        if no_improve_count >= cfg.early_stop_patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch}. "
                f"Best epoch: {best_epoch}, best val dice: {best_dice:.4f}"
            )
            break

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(
            f"\nLoaded best checkpoint from epoch {checkpoint['epoch']} "
            f"with val dice = {checkpoint['best_dice']:.4f}"
        )

    coarse_thresholds = [round(x, 2) for x in torch.arange(0.25, 0.61, 0.01).tolist()]
    best_thr, val_thr_stats = search_best_threshold(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        thresholds=coarse_thresholds,
        use_tta=False,
    )

    print(
        f"\nBest validation threshold (no TTA) = {best_thr:.3f} | "
        f"val dice={val_thr_stats['dice']:.4f} "
        f"iou={val_thr_stats['iou']:.4f}"
    )

    best_thr_tta, val_thr_stats_tta = search_best_threshold(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        thresholds=coarse_thresholds,
        use_tta=True,
    )

    print(
        f"\nBest validation threshold (with TTA) = {best_thr_tta:.3f} | "
        f"val dice={val_thr_stats_tta['dice']:.4f} "
        f"iou={val_thr_stats_tta['iou']:.4f}"
    )

    test_stats = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=best_thr,
    )

    print(
        f"\nTest (no TTA) | threshold={best_thr:.3f} "
        f"loss={test_stats['loss']:.4f} "
        f"dice={test_stats['dice']:.4f} "
        f"iou={test_stats['iou']:.4f} "
        f"precision={test_stats['precision']:.4f} "
        f"accuracy={test_stats['accuracy']:.4f}"
    )

    test_stats_tta = evaluate_with_tta(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=best_thr_tta,
    )

    print(
        f"\nTest (with TTA) | threshold={best_thr_tta:.3f} "
        f"loss={test_stats_tta['loss']:.4f} "
        f"dice={test_stats_tta['dice']:.4f} "
        f"iou={test_stats_tta['iou']:.4f} "
        f"precision={test_stats_tta['precision']:.4f} "
        f"accuracy={test_stats_tta['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
