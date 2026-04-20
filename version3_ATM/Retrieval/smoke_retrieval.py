import argparse
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ATMS_retrieval import ATMS, evaluate_model, train_model
from eegdatasets_leaveone import EEGDataset


def main():
    parser = argparse.ArgumentParser(description="Smoke test for ATM retrieval on a small class subset")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    classes = list(range(args.num_classes))

    train_ds = EEGDataset(args.data_path, subjects=["sub-01"], train=True, classes=classes)
    test_ds = EEGDataset(args.data_path, subjects=["sub-01"], train=False, classes=classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    model = ATMS().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    config = SimpleNamespace()

    train_loss, train_acc, _ = train_model(
        "sub-01",
        model,
        train_loader,
        optimizer,
        device,
        train_ds.text_features,
        train_ds.img_features,
        config,
    )
    eval_loss, eval_acc, top5 = evaluate_model(
        "sub-01",
        model,
        test_loader,
        device,
        test_ds.text_features,
        test_ds.img_features,
        k=args.num_classes,
        config=config,
    )

    print("Smoke retrieval completed")
    print(f"Train loss: {train_loss:.6f}")
    print(f"Train acc: {train_acc:.6f}")
    print(f"Eval loss: {eval_loss:.6f}")
    print(f"Eval acc: {eval_acc:.6f}")
    print(f"Top-5 acc: {top5:.6f}")


if __name__ == "__main__":
    main()
