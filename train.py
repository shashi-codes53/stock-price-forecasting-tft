"""
Training Script — Stock Price Forecasting with TFT

Usage:
    python train.py --ticker TCS.NS --epochs 50
    python train.py --ticker RELIANCE.NS --epochs 100 --batch_size 64
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.tft      import get_model
from data.dataset    import download_stock_data, get_dataloaders, FEATURE_COLS


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",   type=str,   default="TCS.NS")
    p.add_argument("--period",   type=str,   default="3y")
    p.add_argument("--seq_len",  type=int,   default=60)
    p.add_argument("--horizon",  type=int,   default=5)
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--patience", type=int,   default=10)
    return p.parse_args()


def mae(pred, target): return torch.mean(torch.abs(pred - target)).item()
def rmse(pred, target): return torch.sqrt(torch.mean((pred - target)**2)).item()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_mae = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds, _ = model(x)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_mae  += mae(preds.detach(), y.detach())
    return total_loss / len(loader), total_mae / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae, total_rmse = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds, _ = model(x)
        total_loss += criterion(preds, y).item()
        total_mae  += mae(preds, y)
        total_rmse += rmse(preds, y)
    n = len(loader)
    return total_loss/n, total_mae/n, total_rmse/n


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device}  |  Ticker: {args.ticker}")
    os.makedirs("outputs", exist_ok=True)

    # Data
    df = download_stock_data(args.ticker, args.period)
    train_loader, val_loader, test_loader, dataset = get_dataloaders(
        df, seq_len=args.seq_len, forecast_horizon=args.horizon,
        batch_size=args.batch
    )

    # Model
    model     = get_model(input_dim=len(FEATURE_COLS), forecast_horizon=args.horizon).to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    patience_cnt  = 0
    save_path     = f"outputs/{args.ticker.replace('.','_')}_best.pth"

    print(f"\n{'='*55}\n  Training TFT — {args.ticker}\n{'='*55}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_rmse = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:3d}/{args.epochs}]  "
              f"Train Loss: {train_loss:.4f}  MAE: {train_mae:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  MAE: {val_mae:.4f}  ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_loss": val_loss, "ticker": args.ticker,
                "seq_len": args.seq_len, "horizon": args.horizon,
                "input_dim": len(FEATURE_COLS),
            }, save_path)
            print(f"  ✅ Saved best model → {save_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break

    # Test
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    test_loss, test_mae, test_rmse = eval_epoch(model, test_loader, criterion, device)
    print(f"\n{'='*40}\n  Test Results\n{'='*40}")
    print(f"  MAE  : {test_mae:.4f}")
    print(f"  RMSE : {test_rmse:.4f}")
    print(f"\n✅ Done! Model saved at: {save_path}")


if __name__ == "__main__":
    main()
