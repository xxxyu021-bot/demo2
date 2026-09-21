import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import swanlab
from config import load_config
from dataset import NERDataset,build_label_map
from model import MineModel
from metrics import evaluate
import os
from utils import print_config, set_seed
import argparse

def main(args):
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    print_config(cfg)
    device = cfg["device"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"],use_fast=True)
    label2id, id2label = build_label_map(cfg["train_txt_path"])
    num_labels = len(label2id)
    print("Label2ID:", label2id)
    print("Number of labels:", num_labels)

    # 构建数据集
    train_dataset = NERDataset(
        txt_path=cfg["train_txt_path"],
        tokenizer=tokenizer,
        label2id=label2id,
        max_seq_len=cfg["max_seq_len"]
    )

    dev_dataset = NERDataset(
        txt_path=cfg["dev_txt_path"],
        tokenizer=tokenizer,
        label2id=label2id,
        max_seq_len=cfg["max_seq_len"]
    )

    test_dataset = NERDataset(
        txt_path=cfg["test_txt_path"],
        tokenizer=tokenizer,
        label2id=label2id,
        max_seq_len=cfg["max_seq_len"]
    )
    print(f"Train:{len(train_dataset)}  Dev:{len(dev_dataset)}  Test:{len(test_dataset)}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False
    )

    # 模型、损失、优化器
    model = MineModel(
        model_name=cfg["model_name"],
        dropout_rate=cfg["dropout_rate"],
        num_labels=num_labels
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    #新增学习率调度器
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_ratio = cfg["warmup_ratio"]
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    #修改流程，训练完成后选取最好的dev model进行test
    # best_val_acc = 0.0
    #改为以最佳f1为标准
    best_val_f1 = 0.0
    patience = cfg["early_stopping_patience"]
    patience_counter = 0
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Remote", "model", "best_model.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # swanlab
    swanlab.init(project=cfg["swanlab"]["project"])

    # 训练循环
    print("\nStarting training...")
    for epoch in range(cfg["num_epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_num = 0
        total_steps = len(train_loader)
        for batch_idx, batch in enumerate(train_loader, 1):

            input_ids = (batch["input_ids"].to(device))
            attention_mask = (batch["attention_mask"].to(device))
            token_type_ids = (batch["token_type_ids"].to(device))
            labels = (batch["labels"].to(device))
            logits = model(input_ids,attention_mask,token_type_ids)
            loss = criterion(logits.view( -1,num_labels ),labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            #当前 batch 实际有多少条样本
            current_batch_size = labels.size(0)
            train_loss_sum += loss.item() * current_batch_size
            train_num += current_batch_size

            if batch_idx % 20 == 0 or batch_idx == total_steps:
                step_avg_loss = train_loss_sum / train_num
                print(f"Epoch [{epoch+1}/{cfg['num_epochs']}] | Batch [{batch_idx}/{total_steps}] | Loss:{loss.item():.4f} | AvgLoss:{step_avg_loss:.4f}")

        train_avg_loss = train_loss_sum / train_num

        val_loss,val_acc,val_precision,val_recall,val_f1= evaluate(
            model,dev_loader,criterion,device,id2label)

        print(
            f"==== Epoch {epoch + 1} Done ==== Train Loss:{train_avg_loss:.4f} | Val Loss:{val_loss:.4f} | Val Acc:{val_acc:.4f} | Val Recall:{val_recall:.4f} | Val F1:{val_f1:.4f} | Val Precision:{val_precision:.4f}\n")

        swanlab.log({
            "train/loss": float(train_avg_loss),
            "val/loss": float(val_loss),
            "val/acc": float(val_acc),
            "val/recall": float(val_recall),
            "val/f1": float(val_f1),
            "val/precision": float(val_precision),
            "lr": float(scheduler.get_last_lr()[0])
        })

        #保存最优结果

        #改为以最佳f1为标准
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # torch.save(model.state_dict(), best_model_path)
            #新增保存训练轮次，AdamW状态等内容
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_f1": best_val_f1,
                "patience_counter": patience_counter,
                "config": cfg,
                "label2id": label2id,
                "id2label": id2label
            }, best_model_path)
            print(f"Best F1: {best_val_f1:.4f}\n")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}\n")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
    # 测试集评估
    # model.load_state_dict(torch.load(best_model_path, map_location=device))
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Run test set evaluation ...")

    test_loss,test_acc,test_precision,test_recall,test_f1= evaluate(
        model,test_loader,criterion,device,id2label)
    print(
        f"======== Test Result (Best Dev Model) ======== Test Loss:{test_loss:.4f} | Test Acc:{test_acc:.4f} | Test Recall:{test_recall:.4f} | Test F1:{test_f1:.4f} | Test Precision:{test_precision:.4f}")
    swanlab.log({
        "test/loss": float(test_loss),
        "test/acc": float(test_acc),
        "test/recall": float(test_recall),
        "test/f1": float(test_f1),
        "test/precision": float(test_precision)
    })

    swanlab.finish()
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=[
            "config/msra_bert.json",
            "config/msra_wwm.json",
            "config/weibo_bert.json",
            "config/weibo_wwm.json"
        ],
        required=True
    )
    args = parser.parse_args()

    main(args)
