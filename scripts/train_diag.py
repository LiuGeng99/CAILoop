import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import random
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from ema_pytorch import EMA
from utils.data_manager import DataManager

# DINOv2 路径配置
from cailoop.rotation import aggregate_rotation_logits, expand_classifier_state, rotate_batch
from cailoop.runtime import dinov2_backbone

def _set_random(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True # 注释掉以追求A100下的训练速度
    torch.backends.cudnn.benchmark = True

def setup_ddp():
    """初始化分布式训练环境"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def evaluate_model(model, classifier, loader, device, num_orig_classes):
    model.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _, inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                features = model(inputs)
                outputs = classifier(features)
            outputs = aggregate_rotation_logits(outputs, num_orig_classes)
            
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    metrics = torch.tensor([correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics[0].item() / metrics[1].item()

def train(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    _set_random(args.seed + local_rank) # 确保每张卡随机性有偏移

    if local_rank == 0:
        os.makedirs(args.ckpt_path, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[logging.FileHandler(f"{args.ckpt_path}/train.log"), logging.StreamHandler(sys.stdout)]
        )
        logging.info(f"Arguments: {vars(args)}")

    # 1. 数据准备
    data_manager = DataManager(
        dataset_name=args.dataset_name,
        shuffle=False,
        seed=args.seed,
        init_cls=args.num_classes,
        increment=1
    )
    
    train_dataset = data_manager.get_dataset(np.arange(args.num_classes), source="train", mode="train")
    test_dataset = data_manager.get_dataset(np.arange(args.num_classes), source="test", mode="test")

    # DDP 下使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        sampler=test_sampler, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # 2. 模型初始化
    dino_model = dinov2_backbone()(pretrained=False)
    dino_model.load_state_dict(torch.load(args.backbone_ckpt, map_location='cpu'))
    # Four rotations x the original classes. Saved cls_best.pth has this width.
    classifier = nn.Linear(768, args.num_classes * 4)
    
    if args.cls_ckpt and os.path.exists(args.cls_ckpt):
        state = torch.load(args.cls_ckpt, map_location='cpu')
        state = expand_classifier_state(state, args.num_classes)
        classifier.load_state_dict(state)

    dino_model.to(device)
    classifier.to(device)

    # 转换为 DDP 模型
    dino_model = DDP(dino_model, device_ids=[local_rank], find_unused_parameters=True)
    classifier = DDP(classifier, device_ids=[local_rank])

    # EMA (仅在主进程进行 EMA 更新或评估)
    ema_dino = EMA(dino_model.module, beta=0.9999, update_after_step=100, update_every=10)
    ema_cls = EMA(classifier.module, beta=0.9999, update_after_step=100, update_every=10)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() 

    log_steps = 10

    # --- 阶段一：Linear Probe ---
    if args.cls_epochs > 0:
        if local_rank == 0:
            logging.info("Starting Phase 1: Linear Probing...")
            
        cls_optimizer = optim.AdamW(classifier.parameters(), lr=args.cls_lr, weight_decay=args.weight_decay)
        dino_model.eval()
        
        for epoch in range(args.cls_epochs):
            train_sampler.set_epoch(epoch)
            classifier.train()
            
            for step, (_, inputs, targets, _) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = rotate_batch(inputs, targets, args.num_classes)
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.no_grad():
                        features = dino_model.module(inputs)
                    outputs = classifier(features)
                    loss = criterion(outputs, targets)

                cls_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(cls_optimizer)
                scaler.update()
                
                # 周期性日志流输出
                if local_rank == 0 and step % log_steps == 0:
                    logging.info(f"Linear Probe | Epoch [{epoch+1}/{args.cls_epochs}] | Step [{step}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    # --- 阶段二：Joint Fine-tuning ---
    if local_rank == 0:
        logging.info("Starting Phase 2: Joint Fine-tuning...")

    optimizer = optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, dino_model.parameters()), 'lr': args.dino_lr},
        {'params': classifier.parameters(), 'lr': args.cls_lr}
    ], weight_decay=args.weight_decay)

    best_acc = 0.0
    for epoch in range(args.ft_epochs):
        train_sampler.set_epoch(epoch)
        dino_model.train()
        classifier.train()
        
        for step, (_, inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = rotate_batch(inputs, targets, args.num_classes)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                features = dino_model(inputs)
                outputs = classifier(features)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ema_dino.update()
            ema_cls.update()
            
            # 周期性日志流输出
            if local_rank == 0 and step % log_steps == 0:
                logging.info(f"Fine-tuning | Epoch [{epoch+1}/{args.ft_epochs}] | Step [{step}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # 验证
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.ft_epochs - 1:
            # 验证时使用 EMA 模型
            cur_acc = evaluate_model(
                ema_dino.ema_model, ema_cls.ema_model, test_loader, device, args.num_classes
            )
            if local_rank == 0:
                logging.info(f"Epoch {epoch+1} | Test Acc (EMA): {cur_acc:.4f}")
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    torch.save(ema_dino.ema_model.state_dict(), f"{args.ckpt_path}/dino_best.pth")
                    torch.save(ema_cls.ema_model.state_dict(), f"{args.ckpt_path}/cls_best.pth")

    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINOv2 Fine-tuning with DDP on A100')
    
    # --- 路径配置 ---
    parser.add_argument('--backbone_ckpt', type=str, required=True,
                        help='DINOv2 ViT-B/14 register pretrain weights')
    parser.add_argument('--cls_ckpt', type=str, default=None)

    parser.add_argument('--ckpt_path', type=str, default='checkpoints/diag')
    parser.add_argument('--dataset_name', type=str, default='laryngo_holdout')
    
    # Manuscript settings: 30 classes, freeze the backbone for 5 epochs, then fine-tune.
    parser.add_argument('--num_classes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Images per GPU before the four rotations. The loss sees four times this many.') 
    
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--eval_freq', type=int, default=5)
    
    parser.add_argument('--cls_epochs', type=int, default=5, 
                        help='Epochs for classifier-only pre-training')
    parser.add_argument('--cls_lr', type=float, default=1e-5, 
                        help='Learning rate for the classifier head')
    
    parser.add_argument('--ft_epochs', type=int, default=95, 
                        help='Epochs for joint fine-tuning')
    parser.add_argument('--dino_lr', type=float, default=1e-5, 
                        help='Learning rate for DINOv2 backbone')

    # DDP Parameter
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help='Local rank for distributed training')

    args = parser.parse_args()
    
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    train(args)