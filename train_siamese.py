#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
孪生网络增强训练脚本
联合训练主分类任务和家族相似度学习任务
"""

import os
import sys
import time
import logging
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config_manager import ConfigManager
from core.dataset import load_dataset, create_data_loaders
from models.implementations.siamese_moe_model import SiameseMoEModel
from models.implementations.siamese_addon import create_siamese_dataloader


def setup_logging(log_file):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(model, train_loader, siamese_loader, optimizer, device, 
                lambda_siamese=0.5, lambda_cls=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_siamese_loss = 0.0

    correct = 0
    total = 0
    
    # 创建数据迭代器
    siamese_iter = iter(siamese_loader)
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        # 获取孪生网络数据
        try:
            siamese_batch = next(siamese_iter)
        except StopIteration:
            siamese_iter = iter(siamese_loader)
            siamese_batch = next(siamese_iter)
        
        anchor = siamese_batch['anchor'].to(device)
        pair = siamese_batch['pair'].to(device)
        same_family = siamese_batch['same_family'].to(device).float()
        
        optimizer.zero_grad()
        
        # 联合前向传播
        outputs = model.forward_joint(anchor, pair, same_family)
        
        # 主分类损失
        cls_loss = nn.CrossEntropyLoss()(outputs['logits'], target[:len(outputs['logits'])])
        
        # 孪生损失
        siamese_loss = outputs['siamese_loss']
        
        # 总损失
        total_batch_loss = (
            lambda_cls * cls_loss + 
            lambda_siamese * siamese_loss
        )
        
        total_batch_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += total_batch_loss.item()
        total_cls_loss += cls_loss.item()
        total_siamese_loss += siamese_loss.item()
        
        # 计算准确率
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == target[:len(pred)]).sum().item()
        total += len(pred)
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{total_batch_loss.item():.4f}',
            'Cls': f'{cls_loss.item():.4f}',
            'Sim': f'{siamese_loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_siamese_loss = total_siamese_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'siamese_loss': avg_siamese_loss,
        'accuracy': accuracy
    }


def validate_epoch(model, val_loader, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            
            # 只进行主分类
            logits = model(data)
            loss = nn.CrossEntropyLoss()(logits, target)
            
            total_loss += loss.item()
            
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy * 100,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description='孪生网络增强的DGA检测训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--lambda_siamese', type=float, default=0.5, help='孪生损失权重')
    parser.add_argument('--lambda_cls', type=float, default=1.0, help='分类损失权重')
    parser.add_argument('--positive_ratio', type=float, default=0.6, help='正样本对比例')
    
    args = parser.parse_args()
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    
    # 设置日志
    log_file = config['logging']['file'].replace('.log', '_siamese.log')
    logger = setup_logging(log_file)
    logger.info("孪生网络增强训练开始")
    
    # 加载数据
    logger.info(f"加载数据集: {config['dataset']['type']}")
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        dataset_path=config['dataset']['path'],
        batch_size=config['dataset']['batch_size']
    )
    
    # 创建孪生网络数据加载器
    train_dataset = train_loader.dataset
    siamese_train_loader = create_siamese_dataloader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        positive_ratio=args.positive_ratio,
        shuffle=True,
        num_workers=config['dataset']['num_workers']
    )
    
    logger.info(f"数据集加载完成: {dataset_info}")
    
    # 创建模型
    model = SiameseMoEModel(
        vocab_size=dataset_info['vocab_size'],
        d_model=128,
        num_layers=2,
        num_experts=4,
        top_k=2,
        num_classes=dataset_info['num_classes'],
        dropout=0.1,
        siamese_emb_dim=128
    ).to(device)
    
    logger.info(f"模型创建完成，参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 优化器和调度器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = config['training']['patience']
    
    for epoch in range(config['training']['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, siamese_train_loader, optimizer, device,
            lambda_siamese=args.lambda_siamese,
            lambda_cls=args.lambda_cls
        )
        
        # 验证
        val_metrics = validate_epoch(model, val_loader, device)
        
        # 调度器步进
        scheduler.step(val_metrics['accuracy'])
        
        # 日志记录
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Cls: {train_metrics['cls_loss']:.4f}, "
            f"Sim: {train_metrics['siamese_loss']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.2f}%, "
            f"F1: {val_metrics['f1']:.4f}"
        )
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            # 保存模型
            model_save_path = os.path.join(
                config['output']['model_save_dir'],
                'siamese_moe_best.pth'
            )
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc
            }, model_save_path)
            
            logger.info(f"保存最佳模型: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= max_patience:
            logger.info(f"早停触发，最佳验证准确率: {best_val_acc:.2f}%")
            break
    
    # 测试
    logger.info("\n开始测试...")
    test_metrics = validate_epoch(model, test_loader, device)
    logger.info(
        f"Test - Acc: {test_metrics['accuracy']:.2f}%, "
        f"F1: {test_metrics['f1']:.4f}"
    )
    
    logger.info("孪生网络增强训练完成")


if __name__ == '__main__':
    main()