#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAN增强孪生网络训练脚本
实现生成-度量联合训练框架
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
from models.implementations.gan_siamese_model import GANSiameseModel
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


def train_epoch(model, train_loader, siamese_loader, optimizers, device, 
                lambda_siamese=0.5, lambda_cls=1.0, lambda_gan=0.1, lambda_hard=0.2):
    """训练一个epoch"""
    model.train()
    
    optimizer_main, optimizer_g, optimizer_d = optimizers
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_siamese_loss = 0.0
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_hard_loss = 0.0
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
        
        # === 训练判别器 ===
        optimizer_d.zero_grad()
        
        # GAN联合前向传播
        outputs = model.forward_gan_joint(anchor, pair, same_family, use_hard_negatives=True)
        
        # 判别器损失
        d_loss = outputs['discriminator_loss']
        d_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
        optimizer_d.step()
        
        # === 训练生成器和主网络 ===
        optimizer_main.zero_grad()
        optimizer_g.zero_grad()
        
        # 重新前向传播（因为计算图被清除）
        outputs = model.forward_gan_joint(anchor, pair, same_family, use_hard_negatives=True)
        
        # 主分类损失
        cls_loss = nn.CrossEntropyLoss()(outputs['logits'], target[:len(outputs['logits'])])
        
        # 孪生损失
        siamese_loss = outputs['siamese_loss']
        
        # 生成器损失
        g_loss = outputs['generator_loss']
        
        # 难负例损失
        hard_loss = outputs['hard_negative_loss']
        
        # 总损失
        total_batch_loss = (
            lambda_cls * cls_loss + 
            lambda_siamese * siamese_loss + 
            lambda_gan * g_loss +
            lambda_hard * hard_loss
        )
        
        total_batch_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer_main.step()
        optimizer_g.step()
        
        # 统计
        total_loss += total_batch_loss.item()
        total_cls_loss += cls_loss.item()
        total_siamese_loss += siamese_loss.item()
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        if isinstance(hard_loss, torch.Tensor):
            total_hard_loss += hard_loss.item()
        
        # 计算准确率
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == target[:len(pred)]).sum().item()
        total += len(pred)
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{total_batch_loss.item():.4f}',
            'Cls': f'{cls_loss.item():.4f}',
            'Sim': f'{siamese_loss.item():.4f}',
            'G': f'{g_loss.item():.4f}',
            'D': f'{d_loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_siamese_loss = total_siamese_loss / len(train_loader)
    avg_g_loss = total_g_loss / len(train_loader)
    avg_d_loss = total_d_loss / len(train_loader)
    avg_hard_loss = total_hard_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'siamese_loss': avg_siamese_loss,
        'g_loss': avg_g_loss,
        'd_loss': avg_d_loss,
        'hard_loss': avg_hard_loss,
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


def few_shot_evaluation(model, test_loader, device, k_shot=5, n_way=5, num_episodes=100):
    """小样本学习评估"""
    model.eval()
    
    episode_accuracies = []
    
    # 获取所有测试数据
    all_data = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            all_data.append(data)
            all_labels.append(labels)
    
    all_data = torch.cat(all_data, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    
    # 按类别组织数据
    unique_labels = torch.unique(all_labels)
    class_data = {}
    for label in unique_labels:
        mask = all_labels == label
        class_data[label.item()] = all_data[mask]
    
    with torch.no_grad():
        for episode in range(num_episodes):
            # 随机选择n_way个类别
            selected_classes = np.random.choice(len(unique_labels), n_way, replace=False)
            
            support_data = []
            support_labels = []
            query_data = []
            query_labels = []
            
            for i, class_idx in enumerate(selected_classes):
                class_label = unique_labels[class_idx]
                class_samples = class_data[class_label.item()]
                
                # 随机选择k_shot + 1个样本（k_shot用于支持集，1个用于查询集）
                if len(class_samples) >= k_shot + 1:
                    indices = torch.randperm(len(class_samples))[:k_shot + 1]
                    
                    support_data.append(class_samples[indices[:k_shot]])
                    support_labels.extend([i] * k_shot)
                    
                    query_data.append(class_samples[indices[k_shot:]])
                    query_labels.append(i)
            
            if len(support_data) == n_way and len(query_data) == n_way:
                support_data = torch.cat(support_data, dim=0)
                support_labels = torch.tensor(support_labels, device=device)
                query_data = torch.cat(query_data, dim=0)
                query_labels = torch.tensor(query_labels, device=device)
                
                # 获取支持集和查询集的嵌入
                support_embeddings = model.get_embedding(support_data)
                query_embeddings = model.get_embedding(query_data)
                
                # 计算原型（每个类别的平均嵌入）
                prototypes = []
                for i in range(n_way):
                    class_mask = support_labels == i
                    if class_mask.sum() > 0:
                        prototype = support_embeddings[class_mask].mean(dim=0)
                        prototypes.append(prototype)
                
                if len(prototypes) == n_way:
                    prototypes = torch.stack(prototypes)
                    
                    # 计算查询样本与原型的距离
                    distances = torch.cdist(query_embeddings, prototypes)
                    predictions = distances.argmin(dim=1)
                    
                    # 计算准确率
                    accuracy = (predictions == query_labels).float().mean().item()
                    episode_accuracies.append(accuracy)
    
    if episode_accuracies:
        mean_accuracy = np.mean(episode_accuracies)
        std_accuracy = np.std(episode_accuracies)
        return mean_accuracy, std_accuracy
    else:
        return 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description='GAN增强孪生网络训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--lambda_siamese', type=float, default=0.5, help='孪生损失权重')
    parser.add_argument('--lambda_cls', type=float, default=1.0, help='分类损失权重')
    parser.add_argument('--lambda_gan', type=float, default=0.1, help='GAN损失权重')
    parser.add_argument('--lambda_hard', type=float, default=0.2, help='难负例损失权重')
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
    log_file = config['logging']['file'].replace('.log', '_gan_siamese.log')
    logger = setup_logging(log_file)
    logger.info("GAN增强孪生网络训练开始")
    
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
        num_workers=0
    )
    
    logger.info(f"数据集加载完成: {dataset_info}")
    
    # 创建模型
    model = GANSiameseModel(
        vocab_size=dataset_info['vocab_size'],
        d_model=128,
        num_layers=2,
        num_classes=dataset_info['num_classes'],
        dropout=0.1,
        siamese_emb_dim=128,
        latent_dim=100
    ).to(device)
    
    logger.info(f"模型创建完成，参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 优化器
    optimizer_main = optim.Adam(
        list(model.embedding.parameters()) + 
        list(model.feature_extractor.parameters()) +
        list(model.classifier.parameters()) +
        list(model.siamese_head.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    optimizer_g = optim.Adam(
        model.generator.parameters(),
        lr=config['training']['learning_rate'] * 0.5,  # 生成器学习率稍低
        weight_decay=config['training']['weight_decay']
    )
    
    optimizer_d = optim.Adam(
        model.discriminator.parameters(),
        lr=config['training']['learning_rate'] * 2,  # 判别器学习率稍高
        weight_decay=config['training']['weight_decay']
    )
    
    optimizers = (optimizer_main, optimizer_g, optimizer_d)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_main, mode='max', factor=0.5, patience=3
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = config['training']['patience']
    
    for epoch in range(config['training']['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, siamese_train_loader, optimizers, device,
            lambda_siamese=args.lambda_siamese,
            lambda_cls=args.lambda_cls,
            lambda_gan=args.lambda_gan,
            lambda_hard=args.lambda_hard
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
            f"G: {train_metrics['g_loss']:.4f}, "
            f"D: {train_metrics['d_loss']:.4f}, "
            f"Hard: {train_metrics['hard_loss']:.4f}, "
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
                'gan_siamese_best.pth'
            )
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_main_state_dict': optimizer_main.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
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
    
    # 小样本学习评估
    if dataset_info['num_classes'] >= 5:
        logger.info("\n开始小样本学习评估...")
        few_shot_acc, few_shot_std = few_shot_evaluation(
            model, test_loader, device, k_shot=5, n_way=5, num_episodes=100
        )
        logger.info(
            f"5-way 5-shot - Acc: {few_shot_acc*100:.2f}% ± {few_shot_std*100:.2f}%"
        )
    
    logger.info("GAN增强孪生网络训练完成")


if __name__ == '__main__':
    main()