"""
EDA-ResNet50 评估脚本

完整的皮肤癌分类模型评估工具，包含论文所需的所有指标：
- 准确率 (Accuracy)
- 敏感性 (Sensitivity/Recall)
- 特异性 (Specificity)
- 精确率 (Precision)
- F1分数
- 混淆矩阵
- ROC曲线和AUC
- PR曲线
- 分类报告

Usage:
    python evaluate.py --model-path path/to/model.h5 --data-dir path/to/data
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize

# Add src to path for both package and direct invocation
SRC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for path in {SRC_ROOT, PROJECT_ROOT}:
    if path not in sys.path:
        sys.path.append(path)

from data.dataset import SkinCancerDataset
from models.eda_resnet50 import (
    create_eda_resnet50,
    compile_eda_resnet50,
    EDAResNet50,
    EDAResNet50Alternative
)
from models.mfr_module import MFRModule, Swish as MFRSwish
from models.efficient_module import EfficientModule, EfficientModuleSimplified
from models.da_module import DualAttentionModule, Swish as DASwish
from models.backbone import ResNet50FeatureExtractor, ResNet50Backbone
from training.metrics import ArgmaxRecallMetric
CUSTOM_OBJECTS = {
    'EDAResNet50': EDAResNet50,
    'EDAResNet50Alternative': EDAResNet50Alternative,
    'MFRModule': MFRModule,
    'MFRSwish': MFRSwish,
    'EfficientModule': EfficientModule,
    'EfficientModuleSimplified': EfficientModuleSimplified,
    'DualAttentionModule': DualAttentionModule,
    'DASwish': DASwish,
    'ResNet50FeatureExtractor': ResNet50FeatureExtractor,
    'ResNet50Backbone': ResNet50Backbone,
    'ArgmaxRecallMetric': ArgmaxRecallMetric
}

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EDAEvaluator:
    """EDA-ResNet50 模型评估器"""

    def __init__(self, model_path: str, data_dir: str, output_dir: str = None):
        """
        初始化评估器

        Args:
            model_path: 训练好的模型路径
            data_dir: 数据集目录
            output_dir: 评估结果输出目录
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(os.path.dirname(model_path), 'evaluation')

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 论文目标指标
        self.paper_targets = {
            'accuracy': 0.9318,
            'sensitivity': 0.94,
            'specificity': 0.925
        }

        # 类别信息
        self.class_names = ['benign', 'malignant']
        self.class_mapping = {'benign': 0, 'malignant': 1}

        # 初始化数据和模型
        self._setup_data()
        self._load_model()

        print(f"EDA-ResNet50 评估器初始化完成")
        print(f"模型路径: {model_path}")
        print(f"数据路径: {data_dir}")
        print(f"输出路径: {self.output_dir}")

    def _setup_data(self):
        """设置数据加载器"""
        print("设置数据加载器...")

        # 创建数据集实例
        self.dataset = SkinCancerDataset(
            data_dir=self.data_dir,
            image_size=(224, 224),
            batch_size=32,  # 评估时可以用较大的batch size
            shuffle=False
        )

        # 创建测试数据生成器
        self.test_generator = self.dataset.create_test_generator()

        print(f"测试数据集: {self.test_generator.samples} 样本")
        print(f"批次数量: {len(self.test_generator)}")

    def _load_model(self):
        """加载训练好的模型"""
        print(f"加载模型: {self.model_path}")

        full_model_candidate = (
            os.path.isdir(self.model_path) or
            self.model_path.endswith('.keras')
        )

        if full_model_candidate:
            try:
                # 方式1: 直接加载完整模型
                self.model = load_model(
                    self.model_path,
                    custom_objects=CUSTOM_OBJECTS,
                    compile=False
                )
                compile_eda_resnet50(self.model)
                print("✓ 模型加载成功 (完整模型)")
                return
            except Exception as e:
                print(f"完整模型加载失败: {e}")

        # 方式2: 重新构建模型并加载权重
        try:
            print("尝试重新构建模型并加载权重...")
            self.model = create_eda_resnet50(num_classes=2)
            compile_eda_resnet50(self.model)
            self.model.load_weights(self.model_path)
            print("✓ 模型构建和权重加载成功")
        except Exception as e2:
            print(f"模型加载失败: {e2}")
            raise

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        模型预测

        Returns:
            y_true: 真实标签
            y_pred: 预测概率
            y_pred_classes: 预测类别
        """
        print("开始模型预测...")

        # 重置生成器
        self.test_generator.reset()

        # 获取预测结果
        steps = len(self.test_generator)
        predictions = self.model.predict(
            self.test_generator,
            steps=steps,
            verbose=1
        )

        # 获取真实标签
        self.test_generator.reset()
        y_true = []
        for i in range(steps):
            _, batch_labels = next(self.test_generator)
            y_true.extend(np.argmax(batch_labels, axis=1))
        y_true = np.array(y_true)

        # 处理预测结果
        y_pred = predictions[:len(y_true)]  # 确保长度匹配
        y_pred_classes = np.argmax(y_pred, axis=1)

        print(f"预测完成: {len(y_true)} 个样本")
        print(f"预测概率形状: {y_pred.shape}")
        print(f"预测类别形状: {y_pred_classes.shape}")

        return y_true, y_pred, y_pred_classes

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_classes: np.ndarray) -> Dict:
        """
        计算所有评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测概率
            y_pred_classes: 预测类别

        Returns:
            包含所有指标的字典
        """
        print("计算评估指标...")

        # 基本指标
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='binary')
        recall = recall_score(y_true, y_pred_classes, average='binary')  # 敏感性
        f1 = f1_score(y_true, y_pred_classes, average='binary')

        # 计算特异性
        cm = confusion_matrix(y_true, y_pred_classes)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = 0

        # 多分类指标（为扩展性准备）
        precision_macro = precision_score(y_true, y_pred_classes, average='macro')
        recall_macro = recall_score(y_true, y_pred_classes, average='macro')
        f1_macro = f1_score(y_true, y_pred_classes, average='macro')

        # AUC指标
        try:
            if len(np.unique(y_true)) == 2:
                # 二分类AUC
                auc_roc = roc_auc_score(y_true, y_pred[:, 1])
                auc_pr = average_precision_score(y_true, y_pred[:, 1])
            else:
                # 多分类AUC
                y_true_bin = label_binarize(y_true, classes=list(range(len(self.class_names))))
                auc_roc = roc_auc_score(y_true_bin, y_pred, average='macro', multi_class='ovr')
                auc_pr = average_precision_score(y_true_bin, y_pred, average='macro')
        except:
            auc_roc = 0
            auc_pr = 0

        metrics = {
            # 基本指标
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': recall,  # 敏感性 = 召回率
            'specificity': specificity,
            'f1_score': f1,

            # 宏平均指标
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,

            # AUC指标
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,

            # 混淆矩阵
            'confusion_matrix': cm.tolist(),

            # 样本统计
            'total_samples': len(y_true),
            'correct_predictions': int(np.sum(y_true == y_pred_classes)),
            'wrong_predictions': int(np.sum(y_true != y_pred_classes))
        }

        return metrics

    def print_metrics(self, metrics: Dict):
        """打印评估指标"""
        print("\n" + "="*60)
        print("🎯 EDA-ResNet50 模型评估结果")
        print("="*60)

        # 基本指标
        print(f"\n📊 基本分类指标:")
        print(f"  准确率 (Accuracy):     {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision):    {metrics['precision']:.4f}")
        print(f"  敏感性 (Sensitivity):  {metrics['sensitivity']:.4f}")
        print(f"  特异性 (Specificity):  {metrics['specificity']:.4f}")
        print(f"  F1分数 (F1-Score):     {metrics['f1_score']:.4f}")

        # 宏平均指标
        print(f"\n📈 宏平均指标:")
        print(f"  宏平均精确率:          {metrics['precision_macro']:.4f}")
        print(f"  宏平均召回率:          {metrics['recall_macro']:.4f}")
        print(f"  宏平均F1分数:          {metrics['f1_macro']:.4f}")

        # AUC指标
        print(f"\n📉 AUC指标:")
        print(f"  ROC AUC:               {metrics['auc_roc']:.4f}")
        print(f"  PR AUC:                {metrics['auc_pr']:.4f}")

        # 样本统计
        print(f"\n📊 样本统计:")
        print(f"  总样本数:              {metrics['total_samples']}")
        print(f"  正确预测:              {metrics['correct_predictions']}")
        print(f"  错误预测:              {metrics['wrong_predictions']}")

        # 与论文目标对比
        print(f"\n🎯 与论文目标对比:")
        paper_diff_accuracy = metrics['accuracy'] - self.paper_targets['accuracy']
        paper_diff_sensitivity = metrics['sensitivity'] - self.paper_targets['sensitivity']
        paper_diff_specificity = metrics['specificity'] - self.paper_targets['specificity']

        print(f"  准确率: {metrics['accuracy']:.4f} (目标: {self.paper_targets['accuracy']:.4f}, 差异: {paper_diff_accuracy:+.4f}) {'✓' if metrics['accuracy'] >= self.paper_targets['accuracy'] else '✗'}")
        print(f"  敏感性: {metrics['sensitivity']:.4f} (目标: {self.paper_targets['sensitivity']:.4f}, 差异: {paper_diff_sensitivity:+.4f}) {'✓' if metrics['sensitivity'] >= self.paper_targets['sensitivity'] else '✗'}")
        print(f"  特异性: {metrics['specificity']:.4f} (目标: {self.paper_targets['specificity']:.4f}, 差异: {paper_diff_specificity:+.4f}) {'✓' if metrics['specificity'] >= self.paper_targets['specificity'] else '✗'}")

        # 总体评价
        all_targets_met = (
            metrics['accuracy'] >= self.paper_targets['accuracy'] and
            metrics['sensitivity'] >= self.paper_targets['sensitivity'] and
            metrics['specificity'] >= self.paper_targets['specificity']
        )

        print(f"\n🏆 论文复现状态: {'✓ 成功' if all_targets_met else '✗ 部分成功'}")

        # 混淆矩阵
        cm = np.array(metrics['confusion_matrix'])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"\n🔍 混淆矩阵详情:")
            print(f"  真负例 (TN): {tn}")
            print(f"  假正例 (FP): {fp}")
            print(f"  假负例 (FN): {fn}")
            print(f"  真正例 (TP): {tp}")

        print("\n" + "="*60)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred_classes: np.ndarray):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred_classes)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('EDA-ResNet50 Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

        # 保存图片
        save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray):
        """绘制ROC曲线"""
        if len(np.unique(y_true)) != 2:
            print("ROC曲线仅支持二分类，跳过绘制")
            return

        fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title('EDA-ResNet50 ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # 保存图片
        save_path = os.path.join(self.output_dir, 'roc_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存: {save_path}")
        plt.show()

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred: np.ndarray):
        """绘制PR曲线"""
        if len(np.unique(y_true)) != 2:
            print("PR曲线仅支持二分类，跳过绘制")
            return

        precision, recall, _ = precision_recall_curve(y_true, y_pred[:, 1])
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('EDA-ResNet50 Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        # 保存图片
        save_path = os.path.join(self.output_dir, 'pr_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR曲线已保存: {save_path}")
        plt.show()

    def save_results(self, metrics: Dict, y_true: np.ndarray, y_pred: np.ndarray, y_pred_classes: np.ndarray):
        """保存评估结果"""
        results = {
            'evaluation_time': datetime.now().isoformat(),
            'model_path': self.model_path,
            'data_path': self.data_dir,
            'paper_targets': self.paper_targets,
            'metrics': metrics,
            'class_names': self.class_names,
            'detailed_classification_report': classification_report(
                y_true, y_pred_classes,
                target_names=self.class_names,
                output_dict=True
            )
        }

        # 保存JSON结果
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"评估结果已保存: {results_path}")

        # 保存预测结果
        predictions_path = os.path.join(self.output_dir, 'predictions.npz')
        np.savez(predictions_path,
                y_true=y_true,
                y_pred=y_pred,
                y_pred_classes=y_pred_classes)
        print(f"预测结果已保存: {predictions_path}")

    def evaluate(self, plot_curves: bool = True, save_results: bool = True):
        """
        执行完整评估

        Args:
            plot_curves: 是否绘制曲线
            save_results: 是否保存结果
        """
        print("🚀 开始EDA-ResNet50模型评估...")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 模型预测
        y_true, y_pred, y_pred_classes = self.predict()

        # 计算指标
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_classes)

        # 打印结果
        self.print_metrics(metrics)

        # 绘制图表
        if plot_curves:
            print("\n📊 生成可视化图表...")
            self.plot_confusion_matrix(y_true, y_pred_classes)
            self.plot_roc_curve(y_true, y_pred)
            self.plot_precision_recall_curve(y_true, y_pred)

        # 保存结果
        if save_results:
            print("\n💾 保存评估结果...")
            self.save_results(metrics, y_true, y_pred, y_pred_classes)

        print(f"\n✅ 评估完成！结果保存在: {self.output_dir}")
        return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EDA-ResNet50 模型评估')

    parser.add_argument('--model-path', type=str, required=True,
                       help='训练好的模型路径 (.h5 文件)')

    parser.add_argument('--data-dir', type=str,
                       default="/root/EDA-ResNet50/training_data_Skin Cancer_Malignant_vs_Benign",
                       help='数据集目录路径')

    parser.add_argument('--output-dir', type=str, default=None,
                       help='评估结果输出目录')

    parser.add_argument('--no-plots', action='store_true',
                       help='不绘制可视化图表')

    parser.add_argument('--no-save', action='store_true',
                       help='不保存评估结果')

    args = parser.parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)

    try:
        # 创建评估器
        evaluator = EDAEvaluator(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )

        # 执行评估
        metrics = evaluator.evaluate(
            plot_curves=not args.no_plots,
            save_results=not args.no_save
        )

        print("\n🎉 评估脚本执行成功!")

    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()