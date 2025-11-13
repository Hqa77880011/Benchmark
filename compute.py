import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def calculate_metrics(pred, true):
    """计算 IoU、Dice、Precision 和 Recall，将值>0的像素视为前景"""
    pred_foreground = (pred > 0).astype(np.uint8)
    true_foreground = (true > 0).astype(np.uint8)

    # 尺寸对齐
    if pred_foreground.shape != true_foreground.shape:
        pred_foreground = np.array(Image.fromarray(pred_foreground).resize(
            (true_foreground.shape[1], true_foreground.shape[0]), Image.NEAREST
        ))

    TP = np.sum((true_foreground == 1) & (pred_foreground == 1))
    FP = np.sum((true_foreground == 0) & (pred_foreground == 1))
    FN = np.sum((true_foreground == 1) & (pred_foreground == 0))

    eps = 1e-10

    # 交并比 (IoU)
    iou = TP / (TP + FP + FN + eps)

    # Dice系数
    dice = 2 * TP / (2 * TP + FP + FN + eps)

    # Precision
    precision = TP / (TP + FP + eps)

    # Recall
    recall = TP / (TP + FN + eps)

    return {
        'IoU': iou,
        'Dice': dice,
        'Precision': precision,
        'Recall': recall
    }


def main():
    parser = argparse.ArgumentParser(description="计算图像分割指标 (IoU, Dice, Precision, Recall)")
    parser.add_argument("--gt-dir", required=True, help="真实mask文件夹路径")
    parser.add_argument("--pred-dir", required=True, help="预测mask文件夹路径")
    parser.add_argument("--output-dir", required=True, help="结果保存路径")
    parser.add_argument("--suffix", default="_mask", help="预测mask文件名后缀（默认: _mask）")
    args = parser.parse_args()

    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    output_dir = args.output_dir
    suffix = args.suffix

    os.makedirs(output_dir, exist_ok=True)

    gt_files = [f for f in os.listdir(gt_dir)
                if os.path.isfile(os.path.join(gt_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    all_metrics = []

    for gt_file in tqdm(gt_files, desc="Processing images"):
        base_name = os.path.splitext(gt_file)[0]
        true_mask_path = os.path.join(gt_dir, gt_file)

        try:
            true_mask = Image.open(true_mask_path)
            if true_mask.mode != 'L':
                true_mask = true_mask.convert('L')
            true_mask = np.array(true_mask)
        except Exception as e:
            print(f"警告: 无法读取真实mask {true_mask_path}: {e}，跳过")
            continue

        pred_mask_name = f"{base_name}{suffix}.png"
        pred_mask_path = os.path.join(pred_dir, pred_mask_name)

        if not os.path.exists(pred_mask_path):
            print(f"警告: 找不到预测mask {pred_mask_path}，跳过")
            continue

        try:
            pred_mask = Image.open(pred_mask_path)
            if pred_mask.mode != 'L':
                pred_mask = pred_mask.convert('L')
            pred_mask = np.array(pred_mask)
        except Exception as e:
            print(f"警告: 无法读取预测mask {pred_mask_path}: {e}，跳过")
            continue

        metrics = calculate_metrics(pred_mask, true_mask)
        all_metrics.append(metrics)

        print(f"\n处理文件: {base_name}")
        print(f"  IoU: {metrics['IoU']:.4f}")
        print(f"  Dice: {metrics['Dice']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")

    if all_metrics:
        avg_metrics = {
            'IoU': np.mean([m['IoU'] for m in all_metrics]),
            'Dice': np.mean([m['Dice'] for m in all_metrics]),
            'Precision': np.mean([m['Precision'] for m in all_metrics]),
            'Recall': np.mean([m['Recall'] for m in all_metrics])
        }

        print("\n平均指标:")
        print(f"IoU: {avg_metrics['IoU']:.4f}")
        print(f"Dice: {avg_metrics['Dice']:.4f}")
        print(f"Precision: {avg_metrics['Precision']:.4f}")
        print(f"Recall: {avg_metrics['Recall']:.4f}")

        result_path = os.path.join(output_dir, "metrics_results.txt")
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write("图像分割评估指标结果\n")
            f.write("=====================\n\n")
            f.write("单张图像指标:\n")
            for i, metrics in enumerate(all_metrics):
                base_name = os.path.splitext(gt_files[i])[0]
                f.write(f"图像 {base_name}:\n")
                f.write(f"  IoU: {metrics['IoU']:.4f}\n")
                f.write(f"  Dice: {metrics['Dice']:.4f}\n")
                f.write(f"  Precision: {metrics['Precision']:.4f}\n")
                f.write(f"  Recall: {metrics['Recall']:.4f}\n\n")

            f.write("\n平均指标:\n")
            f.write(f"IoU: {avg_metrics['IoU']:.4f}\n")
            f.write(f"Dice: {avg_metrics['Dice']:.4f}\n")
            f.write(f"Precision: {avg_metrics['Precision']:.4f}\n")
            f.write(f"Recall: {avg_metrics['Recall']:.4f}\n")

        print(f"\n完整结果已保存到: {result_path}")
    else:
        print("没有计算任何指标，请检查输入路径和文件格式。")


if __name__ == "__main__":
    main()
