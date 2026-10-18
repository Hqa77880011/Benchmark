# LithoDefectV1 — 光刻缺陷分割合成基准

> **Advancing Lithography Process Control in Integrated Circuit Manufacturing via a Synthetic Benchmark for Defect Segmentation**
>
> Qinao Hu, Jiwei Shen — East China Normal University
>
> **IEEE Transactions on Semiconductor Manufacturing：大修返修已完成，审稿中。**

LithoDefectV1 包含 3,600 对合成 SEM 风格图像与像素级缺陷标注，覆盖 8 类缺陷。本仓库提供合成流水线、YOLO / Swin-UNet 训练入口、SAM 工具和掩码评估脚本。

## 论文修订与评测协议

修订稿采用以下三层评测。Open-test 是同分布留出测试，不能解释为开放集识别或跨背景域测试。

| 场景 | 图像数 | 定义 |
| --- | ---: | --- |
| Train | 2,518 | LithoDefectV1 训练样本 |
| Open-test | 1,082 | 通过分层 7:3 划分留出的合成样本 |
| Re-synth-test | 300 | 使用相同合成流程、不同随机种子独立生成的样本 |
| Fab-test | 92 | 合成流程之外采集并独立标注的真实工业 SEM 图像 |

SAM 使用 **129 张 Open-test 分层子集**，以及完整的 Re-synth-test 和 Fab-test，均为零样本推理。SAM 与其他模型的 Open-test 样本数不同。

本次更新修复了合成参数传递、源码 LFS 指针、验证加载器、数据配对和运行入口。详细的版本边界与验证方式见 [复现说明](docs/REPRODUCIBILITY.md)。论文表格中的数字来自修订稿，本次仓库维护没有重新训练模型或重算这些实验。

## 仓库结构

```text
Benchmark/
├── README.md
├── docs/REPRODUCIBILITY.md       # 论文协议、代码行为、资源与验证记录
├── data.zip                      # Git LFS 数据资源
├── compute.py                    # 完整配对的二值掩码评估
├── tests/test_workflow.py         # 与本次修复对应的回归检查
├── synthesis_pipeline/
│   ├── run_pipeline.py           # 合成 → 划分 → 格式导出
│   ├── config.yaml
│   └── lithodefect/
├── yolo/                         # 保留本地 Ultralytics 实现及其许可证
│   ├── trains.py                 # 参数化实例分割训练入口
│   └── ultralytics/
├── unet/                         # Swin-UNet、PNG/NPZ 加载器、训练与推理
│   └── UPSTREAM.md               # 原始 Swin-Unet 说明与引用
└── sam/                          # SAM 实现与交互式标注工具
```

上游演示网站、示例、文档构建、测试镜像、缓存和旧日志不作为本基准的运行依赖提交。模型实现、配置、许可证和已有权重入口保留。

## 获取数据与权重

请使用 Git 克隆仓库。GitHub 的 ZIP 源码下载可能只包含大文件的 LFS 指针。

```bash
git clone https://github.com/Hqa77880011/Benchmark.git
cd Benchmark
git lfs install
git lfs pull --include="data.zip"
python -m zipfile -e data.zip data
```

按需下载已有模型资源，避免一次拉取所有权重：

```bash
git lfs pull --include="unet/*.pth,unet/pretrained_ckpt/*.pth,yolo/best_*"
```

数据压缩包与模型仍通过原有 LFS 对象提供；本次未下载或验证全部大文件。合成流程要求独立准备种子缺陷图、对应掩码与背景，不能把发布图像集默认当作完整种子素材。具体输入结构见 [合成流水线说明](synthesis_pipeline/README.md)。

## 合成与训练

以下命令从仓库根目录执行。合成 CPU 即可运行，模型训练需另行安装 PyTorch 与模型依赖。

```bash
python -m pip install -r synthesis_pipeline/requirements.txt
python synthesis_pipeline/run_pipeline.py --list-steps
python synthesis_pipeline/run_pipeline.py --config synthesis_pipeline/config.yaml
```

配置中的路径相对于 **YAML 文件所在目录** 解析。默认输入位于 `synthesis_pipeline/data/`，输出位于 `synthesis_pipeline/output/`。同一配置与随机种子可重现生成过程；重新运行时使用空输出目录，避免混入旧样本。

默认划分工具按类别和背景来源生成 70/10/10/10 的 train、val、test/private、test/public，**这是工具的可配置划分，不是论文原始 7:3 划分的索引复刻**。两种背景来源都会进入训练/验证。`test_private` 仍是合成图像，不能当作 Fab-test；`test_public` 也不能直接当作论文 Open-test。论文逐样本划分清单及完整实验环境尚未随本次维护补齐。

YOLO 使用仓库内保留的 Ultralytics 版本，安装和训练：

```bash
python -m pip install -e ./yolo
python yolo/trains.py --data synthesis_pipeline/output/synthetic_yolo/data.yaml --model yolo11n-seg.pt --device 0 --imgsz 384 --epochs 50 --batch 64
```

`--model` 需要分割模型或分割架构；其他 YOLO 版本请传入对应 checkpoint。本地保留版本见 `yolo/ultralytics/__init__.py`，不依赖自动安装最新版本。

Swin-UNet 默认导出二值缺陷标签（背景 0、缺陷 1），与 2 输出通道配套。可选的八类语义分割需设置 `swinunet.binary: false`，并使用 `--num_classes 9`；这属于不同实验设置。

```bash
python -m pip install -r unet/requirements.txt
cd unet
python train.py --root_path ../synthesis_pipeline/output/swin_unet_data --dataset LithoDefect --output_dir ./model_out --num_classes 2 --img_size 224 --batch_size 32 --max_epochs 50
python test.py --root_path ../synthesis_pipeline/output/swin_unet_data --dataset LithoDefect --output_dir ./model_out --num_classes 2 --split_name test_public
```

预训练 Swin-T 权重需先通过 LFS 获取。PNG 数据按现有 train/val 目录读取，验证集不做随机增强；旧 NPZ 数据入口继续支持显式 `--list_dir`。SAM 安装、标注与提示操作见 [SAM 工具说明](sam/SAM-Tool/README.md)。

## 掩码评估

```bash
python compute.py --gt-dir data/masks --pred-dir results/pred_masks --output-dir results --suffix _mask
```

预测文件名为 `<GT文件主名><suffix>.png`。预测使用相同文件名时传入 `--suffix ""`。缺失预测、空数据、损坏文件或尺寸不符会报错；确需对齐尺寸时显式传入 `--resize`，使用最近邻插值。

输出 `metrics_results.txt`，记录样本数、实际文件名、逐图指标和平均指标。默认 `--average image` 为逐图宏平均。若文件主名以 `<Category>_` 开头，可用 `--average category` 先按类别取平均、再对类别等权平均。两种汇总方式不能混用。大于 0 的像素视为前景，零分母按 0 处理。

## 修订稿报告的结果

以下仅为修订稿表格转录，保留四位小数。SAM 的 Open-test 使用 129 张子集。

| 模型 | 测试集 | mIoU | Dice | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8-seg | Open-test | 0.8508 | 0.9177 | 0.8663 | 0.9786 |
| YOLOv8-seg | Re-synth-test | 0.8155 | 0.8841 | 0.8320 | 0.9435 |
| YOLOv8-seg | Fab-test | 0.8130 | 0.8813 | 0.8301 | 0.9422 |
| YOLOv9-seg | Open-test | 0.8552 | 0.9203 | 0.8722 | 0.9770 |
| YOLOv9-seg | Re-synth-test | 0.8202 | 0.8878 | 0.8394 | 0.9420 |
| YOLOv9-seg | Fab-test | 0.8185 | 0.8860 | 0.8375 | 0.9411 |
| YOLOv11-seg | Open-test | 0.8530 | 0.9190 | 0.8687 | 0.9784 |
| YOLOv11-seg | Re-synth-test | 0.8164 | 0.8847 | 0.8336 | 0.9442 |
| YOLOv11-seg | Fab-test | 0.8170 | 0.8845 | 0.8352 | 0.9428 |
| YOLOv12-seg | Open-test | 0.8492 | 0.9168 | 0.8631 | 0.9807 |
| YOLOv12-seg | Re-synth-test | 0.8120 | 0.8820 | 0.8271 | 0.9470 |
| YOLOv12-seg | Fab-test | 0.8144 | 0.8828 | 0.8292 | 0.9453 |
| SAM (zero-shot) | Open-test (129) | 0.7884 | 0.8798 | 0.9978 | 0.7896 |
| SAM (zero-shot) | Re-synth-test | 0.7412 | 0.8365 | 0.9590 | 0.7408 |
| SAM (zero-shot) | Fab-test | 0.7480 | 0.8398 | 0.9594 | 0.7487 |
| Swin-UNet | Open-test | 0.8725 | 0.9273 | 0.9344 | 0.9260 |
| Swin-UNet | Re-synth-test | 0.8280 | 0.8921 | 0.8905 | 0.8804 |
| Swin-UNet | Fab-test | 0.8348 | 0.8927 | 0.8934 | 0.8849 |

## 引用与许可

论文仍在审稿阶段，暂不填写已发表期刊卷期、页码或 DOI。

```bibtex
@unpublished{hu_lithodefect_revision,
  title = {Advancing Lithography Process Control in Integrated Circuit Manufacturing via a Synthetic Benchmark for Defect Segmentation},
  author = {Hu, Qinao and Shen, Jiwei},
  note = {Manuscript under review at IEEE Transactions on Semiconductor Manufacturing; major revision completed}
}
```

数据集沿用原 README 声明的 CC BY 4.0，自有代码沿用原 README 的 MIT 声明。第三方目录保留各自的许可证：YOLO 见 [yolo/LICENSE](yolo/LICENSE)，SAM 见 [sam/segment-anything/LICENSE](sam/segment-anything/LICENSE) 与 [sam/SAM-Tool/LICENSE](sam/SAM-Tool/LICENSE)。根目录的许可声明不覆盖这些第三方代码。

代码与数据问题请通过 [GitHub Issues](https://github.com/Hqa77880011/Benchmark/issues) 反馈。
