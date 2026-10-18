# SAM-Labelimg

Fast annotation with the Segment Anything Model (SAM).

## 1. Obtain the Projects

Project 1: [SAM-Tool](https://github.com/zhouayi/SAM-Tool)

Project 2: [Segment Anything](https://github.com/facebookresearch/segment-anything)

```bash
git clone https://github.com/zhouayi/SAM-Tool.git

git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

Download the [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

## 2. Prepare the Dataset

Place images under `<dataset_path>/images/*` and create an empty directory at `<dataset_path>/embeddings`.

## 3. Prepare Image Embeddings and the ONNX Model

Copy the `helpers` folder from Project 1 into the root directory of Project 2.

### 3.1 Extract Image Embeddings

Run `extract_embeddings.py` to extract image embeddings:

```bash
# Run from the root directory of Project 2.
python helpers\extract_embeddings.py --checkpoint-path sam_vit_h_4b8939.pth --dataset-folder <dataset_path> --device cpu
```

- `checkpoint-path`: path to the downloaded SAM checkpoint.
- `dataset-folder`: path to the dataset.
- `device`: defaults to `cuda`. Use `cpu` if a GPU is unavailable; processing will be slower.

The corresponding NPY files will be written to `<dataset_path>/embeddings`.

### 3.2 Export the Checkpoint to ONNX

Run `generate_onnx.py` to convert the PTH checkpoint to an ONNX model:

```bash
# Run from the root directory of Project 2.
python helpers\generate_onnx.py --checkpoint-path sam_vit_h_4b8939.pth --onnx-model-path ./sam_onnx.onnx --orig-im-size 1080 1920
```

- `checkpoint-path`: path to the same SAM checkpoint.
- `onnx-model-path`: destination path for the exported ONNX model.
- `orig-im-size`: original image dimensions in `(height, width)` order.

**The provided conversion code does not support dynamic original image dimensions.** If the dataset contains different image sizes, export separate ONNX models with the appropriate `orig-im-size` values.

## 4. Annotate Images

Copy the generated `sam_onnx.onnx` file into the root directory of Project 1, then run `segment_anything_annotator.py`:

```bash
# Run from the root directory of Project 1.
python segment_anything_annotator.py --onnx-model-path sam_onnx.onnx --dataset-path <dataset_path> --categories cat,dog
```

- `onnx-model-path`: path to the exported ONNX model.
- `dataset-path`: path to the dataset.
- `categories`: comma-separated category names without spaces.

Left-click on an object to add a positive mask prompt; right-click to add a negative prompt.

Keyboard shortcuts:

| Key | Action |
| --- | --- |
| `Esc` | Exit the application |
| `a` | Previous image |
| `d` | Next image |
| `k` | Decrease transparency |
| `l` | Increase transparency |
| `n` | Add an object |
| `r` | Reset |
| `Ctrl+s` | Save |
| `Ctrl+z` | Undo the last annotated object |

![Interactive annotation example](assets/catdog.gif)

Annotations are saved in COCO format at `<dataset_path>/annotations.json`.

## 5. Inspect Annotations

```bash
python cocoviewer.py -i <dataset_path> -a <dataset_path>\annotations.json
```

![Annotation viewer example](assets/catdog.png)

## 6. Customize the Display

Adjust the annotation box width and mask transparency in `salt/display_utils.py`. The following snippet illustrates these settings:

```python
# salt/display_utils.py
class DisplayUtils:
    def __init__(self):
        self.transparency = 0.65 # Example mask transparency setting
        self.box_width = 2 # Bounding box line width
```

To change the annotation text format, edit `draw_box_on_image` in the same file:

```python
# salt/display_utils.py

def draw_box_on_image(self, image, categories, ann, color):
    x, y, w, h = ann["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_width)

    text = '{} {}'.format(ann["id"],categories[ann["category_id"]])
    txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_size = cv2.getTextSize(text, font, 1.5, 1)[0]
    cv2.rectangle(image, (x, y + 1), (x + txt_size[0] + 1, y + int(1.5*txt_size[1])), color, -1)
    cv2.putText(image, text, (x, y + txt_size[1]), font, 1.5, txt_color, thickness=5)
    return image
```

Undoing the last annotated object with `Ctrl+z` was added on 2023-04-14.

## References

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [SALT](https://github.com/anuragxel/salt)
- [COCO Viewer](https://github.com/trsvchn/coco-viewer)
