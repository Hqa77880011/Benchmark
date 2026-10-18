# SAM-Labelimg

Interactive annotation with the bundled Segment Anything Model (SAM) and SAM-Tool. This is an annotation workflow, not the manuscript's fixed zero-shot evaluation protocol.

## 1. Install the Bundled Projects

Start from the repository root in the activated virtual environment described in the [main README](../../README.md). The SAM implementation and helper scripts are already included; no additional clone or helper-folder copy is required.

```bash
python -X utf8 -m pip install -r sam/SAM-Tool/requirements.txt
python -X utf8 -m pip install -e ./sam/segment-anything
```

The requirements include PyQt5 for the GUI, pycocotools for COCO masks, ONNX for export, and ONNX Runtime for inference. The annotation GUI needs a desktop display. The optional COCO viewer uses Tk, which must be available in the Python installation (for example, `python3-tk` on Ubuntu).

Download the [SAM ViT-H checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and save it as `data/sam_vit_h_4b8939.pth`. The default model type is `default` (ViT-H); a different checkpoint requires its matching `--model_type`, such as `vit_b`.

## 2. Prepare the Annotation Dataset

Create `data/sam_annotation/images/`, place the PNG/JPG images to annotate there, and keep a separate `data/sam_annotation/embeddings/` folder for generated embeddings:

```text
data/sam_annotation/
├── images/*.png
└── embeddings/
```

The embedding script creates the embeddings folder if it does not already exist. This flat annotation layout is separate from the benchmark archive's category-grouped `img/` and `mask/` folders.

## 3. Generate Embeddings and Export ONNX

Run all commands from the repository root:

```bash
python sam/segment-anything/helpers/extract_embeddings.py --checkpoint-path data/sam_vit_h_4b8939.pth --dataset-folder data/sam_annotation --device cpu
python sam/segment-anything/helpers/generate_onnx.py --checkpoint-path data/sam_vit_h_4b8939.pth --onnx-model-path data/sam_onnx.onnx --orig-im-size 360 360
```

`--device cpu` works without a GPU; use `--device cuda` for embedding extraction on a compatible GPU. Embeddings are saved as NPY files with the image filename stem under `data/sam_annotation/embeddings/`.

`--orig-im-size` takes `(height, width)` and must match the images to annotate. The example uses 360 × 360 images. The bundled exporter traces the original-size resizing operation at export time, so prepare a separate model/dataset workflow for each image size. The model type used for export must match the one used for embedding extraction.

## 4. Annotate Images

```bash
python sam/SAM-Tool/segment_anything_annotator.py --onnx-model-path data/sam_onnx.onnx --dataset-path data/sam_annotation --categories defect
```

`--categories` accepts comma-separated names without spaces. Left-click adds a positive prompt and right-click adds a negative prompt.

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

COCO annotations are saved to `data/sam_annotation/annotations.json`.

## 5. Inspect Annotations

```bash
python sam/SAM-Tool/cocoviewer.py -i data/sam_annotation -a data/sam_annotation/annotations.json
```

The generated COCO `file_name` values include the `images/` prefix, so `-i` points to the dataset root, not its `images/` subfolder.

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
