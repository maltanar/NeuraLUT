## NeuraLUT-Assemble on 10-Class 28x28 Quick, Draw!

This folder mirrors the MNIST NeuraLUT flow for a 10-class 28x28 Quick, Draw! example.

It includes:
- Data preparation scripts adapted from https://github.com/maltanar/QuickDraw-brevitas
- Training for dense-mask discovery + sparse NeuraLUT model training
- LUT conversion, Verilog generation/simulation, and optional synthesis
- QONNX export and layer-by-layer comparison helper

### Files

- `prepare_data.py`: Downloads class `.npy` files and builds cached splits.
- `generate_data.py`: Creates `train.npz`, `test.npz`, and `class_names.txt`.
- `dataset.py`: QuickDraw dataset loader (`QD_Dataset`).
- `train.py`: Trains NeuraLUT model on QuickDraw cached splits.
- `neq2lut.py`: Converts trained model to LUTs and optionally Verilog/synthesis.
- `export_qonnx.py`: Exports trained model to QONNX.
- `compare_layers.py`: Compares intermediate tensors between PyTorch and QONNX.

### 1) Prepare data (10 classes, 28x28)

From this directory:

```bash
python prepare_data.py --download 1 --categories 10 -v 0.2 -msc 5000
```

This produces:
- `quickdraw_data/Data/*.npy`
- `quickdraw_data/Dataset/train.npz`
- `quickdraw_data/Dataset/test.npz`
- `quickdraw_data/Dataset/class_names.txt`

### 2) Train (optional if you already have checkpoints)

```bash
python train.py --arch hdr-5l-0.1 --log_dir quickdraw_demo --cuda --device 0
```

Outputs are saved under:
- `test_quickdraw_demo/checkpoint.pth`
- `test_quickdraw_demo/best_accuracy.pth`
- `test_quickdraw_demo/imask.pth`

### 3) Convert to LUT/Verilog and evaluate

```bash
python neq2lut.py --arch hdr-5l-0.1 \
  --checkpoint ./test_quickdraw_demo/best_accuracy.pth \
  --imask ./test_quickdraw_demo/imask.pth \
  --log-dir ./test_quickdraw_demo/verilog/ \
  --data-root ./quickdraw_data/Dataset \
  --add-registers \
  --device 0 \
  --cuda
```

### 4) Export QONNX

```bash
python export_qonnx.py \
  --arch hdr-5l-0.1 \
  --checkpoint ./test_quickdraw_demo/best_accuracy.pth \
  --imask ./test_quickdraw_demo/imask.pth \
  --output ./test_quickdraw_demo/quickdraw_qonnx.onnx
```

### Notes

- The data preparation workflow and category list follow QuickDraw-brevitas.
- `prepare_data.py` is currently configured for the 10-class example (`10categories.txt`).
- Input images are flattened 28x28 grayscale sketches (`784` features) and scaled with `/255.0`.
