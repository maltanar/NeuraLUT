#  This file is part of NeuraLUT.
#
#  NeuraLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.

import os
from argparse import ArgumentParser

import torch
from brevitas.export import export_qonnx

from models import MnistNeqModel
from train import configs, model_config


def main():
    parser = ArgumentParser(description="Export MNIST NeuraLUT model to QONNX with Brevitas")
    parser.add_argument(
        "--arch",
        type=str,
        choices=configs.keys(),
        default="hdr-5l-0.1",
        help="Specific the neural network model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file containing model_dict",
    )
    parser.add_argument(
        "--imask",
        type=str,
        required=True,
        help="Input mask file used to instantiate the sparse model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./test_demo/model_qonnx.onnx",
        help="Output QONNX model path (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy input batch size used for export (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device id when --cuda is enabled (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Export with CUDA model/input tensors (default: %(default)s)",
    )
    args = parser.parse_args()

    defaults = configs[args.arch]
    model_cfg = {k: defaults[k] for k in model_config.keys()}
    model_cfg["input_length"] = 784
    model_cfg["output_length"] = 10
    model_cfg["dense_forward"] = False
    model_cfg["cuda"] = args.cuda

    map_loc = f"cuda:{args.device}" if args.cuda else "cpu"
    model_cfg["imask"] = torch.load(args.imask, map_location=map_loc)

    if args.cuda:
        torch.cuda.set_device(args.device)

    model = MnistNeqModel(model_cfg)
    if args.cuda:
        model.cuda()

    checkpoint = torch.load(args.checkpoint, map_location=map_loc)
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()

    input_t = torch.randn(args.batch_size, model_cfg["input_length"])
    if args.cuda:
        input_t = input_t.cuda()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # torch.onnx exporter in newer torch versions can bypass Brevitas QONNX ops;
    # dynamo=False keeps the legacy export path that correctly emits QONNX nodes.
    export_qonnx(model, input_t, export_path=args.output, dynamo=False)
    print(f"Exported QONNX model to: {args.output}")


if __name__ == "__main__":
    main()
