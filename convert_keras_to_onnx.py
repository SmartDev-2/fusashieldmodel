"""Export the FusaShield Keras model to ONNX for inference."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "fusashield_mobilenetv2.keras"


def _patch_constructor(
    layer_class: type,
    unsupported_arguments: tuple[str, ...],
) -> None:
    original_init = layer_class.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        for argument in unsupported_arguments:
            kwargs.pop(argument, None)
        original_init(self, *args, **kwargs)

    layer_class.__init__ = patched_init


def _patch_legacy_model_config(keras: Any) -> None:
    initializer_modules = [keras.initializers]
    try:
        from keras.src.initializers import random_initializers
    except ImportError:
        pass
    else:
        initializer_modules.append(random_initializers)

    patched_classes: set[type] = set()
    for module in initializer_modules:
        for candidate in vars(module).values():
            if (
                isinstance(candidate, type)
                and issubclass(candidate, keras.initializers.Initializer)
                and candidate not in patched_classes
            ):
                _patch_constructor(candidate, ("input_axes", "output_axes"))
                patched_classes.add(candidate)

    _patch_constructor(
        keras.layers.BatchNormalization,
        ("renorm", "renorm_clipping", "renorm_momentum"),
    )
    _patch_constructor(keras.layers.Dense, ("quantization_config",))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the FusaShield .keras model to an inference ONNX file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Input Keras model (default: {DEFAULT_MODEL.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output ONNX file (default: next to the input model)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    model_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else model_path.with_suffix(".onnx")
    )

    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")
    if model_path == output_path:
        parser.error("Input and output paths must be different.")
    if output_path.suffix.lower() != ".onnx":
        parser.error("Output filename must end in .onnx.")
    if args.opset < 13:
        parser.error("Use ONNX opset 13 or later.")

    os.environ["KERAS_BACKEND"] = "torch"

    try:
        import keras
    except ImportError as exc:
        raise SystemExit(
            "Keras is not installed in this Python environment. "
            "Install the project's Keras and PyTorch dependencies first."
        ) from exc

    try:
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "The ONNX package is not installed for this interpreter. "
            f"Install it with: \"{sys.executable}\" -m pip install onnx onnxscript"
        ) from exc

    if importlib.util.find_spec("onnxscript") is None:
        raise SystemExit(
            "The PyTorch ONNX exporter requires onnxscript. "
            f"Install it with: \"{sys.executable}\" -m pip install onnxscript"
        )

    _patch_legacy_model_config(keras)

    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    if not hasattr(model, "export"):
        raise RuntimeError(
            "This Keras version does not support Model.export(). "
            "Use a recent Keras 3 release."
        )
    print(f"Input shape: {model.input_shape}")
    if isinstance(model.input_shape, list):
        raise ValueError("This converter expects a model with a single input.")
    dummy_shape = tuple(
        1 if dimension is None else int(dimension)
        for dimension in model.input_shape
    )
    model(keras.ops.zeros(dummy_shape, dtype="float32"), training=False)

    print(f"Exporting ONNX (opset {args.opset}): {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.export(
        str(output_path),
        format="onnx",
        verbose=False,
        opset_version=args.opset,
    )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ONNX export did not create a valid file: {output_path}")

    exported_model = onnx.load(str(output_path))
    onnx.checker.check_model(exported_model)
    print(f"ONNX model validated successfully: {output_path}")


if __name__ == "__main__":
    main()
