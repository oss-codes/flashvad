from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flashvad")
    subcommands = parser.add_subparsers(dest="command", required=True)

    training = subcommands.add_parser("train", help="train a VAD checkpoint")
    training.add_argument("--config", required=True)
    training.add_argument("--train-manifest", required=True)
    training.add_argument("--valid-manifest", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--device")
    training.add_argument("--region-profile")
    training.add_argument("--noise-manifest")

    benchmark = subcommands.add_parser("benchmark", help="benchmark streaming CPU latency")
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--iterations", type=int, default=5_000)
    benchmark.add_argument("--warmup", type=int, default=500)
    benchmark.add_argument("--threads", type=int, default=1)

    onnx_benchmark = subcommands.add_parser(
        "benchmark-onnx", help="benchmark a streaming ONNX graph"
    )
    onnx_benchmark.add_argument("--model", required=True)
    onnx_benchmark.add_argument("--iterations", type=int, default=5_000)
    onnx_benchmark.add_argument("--warmup", type=int, default=500)
    onnx_benchmark.add_argument("--threads", type=int, default=1)

    export = subcommands.add_parser("export", help="export an offline-sequence ONNX graph")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--sequence-frames", type=int, default=100)
    export.add_argument("--mode", choices=("streaming", "offline"), default="streaming")

    quantize = subcommands.add_parser("quantize", help="create a dynamic int8 ONNX graph")
    quantize.add_argument("--model", required=True)
    quantize.add_argument("--output", required=True)

    native_export = subcommands.add_parser(
        "export-native",
        help="export embedded C weights for the macOS Accelerate runtime",
    )
    native_export.add_argument("--checkpoint", required=True)
    native_export.add_argument("--output", required=True)

    native_benchmark = subcommands.add_parser(
        "benchmark-native",
        help="build and benchmark the embedded macOS Accelerate runtime",
    )
    native_benchmark.add_argument("--checkpoint", required=True)
    native_benchmark.add_argument("--output", required=True)

    evaluation = subcommands.add_parser(
        "evaluate",
        help="evaluate frame, boundary, language, and domain metrics",
    )
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--output")
    evaluation.add_argument("--threshold", type=float)
    evaluation.add_argument("--bootstrap-iterations", type=int, default=0)
    evaluation.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        from .train import train

        path = train(
            args.config,
            args.train_manifest,
            args.valid_manifest,
            args.output,
            args.device,
            args.region_profile,
            args.noise_manifest,
        )
        print(path)
    elif args.command == "benchmark":
        from .benchmark import benchmark_checkpoint

        benchmark_checkpoint(args.checkpoint, args.iterations, args.warmup, args.threads)
    elif args.command == "benchmark-onnx":
        from .benchmark import benchmark_onnx

        benchmark_onnx(args.model, args.iterations, args.warmup, args.threads)
    elif args.command == "export":
        from .export import export_onnx

        path = export_onnx(args.checkpoint, args.output, args.sequence_frames, args.mode)
        print(path)
    elif args.command == "quantize":
        from .quantize import quantize_dynamic_int8

        path = quantize_dynamic_int8(args.model, args.output)
        print(path)
    elif args.command == "export-native":
        from .native_export import export_macos_native

        path = export_macos_native(args.checkpoint, args.output)
        print(path)
    elif args.command == "benchmark-native":
        from .native_export import benchmark_macos_native

        benchmark_macos_native(args.checkpoint, args.output)
    elif args.command == "evaluate":
        from .evaluation import evaluate_checkpoint

        evaluate_checkpoint(
            args.checkpoint,
            args.manifest,
            args.output,
            args.threshold,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )


if __name__ == "__main__":
    main()
