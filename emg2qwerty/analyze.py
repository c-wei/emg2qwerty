import argparse
import torch
from pprint import pprint

def analyze_checkpoint(checkpoint_path: str) -> None:
    """Analyze model hyperparameters from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    print(f"Checkpoint file: {checkpoint_path}")
    print("=" * 60)
    
    # Print top-level keys in the checkpoint
    print("\nTop-level keys in checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Print hyperparameters if available
    if "hyper_parameters" in checkpoint:
        print("\nHyperparameters:")
        print("-" * 40)
        pprint(checkpoint["hyper_parameters"], indent=2)
    
    # Print hparams if available (alternative key name)
    if "hparams" in checkpoint:
        print("\nHparams:")
        print("-" * 40)
        pprint(checkpoint["hparams"], indent=2)
    
    # Print model state dict info
    if "state_dict" in checkpoint:
        print("\nModel state_dict keys and shapes:")
        print("-" * 40)
        state_dict = checkpoint["state_dict"]
        total_params = 0
        for key, value in state_dict.items():
            if hasattr(value, "shape"):
                num_params = value.numel()
                total_params += num_params
                print(f"  {key}: {list(value.shape)} ({num_params:,} params)")
            else:
                print(f"  {key}: {type(value)}")
        print(f"\nTotal parameters: {total_params:,}")
    
    # Print optimizer state if available
    if "optimizer_states" in checkpoint:
        print("\nOptimizer states available: Yes")
    
    # Print epoch/step info if available
    if "epoch" in checkpoint:
        print(f"\nEpoch: {checkpoint['epoch']}")
    if "global_step" in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")
    
    # Print callbacks if available (PyTorch Lightning)
    if "callbacks" in checkpoint:
        print("\nCallbacks:")
        print("-" * 40)
        pprint(checkpoint["callbacks"], indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model hyperparameters from a checkpoint file"
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint file (.ckpt or .pt)",
    )
    args = parser.parse_args()
    
    analyze_checkpoint(args.checkpoint_path)


if __name__ == "__main__":
    main()