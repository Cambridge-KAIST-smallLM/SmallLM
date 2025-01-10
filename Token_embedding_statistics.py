import os
import torch
import csv
from safetensors.torch import safe_open  # Efficient safetensors loading

def root_mean_square(tensor):
    """Computes the root mean square (RMS) of each row."""
    return torch.sqrt(torch.mean(tensor ** 2, dim=-1))

def analyze_checkpoints(output_dir, csv_filename="checkpoint_statistics.csv"):
    """
    Iterate over all checkpoint folders in the output directory, compute statistics for `lm_head.weight`,
    and save results into a CSV file efficiently.
    """
    checkpoint_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint")])

    # Open CSV file for writing (to avoid keeping everything in memory)
    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Checkpoint", "Norm Mean", "Mean Abs Mean", "Mean Std", "Mean RMS", "Std RMS"])  # CSV Header

        for checkpoint in checkpoint_dirs:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            model_path = os.path.join(checkpoint_path, "model.safetensors")  # Adjust if needed

            if not os.path.exists(model_path):
                print(f"Skipping {checkpoint} (no model file found)")
                continue

            # Open `model.safetensors` in a memory-efficient way
            with safe_open(model_path, framework="pt", device="cpu") as f:
                if "lm_head.weight" not in f.keys():
                    print(f"Skipping {checkpoint} (lm_head.weight not found)")
                    continue

                # Load only the required tensor
                lm_head_weight = f.get_tensor("lm_head.weight")

                # Compute requested statistics
                norm_mean_1 = torch.norm(torch.mean(lm_head_weight, dim=0)).item()
                abs_mean_2 = torch.mean(torch.abs(torch.mean(lm_head_weight, dim=-1))).item()
                std_mean_3 = torch.mean(torch.std(lm_head_weight, dim=-1)).item()

                # Compute root mean square (RMS) row-wise
                rms_values = root_mean_square(lm_head_weight)
                mean_rms_4 = torch.mean(rms_values).item()  # Mean of RMS values
                std_rms_5 = torch.std(rms_values).item()    # Standard deviation of RMS values

                # Write directly to CSV (efficient memory usage)
                writer.writerow([checkpoint, norm_mean_1, abs_mean_2, std_mean_3, mean_rms_4, std_rms_5])

    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    output_dir = "./output"  # Adjust this path as needed
    analyze_checkpoints(output_dir)
