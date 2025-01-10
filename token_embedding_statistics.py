import os
import torch
import csv
from transformers import AutoModelForCausalLM

def root_mean_square(tensor):
    """Computes the root mean square (RMS) of each row."""
    return torch.sqrt(torch.mean(tensor ** 2, dim=-1))

def analyze_checkpoints(output_dir, csv_filename="token_embeddings_statistics.csv"):
    """
    Iterate over all checkpoint folders in the output directory, load models properly with AutoModelForCausalLM,
    compute statistics for `lm_head.weight`, and save results into a CSV file.
    """
    checkpoint_dirs = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split('-')[-1])  # Sort numerically by checkpoint number
    )

    # Open CSV file for writing (efficient memory usage)
    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Checkpoint", "Global Mean Bias", "Per-token Mean Bias", "Token Embedding Variability", "Mean of RMS", "Std of RMS"])  # CSV Header

        for checkpoint in checkpoint_dirs:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            
            # Load model using AutoModelForCausalLM (ensures lm_head.weight is not discarded)
            try:
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float32, device_map="cpu")
            except Exception as e:
                print(f"Skipping {checkpoint} (failed to load model): {e}")
                continue

            # Ensure lm_head exists
            if not hasattr(model, "lm_head"):
                print(f"Skipping {checkpoint} (lm_head not found)")
                continue

            lm_head_weight = model.lm_head.weight.detach().cpu()

            # Compute requested statistics
            norm_mean_1 = torch.norm(torch.mean(lm_head_weight, dim=0)).item()
            abs_mean_2 = torch.mean(torch.abs(torch.mean(lm_head_weight, dim=-1))).item()
            std_mean_3 = torch.mean(torch.std(lm_head_weight, dim=-1)).item()

            # Compute root mean square (RMS) row-wise
            rms_values = root_mean_square(lm_head_weight)
            mean_rms_4 = torch.mean(rms_values).item()  # Mean of RMS values
            std_rms_5 = torch.std(rms_values).item()    # Standard deviation of RMS values

            # Write results to CSV
            writer.writerow([checkpoint, norm_mean_1, abs_mean_2, std_mean_3, mean_rms_4, std_rms_5])

            # Free memory
            del model
            torch.cuda.empty_cache()

    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    output_dir = "./output_base"  # Adjust this path as needed
    analyze_checkpoints(output_dir)
