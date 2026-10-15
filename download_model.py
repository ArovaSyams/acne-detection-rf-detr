import os
import modal


def main():
    print("Connecting to Modal Volume 'acne-model-storage'...")
    vol = modal.Volume.from_name("acne-model-storage")

    os.makedirs("try", exist_ok=True)

    # 1. Download Best Model Checkpoint
    cloud_model_path = "output/checkpoint_best_total.pth"
    local_model_path = "try/best_rfdetr_acne.pth"

    print(f"\nDownloading {cloud_model_path}...")
    try:
        file_generator = vol.read_file(cloud_model_path)
        with open(local_model_path, "wb") as f:
            for chunk in file_generator:
                f.write(chunk)
        print(f"Successfully downloaded model to: {local_model_path}")
    except Exception as e:
        print(f"Failed to download model checkpoint. Error: {e}")

    # 2. Download Training Metrics Log
    cloud_metrics_path = "output/metrics.csv"
    local_metrics_path = "try/metrics_acne.csv"

    print(f"\nDownloading {cloud_metrics_path}...")
    try:
        file_generator = vol.read_file(cloud_metrics_path)
        with open(local_metrics_path, "wb") as f:
            for chunk in file_generator:
                f.write(chunk)
        print(f"Successfully downloaded metrics to: {local_metrics_path}")
    except Exception as e:
        print(f"Failed to download metrics log. Error: {e}")


if __name__ == "__main__":
    main()