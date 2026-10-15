import modal
import subprocess

# 1. Define container image and mount local project files to cloud environment
image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install required system dependencies for OpenCV and image operations
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "numpy<2.0.0",
        "torch",
        "torchvision",
        "rfdetr[train,loggers]>=1.4.0",
        "roboflow",
        "pycocotools==2.0.7",
        "opencv-python-headless",
        "pyyaml",
        "albumentations"
    )
    .add_local_dir("./dataset_coco", remote_path="/root/dataset_coco")
    .add_local_file("./config.yaml", remote_path="/root/config.yaml")
    .add_local_file("./train.py", remote_path="/root/train.py")
)

# 2. Initialize Modal App
app = modal.App("rfdetr-acne-training", image=image)

# 3. Create or attach persistent cloud volume for checkpoints and artifacts
storage_volume = modal.Volume.from_name("acne-model-storage", create_if_missing=True)

# 4. Define Cloud Function and GPU Resources
@app.function(
    gpu="A10G",
    timeout=86400,
    volumes={"/root/output_model": storage_volume}
)
def train_on_cloud():
    print("GPU instance initialized. Executing train.py...")

    process = subprocess.run(["python", "/root/train.py"], capture_output=False)

    if process.returncode == 0:
        print("Training completed successfully! Committing outputs to Modal Volume...")
        storage_volume.commit()
    else:
        print("Training encountered an error. Check the execution logs above.")

# 5. Local Terminal Entrypoint
@app.local_entrypoint()
def main():
    print("Packaging environment and dataset, triggering remote training on Modal...")
    train_on_cloud.remote()