import os
import yaml
from rfdetr import RFDETRLarge

def main():
    print("Loading configuration from /root/config.yaml...")

    with open("/root/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    output_dir = "/root/output_model"
    os.makedirs(output_dir, exist_ok=True)

    os.chdir(output_dir)

    print(f"Initializing model architecture: {config['model']['architecture']}")
    model = RFDETRLarge()

    print("Starting RF-DETR-L training...")

    result = model.train(
        dataset_dir=config['dataset']['img_dir'],
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        grad_accum_steps=config['training']['grad_accum_steps'],
        early_stopping=config['training']['early_stopping'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        # aug_config=config['augmentation'],
        resolution=config['training']['resolution'],
        lr=config['training']['lr'],
        lr_encoder=config['training']['lr_encoder'],
        weight_decay=config['training']['weight_decay'],
        seed=config['training']['seed'],
        threshold=config['training']['threshold']
    )

    print("Training complete! Model artifacts saved to output_model directory.")

if __name__ == "__main__":
    main()