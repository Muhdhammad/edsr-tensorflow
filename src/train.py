import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import yaml
from src.model import edsr
from src.utils import calculate_psnr, calculate_ssim, rgb2y, shave_borders
from data.dataset import create_data_splits, create_datasets

def train_model(model, train_dataset, val_dataset, num_epochs, optimizer, criterion,
                steps_per_epoch, val_steps, reduce_lr_wait, early_stop_patience,
                epsilon=1e-4, early_stopping=True, checkpoints_dir='checkpoints'):
    
    # To save trained model and training logs
    os.makedirs(checkpoints_dir, exist_ok=True)

    history = []
    best_psnr = 0.0
    best_ssim = 0.0
    wait = 0

    for epoch in range(num_epochs):
        
        # ===== Training =====
        current_lr = optimizer.learning_rate.numpy()
        train_loss = 0
        num_batches = 0

        for steps, (hr_img, lr_img) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            if steps >= steps_per_epoch:
                break

            with tf.GradientTape() as tape:
                sr_img = model(lr_img, training=True)
                loss = criterion(hr_img, sr_img)

            grads = tape.gradient(loss, model.trainable_variables)  
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss += loss.numpy()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # ===== Validation =====
        val_loss = 0
        val_batches = 0
        psnr_scores = []
        ssim_scores = []

        for steps, (hr_img, lr_img) in enumerate(tqdm(val_dataset, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")):
            if steps >= val_steps:
                break

            sr_img = model(lr_img, training=False)
            loss = criterion(hr_img, sr_img)

            sr_img_y = rgb2y(sr_img)
            hr_img_y = rgb2y(hr_img)

            sr_img_y = shave_borders(sr_img_y, 4)
            hr_img_y = shave_borders(hr_img_y, 4)

            val_loss += loss.numpy()
            val_batches += 1

            psnr_scores.append(tf.reduce_mean(calculate_psnr(sr_img_y, hr_img_y)).numpy())
            ssim_scores.append(tf.reduce_mean(calculate_ssim(sr_img_y, hr_img_y)).numpy())

        avg_val_loss = val_loss / val_batches
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)

        print(f"Epoch {epoch+1}: L1 loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}, PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")

        history.append({
            'epoch': epoch+1,
            'L1 loss': avg_train_loss,
            'Val loss': avg_val_loss,
            'PSNR': avg_psnr,
            'SSIM': avg_ssim
        })

        # ===== Checkpoints =====
        psnr_improved = avg_psnr > best_psnr + epsilon
        if psnr_improved:
            best_psnr = avg_psnr
            best_ssim = avg_ssim
            wait = 0
            model.save(os.path.join(checkpoints_dir,"best_sr_image.keras"))
            print("New best model saved.")
        else:
            wait += 1
            print(f"No PSNR or SSIM improved for {wait} epochs")

            if wait == reduce_lr_wait:
                new_lr = current_lr * 0.1
                optimizer.learning_rate.assign(new_lr)
                print(f"LR reduced to {new_lr}")
                
            if early_stopping and wait >= early_stop_patience:
                print("Early Stopping triggered")
                break

    # ===== Final save =====
    final_model_path = os.path.join(checkpoints_dir, "final_sr_model.keras")
    model.save(final_model_path)
    print(f"Final model saved - {final_model_path}")

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(checkpoints_dir, "training_log.csv"), index=False)
    print("Training log saved to training_log.csv")

if __name__ == "__main__":

    root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(root, "config", "config.yaml")

    # ===== Load config =====
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ===== Dataset =====
    root = config["data"]["hr_images_path"]
    train_path = os.path.join(root, "DIV2K_train_HR")
    val_path = os.path.join(root, "DIV2K_valid_HR")

    train_img_path = create_data_splits(train_path)
    val_img_path = create_data_splits(val_path)

    train_dataset, val_dataset, train_steps, val_steps = create_datasets(
        train_path=train_img_path,
        val_path=val_img_path,
        hr_size=config["data"]["hr_size"],
        lr_size=config["data"]["lr_size"],
        batch_size=config["data"]["batch_size"],
    )

    # ===== Model =====
    model = edsr(num_channels=config["model"]["num_channels"],
                 scale_factor=config["model"]["scale_factor"],
                 num_blocks=config["model"]["num_blocks"]
                 )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"])
    criterion = tf.keras.losses.MeanAbsoluteError()
    
    # ====== Training =====
    train_model(model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_epochs=config["training"]["num_epochs"],
                optimizer=optimizer,
                criterion=criterion,
                steps_per_epoch=train_steps,
                val_steps=val_steps,
                reduce_lr_wait=config["training"]["reduce_lr_wait"],
                early_stop_patience=config["training"]["early_stop_patience"],
                checkpoints_dir=config["training"]["checkpoints_dir"]
                )