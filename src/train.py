import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from metrics import calculate_psnr, calculate_ssim

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

        for steps, (lr_img, hr_img) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
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

        for steps, (lr_img, hr_img) in enumerate(tqdm(val_dataset, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")):
            if steps >= val_steps:
                break

            sr_img = model(lr_img, training=False)
            loss = criterion(hr_img, sr_img)

            val_loss += loss.numpy()
            val_batches += 1

            psnr_scores.append(tf.reduce_mean(calculate_psnr(sr_img, hr_img)).numpy())
            ssim_scores.append(tf.reduce_mean(calculate_ssim(sr_img, hr_img)).numpy())

        avg_val_loss = val_loss / val_batches
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)

        print(f"Epoch {epoch+1}: L1 loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}, PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")

        history.append({
            'epoch': epoch+1,
            'L2 loss': avg_train_loss,
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
