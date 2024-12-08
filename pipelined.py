import subprocess

# Run data augmentation step
subprocess.run(["python", "1-data_augmentation.py"], check=True)

# Run the training step
subprocess.run(["python", "2-fit.py"], check=True)

# Run the prediction step
subprocess.run(["python", "3-predict.py"], check=True)

# Run the mask-to-submission conversion step
subprocess.run(["python", "mask_to_submission.py"], check=True)

print("All steps completed successfully!")
