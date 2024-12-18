[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

# Road Segmentation on Satellite Images

This project aims to classify roads on satellite images from Google Maps in suburban areas of the USA. We use a U-Net model to perform road segmentation, assigning each pixel a label: `road=1` or `background=0`.

The dataset is sourced from the [AIcrowd EPFL ML Road Segmentation challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files).

---

## Project Overview

The key steps in the project include:
1. **Data Augmentation**: Expanding the training set with rotated and cropped patches of size 256x256, followed by color adjustments.
2. **Model Training**: Training a U-Net model with the augmented dataset.
3. **Prediction**: Generating prediction masks for test images.
4. **Visualization**: Overlaying test images with predicted road masks for evaluation.
5. **Submission**: Generating a CSV file from prediction masks for submission.

---

## Code Structure

- **`data/`**  
  Contains the training set, test set, augmented images, and prediction masks.
  
- **`saved_models/`**  
  Directory for storing the trained U-Net model (`unet.keras`).

- **`requirements.txt`**  
  Defines all of the requirements to run the code.

- **`config.py`**  
  Defines all of the constants used.
  
- **`data_augmentation.py`**  
  Expands the training dataset with rotations, cropping, and color adjustments.
  
- **`model.py`**  
  Defines a 5-layer U-Net model and outputs `unet.keras`.
  
- **`training.py`**  
  Trains the U-Net model using augmented data.
  
- **`prediction.py`**  
  Creates prediction masks for the test set.
  
- **`viewer.py`**  
  Displays test images overlaid with prediction masks.
  
- **`mask_to_submissions.py`**  
  Converts prediction masks to a `submission.csv` file.
  
- **`run.py`**  
  Executes the entire pipeline, including training and predictions.
  
- **`report.pdf`**  
  Documents findings and project analysis.

---

## Project Usage

### Clone the Repository
```bash
git clone https://github.com/CS-433/ml-project-2-monkeylearning.git
cd ml-project-2-monkeylearning
```

### Download the dataset
Download the dataset from the [AIcrowd website](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files) and place it in the `data/` directory.

### Create a Python Environment
Set up a virtual environment and install the required dependencies:

```bash
Copy code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Best Model
Execute the run.py script to perform the full pipeline of the best model:
```bash
python3 run.py
```