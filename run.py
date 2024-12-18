import sys
import data_augmentation
import training
import prediction
import mask_to_submission
import viewer

from config import *

def main():
    print("Running data augmentation...")
    data_augmentation.main()

    print("Running training...")
    training.main()

    print("Running prediction...")
    prediction.main()

    print("Converting masks to submission format...")
    mask_to_submission.main()

    print("Running viewer...")
    viewer.main()

    print("All steps completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "custom_prediction" or sys.argv[1] == "cp":
            prediction.main(CUSTOM_TEST_IMAGES_DIR)
            viewer.main(CUSTOM_TEST_IMAGES_DIR)
        elif sys.argv[1] == "prediction" or sys.argv[1] == "p":
            prediction.main()
            viewer.main()
    else:        
        main()
