import sys
import a_data_augmentation
import b_fit
import c_predict
import mask_to_submission
import d_viewer

from config import *

def main():
    print("Running data augmentation...")
    a_data_augmentation.main()

    print("Running training...")
    b_fit.main()

    print("Running prediction...")
    c_predict.main()

    print("Converting masks to submission format...")
    mask_to_submission.main()

    print("Running viewer...")
    d_viewer.main()

    print("All steps completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "custom_prediction" or sys.argv[1] == "cp":
            c_predict.main(CUSTOM_TEST_IMAGES_DIR)
            d_viewer.main(CUSTOM_TEST_IMAGES_DIR)
        elif sys.argv[1] == "prediction" or sys.argv[1] == "p":
            c_predict.main()
            d_viewer.main()
    else:        
        main()
