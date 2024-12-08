import sys

# Import the scripts as modules
import a_data_augmentation
import b_fit
import c_predict
import mask_to_submission
import d_viewer

def main():
    try:
        # Call each script's main function
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
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
