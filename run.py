import sys
import data_augmentation
import training
import prediction
import mask_to_submission
import viewer
import argparse

from config import *

def main(show_train_graph=False):
	print("Running data augmentation...")
	data_augmentation.main()

	print("Running training...")
	training.main(show_train_graph)

	print("Running prediction...")
	prediction.main()

	print("Converting masks to submission format...")
	mask_to_submission.main()

	print("Running viewer...")
	viewer.main()

	print("All steps completed successfully!")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Performs data augmentation, training and prediction",
        usage="python3 run.py [--prediction | -p] [--custom_prediction | -cp] [--show_train_graph | -stg] [--help]"
    )
	parser.add_argument(
		"--prediction", "-p",
		action="store_true",
		help="prediction only"
	)
	parser.add_argument(
		"--custom_prediction", "-cp",
		action="store_true",
		help="prediction only (use the custom test set)"
	)
	parser.add_argument(
		"--show_train_graph", "-stg",
		action="store_true",
		help="displays the train history graph after training"
	)
	args = parser.parse_args()

	if args.prediction:
		prediction.main()
		viewer.main()
	elif args.custom_prediction:
		prediction.main(CUSTOM_TEST_IMAGES_DIR)
		viewer.main(CUSTOM_TEST_IMAGES_DIR)
	else:
		main(args.show_train_graph)