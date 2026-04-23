"""
Prepare the data layout expected by main_eeg_course.py.

The script creates symlinks under:
  data/things-eeg/Preprocessed_data/sub-01/
  data/things-eeg/Image_set/
"""
import argparse
import os


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def relink(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.unlink(dst)
    os.symlink(os.path.abspath(src), dst)
    print(f"{dst} -> {src}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        required=True,
        help="Course image-eeg-data directory on the target machine.",
    )
    parser.add_argument(
        "--repo_data_dir",
        default=os.path.join(REPO_DIR, "data", "things-eeg"),
        help="Destination data/things-eeg directory inside this repo.",
    )
    args = parser.parse_args()

    preprocessed = os.path.join(
        args.data_root,
        "converted_for_cogcappro",
        "ThingsEEG",
        "Preprocessed_data_250Hz_whiten",
        "sub-01",
    )
    relink(
        os.path.join(preprocessed, "train.pt"),
        os.path.join(args.repo_data_dir, "Preprocessed_data", "sub-01", "train.pt"),
    )
    relink(
        os.path.join(preprocessed, "test.pt"),
        os.path.join(args.repo_data_dir, "Preprocessed_data", "sub-01", "test.pt"),
    )
    relink(
        os.path.join(args.data_root, "training_images"),
        os.path.join(args.repo_data_dir, "Image_set", "train_images"),
    )
    relink(
        os.path.join(args.data_root, "test_images"),
        os.path.join(args.repo_data_dir, "Image_set", "test_images"),
    )


if __name__ == "__main__":
    main()
