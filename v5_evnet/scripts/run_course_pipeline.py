"""
One-command course pipeline:
1. Create local data symlinks.
2. Generate multi-blur RN50 image features if missing.
3. Train the EEG encoder.
"""
import argparse
import os
import subprocess
import sys


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to image-eeg-data.")
    parser.add_argument("--clip_checkpoint", required=True, help="Path to OpenCLIP RN50 .bin checkpoint.")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--first_seed", type=int, default=21)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    run([sys.executable, "scripts/prepare_course_data.py", "--data_root", args.data_root])

    feature_dir = os.path.join(REPO_DIR, "output", "Image_feature")
    log_dir = os.path.join(REPO_DIR, "output", "logs", "main_eeg_course")
    train_feature = os.path.join(feature_dir, "MultiBlur_RN50_train.pt")
    test_feature = os.path.join(feature_dir, "MultiBlur_RN50_test.pt")
    if not args.skip_features and not (os.path.exists(train_feature) and os.path.exists(test_feature)):
        run([
            sys.executable,
            "preprocess/process_image_course.py",
            "--clip_checkpoint",
            args.clip_checkpoint,
            "--save_dir",
            feature_dir,
        ])
    else:
        print("Skip feature generation: feature files already exist or --skip_features was set.")

    if not args.skip_train:
        run([
            sys.executable,
            "main_eeg_course.py",
            "--epoch",
            str(args.epoch),
            "--train_batch_size",
            str(args.train_batch_size),
            "--lr",
            str(args.lr),
            "--n_seeds",
            str(args.n_seeds),
            "--first_seed",
            str(args.first_seed),
            "--feature_path",
            feature_dir,
            "--output_dir",
            log_dir,
        ])


if __name__ == "__main__":
    main()
