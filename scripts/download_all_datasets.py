from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from data.downloaders.medmnist_data import download_medmnist, FLAGS


def main():
    parser = argparse.ArgumentParser(description='Download public MedMNIST reference images with official splits')
    parser.add_argument('--save_dir', default='./datasets')
    parser.add_argument('--datasets', nargs='+', choices=list(FLAGS), default=['xray'])
    parser.add_argument('--size', type=int, choices=[28, 64, 128, 224], default=128)
    parser.add_argument('--limit', type=int, default=None, help='Optional images per official split')
    args = parser.parse_args()
    for modality in args.datasets:
        count = download_medmnist(args.save_dir, modality, args.size, args.limit)
        print(f'{modality}: {count} references saved with official train/val/test splits')


if __name__ == '__main__':
    main()
