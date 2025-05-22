import os
import argparse
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess NIH Chest X-ray data")
    parser.add_argument('--raw_csv', type=str, required=True, help='Path to Data_Entry_2017.csv')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to raw image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save resized images and splits')
    parser.add_argument('--image_size', type=int, default=224, help='Resize images to this size (square)')
    parser.add_argument('--test_split', type=float, default=0.15, help='Test set proportion')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation set proportion (of remaining data)')
    return parser.parse_args()

def resize_and_save_image(img_path, save_path, size=(224, 224)):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size)
        img.save(save_path)
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

def preprocess(args):
    # Step 1: Load and filter metadata
    df = pd.read_csv(args.raw_csv)
    pneumonia_df = df[df['Finding Labels'] == 'Pneumonia'].copy()
    normal_df = df[df['Finding Labels'] == 'No Finding'].copy()

    pneumonia_df['Label'] = 1
    normal_df['Label'] = 0

    combined_df = pd.concat([pneumonia_df, normal_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Step 2: Resize and save images
    resized_dir = os.path.join(args.output_dir, 'images_resized')
    os.makedirs(resized_dir, exist_ok=True)

    print("Resizing and saving images...")
    for _, row in tqdm(combined_df.iterrows(), total=len(combined_df)):
        src = os.path.join(args.image_dir, row['Image Index'])
        dst = os.path.join(resized_dir, row['Image Index'])
        resize_and_save_image(src, dst, size=(args.image_size, args.image_size))

    combined_df['Image Path'] = combined_df['Image Index'].apply(lambda x: os.path.join(resized_dir, x))

    # Step 3: Split dataset
    train_val, test = train_test_split(
        combined_df, test_size=args.test_split, stratify=combined_df['Label'], random_state=42
    )
    train, val = train_test_split(
        train_val, test_size=args.val_split / (1 - args.test_split), stratify=train_val['Label'], random_state=42
    )

    # Step 4: Save splits
    split_dir = os.path.join(args.output_dir, 'splits')
    os.makedirs(split_dir, exist_ok=True)

    train.to_csv(os.path.join(split_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(split_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(split_dir, 'test.csv'), index=False)

    print("Preprocessing complete. Files saved to:")
    print(f"- {split_dir}/train.csv")
    print(f"- {split_dir}/val.csv")
    print(f"- {split_dir}/test.csv")
    print(f"- Resized images: {resized_dir}/")

if __name__ == "__main__":
    args = parse_args()
    preprocess(args)