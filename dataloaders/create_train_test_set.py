import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

COLOR_PATH = '../color'
NORMALS_PATH = '../normals'

COLOR_PRODUCTION_PATH = '../color_production'
NORMALS_PRODUCTION_PATH = '../normals_production'

TRAIN_TEST_RATIO = 0.1

def main():
    color_files = list(sorted(os.listdir(f'{COLOR_PATH}')))
    normals_files = list(sorted(os.listdir(f'{NORMALS_PATH}')))

    color_production_files = list(sorted(os.listdir(f'{COLOR_PRODUCTION_PATH}')))
    normals_production_files = list(sorted(os.listdir(f'{NORMALS_PRODUCTION_PATH}')))

    # Note: zip will take the minimum length
    print(f'[SIZES] Color: {len(color_files)} | Normals: {len(normals_files)} ')
    color_and_normals = zip(color_files, normals_files)
    color, normals = list(zip(*color_and_normals))

    for x, y in tqdm(zip(color, normals), desc='Checking if the zipped filenames are the same...'):
        x = x.strip('.png')
        y = y.strip('.npz')
        assert x == y, f'Wrong files: {x} | {y}'


    print(f'[SIZES] Color production: {len(color_production_files)} | Normals production: {len(normals_production_files)} ')
    color_and_normals_production = zip(color_production_files, normals_production_files)
    color_production, normals_production = list(zip(*color_and_normals_production))

    for x, y in tqdm(zip(color_production, normals_production), desc='Checking if the zipped filenames are the same...'):
        x = x.strip('.png').split('_')
        y = y.strip('.npz').split('_')
        assert x[-1] == y[-1], f'Wrong files: {x} | {y}'

    color_train, color_test, normals_train, normals_test = train_test_split(
        color, normals, test_size=TRAIN_TEST_RATIO, random_state=42)

    print(f'[FINAL SIZES]: color_train: {len(color_train)} | color_test: {len(color_test)} | normals_train: {len(normals_train)} | normals_test: {len(normals_test)}')

    color_production_train, color_production_test, normals_production_train, normals_production_test = train_test_split(
        color_production, normals_production, test_size=TRAIN_TEST_RATIO, random_state=42)

    print(f'[FINAL SIZES]: color_production_train: {len(color_production_train)} | color_production_test: {len(color_production_test)} \
    | normals_production_train: {len(normals_production_train)} | normals_production_test: {len(normals_production_test)}')

    color_normals_train_df = pd.DataFrame(zip(color_train, normals_train),  columns=['color_train', 'normals_train'])
    color_normals_train_df.to_csv('color_normals_train_df.csv')
    print(color_normals_train_df.head())
    print('\n' + '=' * 10 + '\n')

    color_normals_test_df = pd.DataFrame(zip(color_test, normals_test),  columns=['color_test', 'normals_test'])
    color_normals_test_df.to_csv('color_normals_test_df.csv')
    print(color_normals_test_df.head())
    print('\n' + '=' * 10 + '\n')

    color_production_normals_train_df = pd.DataFrame(zip(color_production_train, normals_production_train),  columns=['color_production_train', 'normals_production_train'])
    color_production_normals_train_df.to_csv('color_production_normals_train_df.csv')
    print(color_production_normals_train_df.head())
    print('\n' + '=' * 10 + '\n')

    color_production_normals_test_df = pd.DataFrame(zip(color_production_test, normals_production_test),  columns=['color_production_test', 'normals_production_test'])
    color_production_normals_test_df.to_csv('color_production_normals_test_df.csv')
    print(color_production_normals_test_df.head())
    print('\n' + '=' * 10 + '\n')

if __name__ == '__main__':
    main()