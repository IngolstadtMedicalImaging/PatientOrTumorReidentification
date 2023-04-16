import argparse
import datetime
import os

from augmentations import *


def create_dir(dir: str, name: str) -> str:
    date = datetime.datetime.now().strftime('%d_%m_%Y-%H_%M')
    directory = os.path.join(dir, f'{name}_{date}')
    os.makedirs(directory, exist_ok=True)
    return directory


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    false = {"off", "false", "0"}
    true = {"on", "true", "1"}
    if s.lower() in false:
        return False
    elif s.lower() in true:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
    

def parse_augmentations(args):
    train, test = [], []

    for augmentation in args.augmentations_test.split(","):
        test.append(eval(augmentation))

    for augmentation in args.augmentations_train.split(","):
        train.append(eval(augmentation))

    return CustomCompose(train), CustomCompose(test)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('--augmentations_train', type=str, default='ToTensor(),Greyscale(),HorizontalFlip(),Normalize()')
    parser.add_argument('--augmentations_test', type=str, default='ToTensor(),Greyscale(),Normalize()')
    args = parser.parse_args()

    parse_augmentations(args)

if __name__ == '__main__':
    main()