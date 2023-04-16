import argparse
import math
import os
import sys
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from augmentations import *
from dataset import SinglePatch
from model import get_model
from utils import create_dir, parse_augmentations
from model import load_pretrained_model
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-88623a1c-c660-5e58-8722-cd90a957deee"

def eval_one_epoch(model, criterion, loader, device, writer: SummaryWriter, epoch, n_valid_classes, n_train_classes) -> dict:
    model.eval()
    outputs = {'normal': [],
               'argmax': [],
               'targets': []}
    print("\n----------\nValid\n----------")
    with torch.no_grad():
        epoch_loss, epoch_accuracy_single, epoch_accuracy_majority = .0, .0, .0
        for ix, (images, targets) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)

            output = model(images)
            output_argmax = output.argmax(dim=-1)

            outputs['normal'].append(output)
            outputs['argmax'].append(output_argmax)
            outputs['targets'].append(targets)
            
            epoch_loss += criterion(output, targets).item()
            epoch_accuracy_single += (output_argmax == targets).float().sum() / len(targets)

            output = output.detach().cpu()

    epoch_loss /= len(loader)
    epoch_accuracy_single /= len(loader)
    epoch_accuracy_single *= 100

    all_argmax = torch.cat(outputs['argmax']).to(device='cpu')
    all_targets = torch.cat(outputs['targets']).to(device='cpu')

    confmat = confusion_matrix(y_pred=all_argmax, y_true=all_targets, labels=[patient_id for patient_id in range(n_train_classes)])
    confmat_argmax = np.argmax(confmat, axis=1)
    
    for ix, id in enumerate(confmat_argmax):
        if ix == id:
            epoch_accuracy_majority += 1

    epoch_accuracy_majority /= n_valid_classes
    epoch_accuracy_majority *= 100

    print('\nVal/Loss: ', epoch_loss)
    print('Val/Acc-Single: ', f"{epoch_accuracy_single}%")
    print('Val/Acc-Major: ', f"{epoch_accuracy_majority}%")

    writer.add_scalar('Val/Loss', epoch_loss, epoch)
    writer.add_scalar('Val/Acc-Single', epoch_accuracy_single, epoch)
    writer.add_scalar('Val/Acc-Major:', epoch_accuracy_majority, epoch)


def train_one_epoch(model, criterion, optimizer, loader, device, writer, epoch) -> None:
    model.train()
    epoch_loss, epoch_accuracy = .0, .0
    print("\n----------\nTrain\n----------")
    for ix, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        output = model(images)
        output_argmax = output.argmax(dim=-1)

        epoch_accuracy += (output_argmax == targets).float().sum() / len(targets)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        epoch_loss += loss.item()
        output = output.detach().cpu()

    epoch_accuracy /= len(loader)
    epoch_accuracy *= 100
    epoch_loss /= len(loader)

    print('\nTrain/Loss: ', epoch_loss)
    print('Train/Accuracy: ', f"{epoch_accuracy}%")

    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Acc', epoch_accuracy, epoch)


def train_model(args) -> None:
    ckp_dir = create_dir(args.ckp_dir, f"{args.arch}")
    writer = SummaryWriter(log_dir=os.path.join(ckp_dir, 'runs'))
    transform_train, transform_val = parse_augmentations(args)

    data_train = SinglePatch(data_csv=args.csv, slide_level=args.slide_level, patch_imgsize=args.patch_imgsize, mode='train', transform=transform_train, pseudo_epoch_length=args.pseudo_epoch_length)
    data_val = SinglePatch(data_csv=args.csv, slide_level=args.slide_level, patch_imgsize=args.patch_imgsize, mode='valid', transform=transform_val, patches_valid=args.patches_valid)

    n_train_classes = data_train.get_num_classes()
    n_valid_classes = data_val.get_num_classes()

    loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
    loader_val = DataLoader(data_val, batch_size=args.batch_size_val, shuffle=True, drop_last=True, num_workers=8)

    model = get_model(args, n_classes=n_train_classes, pretrained=args.pretrained_model)

    if args.weights_dir != 'none':
        model = load_pretrained_model(args, n_train_classes)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.momentum, args.alpha),
                                  weight_decay=args.weight_decay)

    model.to(device)

    writer.add_hparams(hparam_dict=args.__dict__, metric_dict=dict())

    for epoch in range(1, args.n_epochs +1):
        print(f'\n------------------\nEpoch[{epoch:>3d}/{args.n_epochs:>3d}]:')
        
        train_one_epoch(model, criterion, optimizer, loader_train, device, writer, epoch)
        data_train.resample_patients()

        if epoch % args.eval_epoch == 0 or epoch == args.n_epochs:
            eval_one_epoch(model, criterion, loader_val, device, writer, epoch, n_valid_classes, n_train_classes)
            torch.save(model.state_dict(), os.path.join(ckp_dir, f'{args.arch}_{epoch}.pth'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/Meningeome_1000_anon')
    parser.add_argument('--ckp_dir', type=str, default='scripts/single-resnet/checkpoints')
    parser.add_argument('--arch', type=str, default='resnet50-pretraining',
                        choices=['resnet50-pretraining', 'resnet18-pretraining'])
    parser.add_argument('--weights_dir', type=str, default='none')

    # Augmentations
    parser.add_argument('--augmentations_train', type=str, default='ToTensor(),ColorJitter(),RandomVerticalFlip(),RandomHorizontalFlip(),GaussianBlur(),Normalize()')
    parser.add_argument('--augmentations_test', type=str, default='ToTensor(),Normalize()')

    # Trainingsparameter
    # parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--alpha', type=float, default=.999)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    
    parser.add_argument('--gamma', type=float, default=0.9)
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_val', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=5000)
    parser.add_argument('--pseudo_epoch_length', type=int, default=1024)
    parser.add_argument('--patches_valid', type=int, default=50)

    parser.add_argument('--csv', type=str, default="data/final/reid_patches.csv")
    parser.add_argument('--slide_level', type=int, default=1)
    parser.add_argument('--patch_imgsize', type=int, default=512)

    parser.add_argument('--eval_epoch', type=int, default=20)
    parser.add_argument('--pretrained_model', type=bool, default=True)
    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()