import argparse
import math
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from augmentations import *
from dataset import MultiPatch
from model import get_model
from utils import create_dir, bool_flag, parse_augmentations
from torchvision.utils import make_grid
from model import load_pretrained_model

os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-88623a1c-c660-5e58-8722-cd90a957deee"

def eval_one_epoch(model, criterion, loader, device, writer: SummaryWriter, epoch) -> dict:
    model.eval()
    print("\n----------\nValid\n----------")
    with torch.no_grad():
        epoch_loss, epoch_accuracy = .0, .0
        for ix, (images, targets) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)
            # print(f'\nTargets Test: {targets}')

            output = model(images)
            output_argmax = output.argmax(dim=-1)
            
            epoch_loss += criterion(output, targets).item()
            epoch_accuracy += (output_argmax == targets).float().sum() / len(targets)

            # print((output_argmax == targets))
            # print((output_argmax == targets).float().sum())
            # print(epoch_accuracy)

            output = output.detach().cpu()

    epoch_loss /= len(loader)
    epoch_accuracy /= len(loader)
    epoch_accuracy *= 100

    print('\nVal/Loss: ', epoch_loss)
    print('Val/Acc: ', f"{epoch_accuracy}%")

    writer.add_scalar('Val/Loss', epoch_loss, epoch)
    writer.add_scalar('Val/Acc', epoch_accuracy, epoch)

    return {'Epoch_Loss': epoch_loss, 'Epoch_Acc': epoch_accuracy}


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

    data_train = MultiPatch(data_csv=args.csv, slide_level=args.slide_level, patch_imgsize=args.patch_imgsize, mode='train', transform=transform_train, pseudo_epoch_length=args.pseudo_epoch_length, bag_size=args.bag_size)
    data_val = MultiPatch(data_csv=args.csv, slide_level=args.slide_level, patch_imgsize=args.patch_imgsize, mode='valid', transform=transform_val, pseudo_epoch_length=args.pseudo_epoch_length, bag_size=args.bag_size)

    n_classes = data_train.get_num_classes()

    loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    loader_val = DataLoader(data_val, batch_size=args.batch_size_val, shuffle=True, drop_last=True, num_workers=4)

    model = get_model(args, n_classes=n_classes, pretrained=args.pretrained_model)

    if args.weights_dir != 'none':
        model = load_pretrained_model(args, n_classes)

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
            eval_one_epoch(model, criterion, loader_val, device, writer, epoch)
            data_val.resample_patients()
            torch.save(model.state_dict(), os.path.join(ckp_dir, f'{args.arch}_{epoch}.pth'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/Meningeome_1000_anon')
    parser.add_argument('--ckp_dir', type=str, default='mil-resnet/checkpoints')
    parser.add_argument('--arch', type=str, default='mil-resnet50',
                        choices=['mil-resnet18', 'mil-resnet50', 'ilse'])
    parser.add_argument('--weights_dir', type=str, default='none')
    parser.add_argument('--pretrained_backbone', type=str, default='single-resnet/checkpoints/resnet50-pretraining_31_03_2023-16_52/resnet50-pretraining_5000.pth')
    
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
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size_val', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=5000)
    parser.add_argument('--pseudo_epoch_length', type=int, default=512)

    parser.add_argument('--csv', type=str, default="data/final/reid_patches.csv")
    parser.add_argument('--slide_level', type=int, default=1)
    parser.add_argument('--patch_imgsize', type=int, default=512)

    parser.add_argument('--eval_epoch', type=int, default=10)
    parser.add_argument('--pretrained_model', type=bool, default=True)

    parser.add_argument('--bag_size', type=int, default=50)
    parser.add_argument('--attention_size', type=int, default=128)
    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()