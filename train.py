import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import load_checkpoint, save_checkpoint, check_accuracy, \
                  save_predictions_as_imgs, get_loaders, \
                  set_deterministic, set_all_seeds


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser(description='Hyperparameters for training')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--random-seed', type=int, default=123, help='batch size')
    parser.add_argument('--num-classes', type=int, default=10, help='batch size')
    parser.add_argument('--image-height', type=int, default=160, help='image size (height)')
    parser.add_argument('--image-width', type=int, default=240, help='image size (width)')
    parser.add_argument('--load-model', action='store_true', help='load pre-trained model')
    parser.add_argument('--train-img-dir', type=str, default="data/train_images/", help='train image directory')
    parser.add_argument('--val-img-dir', type=str, default="data/val_images/", help='validation image directory')
    parser.add_argument('--train-mask-dir', type=str, default="data/train_masks/", help='train mask directory')
    parser.add_argument('--val-mask-dir', type=str, default="data/val_masks/", help='validation mask directory')
    config = parser.parse_args()
    return config


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, epoch, num_epochs, device):
    """ One forward pass of U-NET """
    
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (image, mask) in enumerate(loop):
        
        image = image.to(device)
        mask = mask.float().unsqueeze(1).to(device)

        with torch.cuda.amp.autocast():
            output = model(image)
            loss = loss_fn(output, mask)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
   

def main(config):
    """ Training of U-NET """
    
    set_deterministic
    set_all_seeds(config.random_seed)
    
    train_transform = A.Compose(
        [
            A.Resize(height=config.image_height, width=config.image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=config.image_height, width=config.image_width),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(config.device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_loader, val_loader = get_loaders(config.train_img_dir, config.train_mask_dir, config.val_img_dir, config.val_mask_dir, config.batch_size, train_transform, val_transforms, config.num_workers, True)
    
    if config.load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    check_accuracy(val_loader, model, device=config.device)
    scaler = torch.cuda.amp.GradScaler()
    
    
    for epoch in range(config.num_epochs):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, epoch, config.num_epochs, config.device)

        # Save model
        checkpoint = {"state_dict": model.state_dict(), "optimizer":optimizer.state_dict()}
        save_checkpoint(checkpoint)

        # Check Accuracy
        check_accuracy(val_loader, model, device=config.device)

        # Print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=config.device)
    
    
if __name__ == "__main__":
    config = parse_opt()
    main(config)