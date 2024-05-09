import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision import transforms
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torchsummary import summary

from configs.model import UNet
from configs.dataset import PALMDataset
from configs import config

#####

if __name__ == "__main__":
    img_paths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    mask_paths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

    imgs_train, imgs_test, masks_train, masks_test = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)
    imgs_train, imgs_val, masks_train, masks_val = train_test_split(imgs_train, masks_train, test_size=0.25, random_state=42)

    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_ds = PALMDataset(imgs_train, masks_train, trans)
    val_ds = PALMDataset(imgs_val, masks_val, trans)
    # test_ds = PALMDataset(imgs_test, masks_test)

    # calculate steps per epoch for datasets
    train_steps = len(train_ds) // config.BATCH_SIZE
    val_steps = len(val_ds) // config.BATCH_SIZE
    # test_steps = len(test_ds) // config.BATCH_SIZE

    # dataloaders
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=config.BATCH_SIZE)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=config.BATCH_SIZE)
    # test_loader = DataLoader(test_ds, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

    # training
    model = UNet(config.NUM_CHANNELS, config.NUM_CLASSES).to(config.DEVICE)
    # summary(model, (3, 512, 512))
    
    loss_f = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config.INIT_LR)

    # initialize history dictionary
    H = {"train_loss": [], "val_loss": []}

    print("[INFO] training the network...")
    start_time = time.time()

    for e in tqdm(range(config.NUM_EPOCHS)):
        # training mode
        model.train()

        # initialize losses
        train_loss = 0
        val_loss = 0

        for (i, (x, y)) in enumerate(train_loader):
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # perform forward pass
            pred = model(x)
            loss = loss_f(pred, y)

            # perform back propagation and optimize weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

        # eval mode
        with torch.no_grad():
            model.eval()

            for (i, (x, y)) in enumerate(val_loader):
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                pred = model(x)
                loss = loss_f(pred, y)
                val_loss += loss

       # calculate average losses
        avr_train_loss = train_loss / train_steps 
        avr_val_loss = val_loss / val_steps

        #update training hostory
        H["train_loss"].append(avr_train_loss.cpu().detach().numpy())
        H["val_loss"].append(avr_val_loss.cpu().detach().numpy())

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Validation loss: {:.4f}".format(avr_train_loss, avr_val_loss))

# display the total time needed to perform the training
end_time = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)

# serialize the model to disk
torch.save(model, config.MODEL_PATH)


