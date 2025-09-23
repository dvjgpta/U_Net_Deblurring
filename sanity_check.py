import torch
from unet_model import *
from data_loader import *
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataloader, test_dataloader = get_dataloaders(
    blurred_dir=train_dir_blurred,
    sharp_dir=train_dir_sharp,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    batch_size=64
)
# Initialize a tiny model
model_0 = UNet(n_channels=3, n_classes=3).to(device)

# Loss & optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model_0.parameters(), lr=1e-4)

# Take one batch from train and test
train_batch = next(iter(train_dataloader))
test_batch  = next(iter(test_dataloader))

X_train, y_train = train_batch
X_test, y_test   = test_batch

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test   = X_test.to(device), y_test.to(device)

# Forward pass on train batch
model_0.train()
y_pred_train = model_0(X_train)
train_loss = loss_fn(y_pred_train, y_train).item()

# Forward pass on test batch
model_0.eval()
with torch.inference_mode():
    y_pred_test = model_0(X_test)
    test_loss = loss_fn(y_pred_test, y_test).item()

print(f"Sanity check → Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
