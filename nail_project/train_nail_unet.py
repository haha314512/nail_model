import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import cv2
import warnings
import random
from typing import Tuple

warnings.filterwarnings('ignore')

# ====================== 配置区 ======================
class Config:
    IMG_DIR = "/workspace/PythonProject/train/images"
    MASK_DIR = "/workspace/PythonProject/train/masks"
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    MODEL_SAVE_PATH = 'best_nail_unet.pth'
    VAL_SPLIT = 0.2
    THRESHOLD = 0.50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    EARLY_STOP_PATIENCE = 30

random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
torch.cuda.manual_seed_all(Config.SEED)

# ==================== 预处理 ====================
def apply_clahe(img_np: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

def extract_hsv_channels(img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    s_ch = hsv[:, :, 1:2] / 255.0
    v_ch = hsv[:, :, 2:3] / 255.0
    return s_ch, v_ch

def extract_smoothness(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    local_var = np.abs(gray - blur)
    smoothness = 1.0 - np.clip(local_var * 3.0, 0, 1)
    return smoothness[:, :, np.newaxis]

def extract_highlight(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    base = cv2.GaussianBlur(gray, (21, 21), 5.0)
    highlight = np.clip(gray - base + 0.3, 0, 1)
    return highlight[:, :, np.newaxis]

def preprocess_image(img_pil: Image.Image, img_size: Tuple[int, int], is_train: bool = True) -> torch.Tensor:
    img_np = np.array(img_pil.resize(img_size, Image.BILINEAR))
    rgb = img_np.astype(np.float32) / 255.0
    clahe_rgb = apply_clahe(img_np).astype(np.float32) / 255.0
    s_ch, v_ch = extract_hsv_channels(img_np)
    smooth = extract_smoothness(img_np)
    highlight = extract_highlight(img_np)

    multi_ch = np.concatenate([rgb, clahe_rgb, s_ch, v_ch, smooth, highlight], axis=2)
    tensor = torch.from_numpy(multi_ch.transpose(2, 0, 1)).float()

    tensor[:3] = (tensor[:3] - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                 torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor[3:6] = (tensor[3:6] - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                  torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor

# ================= 数据集 =================
class NailDatasetV2(Dataset):
    def __init__(self, img_paths: list, mask_paths: list, img_size: Tuple[int, int], is_train: bool = True):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.is_train and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img_tensor = preprocess_image(img, self.img_size, is_train=self.is_train)
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()

        return img_tensor, mask

# ================= 边缘检测 =================
class MultiScaleEdgeDetection(nn.Module):
    def __init__(self, in_channels: int = 10):
        super().__init__()
        laplacian = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        self.lap_conv = nn.Conv2d(in_channels, 1, 3, padding=1, bias=False)
        lap_kernel = laplacian.unsqueeze(0).unsqueeze(0).repeat(1, in_channels, 1, 1)
        self.lap_conv.weight = nn.Parameter(lap_kernel, requires_grad=False)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x_conv = nn.Conv2d(in_channels, 1, 3, padding=1, bias=False)
        self.sobel_y_conv = nn.Conv2d(in_channels, 1, 3, padding=1, bias=False)
        sx_k = sobel_x.unsqueeze(0).unsqueeze(0).repeat(1, in_channels, 1, 1)
        sy_k = sobel_y.unsqueeze(0).unsqueeze(0).repeat(1, in_channels, 1, 1)
        self.sobel_x_conv.weight = nn.Parameter(sx_k, requires_grad=False)
        self.sobel_y_conv.weight = nn.Parameter(sy_k, requires_grad=False)

        self.fuse = nn.Sequential(
            nn.Conv2d(3, 2, 1), nn.ReLU(inplace=True), nn.Conv2d(2, 1, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e_lap = torch.abs(self.lap_conv(x))
        e_sx = torch.abs(self.sobel_x_conv(x))
        e_sy = torch.abs(self.sobel_y_conv(x))
        edge_combined = torch.cat([e_lap, e_sx, e_sy], dim=1)
        return self.fuse(edge_combined)

# ================= 注意力 =================
class AttentionBlock(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

# ================= 卷积块 =================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

# ================= UNetV2 =================
class UNetV2(nn.Module):
    def __init__(self, in_ch=10, out_ch=1):
        super().__init__()
        self.edge = MultiScaleEdgeDetection(in_ch)
        self.down1 = ConvBlock(in_ch+1, 64, 0.05)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(64, 128, 0.1)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvBlock(128, 256, 0.1)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = ConvBlock(256, 512, 0.1)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024, 0.15)

        self.up4 = nn.ConvTranspose2d(1024,512,2,2)
        self.att4 = AttentionBlock(512,512,256)
        self.upconv4 = ConvBlock(1024,512,0.1)

        self.up3 = nn.ConvTranspose2d(512,256,2,2)
        self.att3 = AttentionBlock(256,256,128)
        self.upconv3 = ConvBlock(512,256,0.1)

        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.att2 = AttentionBlock(128,128,64)
        self.upconv2 = ConvBlock(256,128,0.05)

        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.att1 = AttentionBlock(64,64,32)
        self.upconv1 = ConvBlock(128,64,0.05)

        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e = self.edge(x)
        x = torch.cat([x,e],1)
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        b = self.bottleneck(self.pool4(d4))

        u4 = self.up4(b)
        u4 = torch.cat([self.att4(u4,d4),d4],1)
        u4 = self.upconv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([self.att3(u3,d3),d3],1)
        u3 = self.upconv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([self.att2(u2,d2),d2],1)
        u2 = self.upconv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([self.att1(u1,d1),d1],1)
        u1 = self.upconv1(u1)

        return self.final(u1)

# ====================== 损失函数 ======================
class DiceFocalWithLogitsLoss(nn.Module):
    def __init__(self, smooth=1e-6, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.smooth = smooth
        self.gamma = focal_gamma
        self.alpha = focal_alpha

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma * bce_loss).mean()

        inputs_sig = torch.sigmoid(inputs)
        intersection = (inputs_sig * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_sig.sum() + targets.sum() + self.smooth)
        return dice_loss + focal_loss

class BoundaryLoss(nn.Module):
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight
    def compute_edge(self, x):
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],device=x.device).float().view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],device=x.device).float().view(1,1,3,3)
        return torch.clamp(torch.abs(nn.functional.conv2d(x,sobel_x,padding=1)) + torch.abs(nn.functional.conv2d(x,sobel_y,padding=1)),0,1)
    def forward(self, inputs, targets):
        inputs_sig = torch.sigmoid(inputs)
        return self.weight * nn.functional.mse_loss(self.compute_edge(inputs_sig), self.compute_edge(targets))

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_focal = DiceFocalWithLogitsLoss()
        self.boundary = BoundaryLoss()
    def forward(self, inputs, targets):
        return 0.8 * self.dice_focal(inputs, targets) + 0.2 * self.boundary(inputs, targets)

def calculate_metrics(pred, target, th=0.5):
    pred = (torch.sigmoid(pred) > th).float()
    intersection = (pred*target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection/(union+1e-6)
    dice = (2*intersection)/(pred.sum()+target.sum()+1e-6)
    return iou.item(), dice.item()

# ================= 训练 =================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

@torch.no_grad()
def val_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss=0
    ious,dices=[],[]
    for imgs,masks in loader:
        imgs,masks=imgs.to(device),masks.to(device)
        outputs=model(imgs)
        total_loss+=criterion(outputs,masks).item()
        iou,dice=calculate_metrics(outputs,masks)
        ious.append(iou)
        dices.append(dice)
    return total_loss/len(loader), np.mean(ious), np.mean(dices)

# ================= 主函数 =================
def main():
    
    img_paths = sorted([p for e in ['*.png','*.jpg','*.jpeg'] for p in glob.glob(os.path.join(Config.IMG_DIR,e))])
    mask_paths = sorted([p for e in ['*.png','*.jpg','*.jpeg'] for p in glob.glob(os.path.join(Config.MASK_DIR,e))])
    dataset = list(zip(img_paths,mask_paths))
    val_size = int(len(dataset)*Config.VAL_SPLIT)
    train_set, val_set = random_split(dataset,[len(dataset)-val_size,val_size], torch.Generator().manual_seed(Config.SEED))

    train_loader = DataLoader(NailDatasetV2([x[0]for x in train_set],[x[1]for x in train_set],Config.IMG_SIZE,True), Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(NailDatasetV2([x[0]for x in val_set],[x[1]for x in val_set],Config.IMG_SIZE,False), Config.BATCH_SIZE, shuffle=False, num_workers=2)

    model = UNetV2(10,1).to(Config.DEVICE)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)

    best_iou=0
    patience=0
    print("🔥 训练开始")
    for e in range(Config.EPOCHS):
        tl = train_one_epoch(model,train_loader,criterion,optimizer,Config.DEVICE)
        vl,vi,vd = val_one_epoch(model,val_loader,criterion,Config.DEVICE)
        scheduler.step(vi)
        print(f"Epoch {e+1:3d} | Train {tl:.4f} | Val {vl:.4f} | IoU {vi:.4f} | Dice {vd:.4f}")

        if vi>best_iou:
            best_iou=vi
            patience=0
            torch.save({'model_state_dict':model.state_dict(),'iou':best_iou}, Config.MODEL_SAVE_PATH)
            print(f"✅ 保存最佳模型 IoU={best_iou:.4f}")
        else:
            patience+=1
            if patience>=Config.EARLY_STOP_PATIENCE:
                print("\n⏹️ 早停")
                break
    print(f"\n🎉 完成 最佳IoU={best_iou:.4f}")

if __name__ == '__main__':
    main()