import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.metrics import precision_recall_curve, auc

# ================= 配置 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '/workspace/best_nail_unet.pth'
IMG_PATH = '/workspace/PythonProject/train1/images/1.jpg'
GT_MASK_PATH = '/workspace/PythonProject/train1/masks/1.jpg'
SAVE_RESULT_PATH = '/workspace/PythonProject/result.png'
IMG_SIZE = (256, 256)
THRESHOLD = 0.5

# ==================== 预处理 ====================
def apply_clahe(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

def extract_hsv_channels(img_np):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    s_ch = hsv[:, :, 1:2] / 255.0
    v_ch = hsv[:, :, 2:3] / 255.0
    return s_ch, v_ch

def extract_smoothness(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    local_var = np.abs(gray - blur)
    smoothness = 1.0 - np.clip(local_var * 3.0, 0, 1)
    return smoothness[:, :, np.newaxis]

def extract_highlight(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    base = cv2.GaussianBlur(gray, (21, 21), 5.0)
    highlight = np.clip(gray - base + 0.3, 0, 1)
    return highlight[:, :, np.newaxis]

def preprocess_image(img_pil, img_size):
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

# ================= 原始模型结构 =================
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

class AttentionBlock(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

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

# ================= 评价指标计算 =================
def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()
    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))
    eps = 1e-7

    iou = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    return iou, dice, precision, recall

# ================= 可视化对比图 =================
def visualize_compare(img, gt, pred, save_path="compare.png"):
    overlay = img.copy().astype(np.float32) / 255.0
    correct = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    overlay[correct] = [0, 1, 0]
    overlay[fp] = [1, 0, 0]
    overlay[fn] = [0, 0, 1]

    plt.figure(figsize=(20, 5))
    plt.subplot(1,4,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
    plt.subplot(1,4,2); plt.imshow(gt, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1,4,3); plt.imshow(pred, cmap="gray"); plt.title("Prediction"); plt.axis("off")
    plt.subplot(1,4,4); plt.imshow(overlay); plt.title("Error(G=Correct,R=FP,B=FN)"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ================= PR曲线 =================
def plot_pr_curve(gt_256, prob_256, save_path="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(gt_256.flatten(), prob_256.flatten())
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'AUC={pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ================= 阈值曲线 =================
def plot_threshold_curve(prob_256, gt_256, save_path="threshold_curve.png"):
    ths = np.linspace(0.1, 0.9, 50)
    ious = []
    for t in ths:
        pred = (prob_256 > t).astype(np.uint8)
        iou, _, _, _ = compute_metrics(pred, gt_256)
        ious.append(iou)
    plt.figure()
    plt.plot(ths, ious, label='IoU', color='r')
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    plt.title('IoU vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ================= 预处理 & 后处理 =================
def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    original_size = img.size[::-1]
    tensor = preprocess_image(img, IMG_SIZE).unsqueeze(0)
    return tensor, img, original_size

def postprocess_mask(mask_prob, original_shape, threshold=THRESHOLD):
    raw_mask = (mask_prob > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    result = cv2.resize(cleaned, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (result > 127).astype(np.float32)

# ================= 主程序 =================
def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ 模型不存在")
        return
    if not os.path.exists(GT_MASK_PATH):
        print("❌ 真实掩码不存在")
        return

    model = UNetV2(in_ch=10, out_ch=1).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("🚀 模型加载成功！")

    with torch.no_grad():
        x, original, orig_size = preprocess(IMG_PATH)
        x = x.to(DEVICE)
        out = model(x)
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
        mask = postprocess_mask(prob, orig_size)

    gt_256 = np.array(Image.open(GT_MASK_PATH).convert('L').resize((256,256)))
    gt_256 = (gt_256 > 127).astype(np.float32)
    
    gt_pil = Image.open(GT_MASK_PATH).convert('L').resize((mask.shape[1], mask.shape[0]), Image.NEAREST)
    gt_mask = (np.array(gt_pil) > 127).astype(np.float32)

    iou, dice, precision, recall = compute_metrics(mask, gt_mask)

    print("\n==================== 分割评价指标 ====================")
    print(f"IoU         : {iou:.4f}")
    print(f"Dice        : {dice:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print("=======================================================")

    visualize_compare(np.array(original), gt_mask, mask, save_path="/workspace/PythonProject/compare.png")
    plot_pr_curve(gt_256, prob, save_path="/workspace/PythonProject/pr_curve.png")
    plot_threshold_curve(prob, gt_256, save_path="/workspace/PythonProject/threshold_curve.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Segmentation"); axes[1].axis("off")
    overlay = np.array(original.resize((mask.shape[1], mask.shape[0]))) / 255.0
    overlay[mask > 0.5] = [1, 0, 0]
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(SAVE_RESULT_PATH, dpi=150)

    print("✅ 全部结果已保存完成！")

if __name__ == '__main__':
    main()