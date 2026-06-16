import os
import cv2
import math
import random
import numpy as np

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from PIL import Image

################################################################################
# CONFIG
################################################################################
ROI_SIZE = 224

VALID_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
]

################################################################################
# IMAGE HELPERS
################################################################################
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot load image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def yolo_to_xyxy(box, img_w, img_h):
    cls, cx, cy, bw, bh = box
    cx *= img_w
    cy *= img_h
    bw *= img_w
    bh *= img_h
    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)
    return x1, y1, x2, y2

def crop_roi(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = yolo_to_xyxy(
        bbox,
        w,
        h
    )
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    return roi

################################################################################
# LABEL READER
################################################################################
def read_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            boxes.append(
                [cls, cx, cy, bw, bh]
            )

    return boxes

################################################################################
# ROI COLLECTION
################################################################################
def collect_roi_metadata(dataset_dir):
    dataset_dir = Path(dataset_dir)
    images = []
    for ext in VALID_EXTENSIONS:
        images.extend(
            dataset_dir.glob(f"*{ext}")
        )

    images = sorted(images)
    samples = []
    for image_path in images:
        label_path = image_path.with_suffix(".txt")
        boxes = read_yolo_labels(label_path)
        for box_id, box in enumerate(boxes):
            samples.append(
                {
                    "image_path": str(image_path),
                    "bbox": box,
                    "box_id": box_id
                }
            )

    return samples

################################################################################
# AUGMENTATIONS
################################################################################
class RandomJPEGCompression:
    def __call__(self, image):
        quality = random.randint(10, 60)
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            quality
        ]
        image_bgr = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )
        _, encimg = cv2.imencode(
            ".jpg",
            image_bgr,
            encode_param
        )
        decimg = cv2.imdecode(
            encimg,
            cv2.IMREAD_COLOR
        )
        decimg = cv2.cvtColor(
            decimg,
            cv2.COLOR_BGR2RGB
        )
        return Image.fromarray(decimg)

class RandomOcclusion:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        img = np.array(image)
        h, w = img.shape[:2]
        occ_w = random.randint(
            int(0.1 * w),
            int(0.4 * w)
        )
        occ_h = random.randint(
            int(0.1 * h),
            int(0.4 * h)
        )
        x = random.randint(0, w - occ_w)
        y = random.randint(0, h - occ_h)
        img[
            y:y+occ_h,
            x:x+occ_w
        ] = 0
        return Image.fromarray(img)

################################################################################
# PATCHCVT AUGMENTATION PIPELINE
################################################################################
def build_degradation_transform():
    return T.Compose([
        T.RandomApply([
            T.GaussianBlur(7)
        ], p=0.5),
        T.RandomApply([
            T.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.2
            )
        ], p=0.7),
        RandomJPEGCompression(),
        RandomOcclusion(p=0.5),
        T.RandomPerspective(
            distortion_scale=0.3,
            p=0.3
        )
    ])

################################################################################
# BASE ROI DATASET
################################################################################
class BaseROIDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 roi_size=224):
        self.samples = collect_roi_metadata(
            dataset_dir
        )
        self.roi_size = roi_size
        self.image_transform = T.Compose([
            T.Resize(
                (roi_size, roi_size)
            ),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def get_roi(self, idx):
        sample = self.samples[idx]
        image = load_image(
            sample["image_path"]
        )
        roi = crop_roi(
            image,
            sample["bbox"]
        )
        if roi is None:

            roi = np.zeros(
                (
                    self.roi_size,
                    self.roi_size,
                    3
                ),
                dtype=np.uint8
            )

        roi = Image.fromarray(roi)
        return roi

################################################################################
# MEMORY BANK DATASET
################################################################################
class MemoryBankDataset(BaseROIDataset):
    def __getitem__(self, idx):
        roi = self.get_roi(idx)
        roi = self.image_transform(roi)
        return roi

################################################################################
# PATCHCVT TRAIN DATASET
################################################################################
class PatchCvTDataset(BaseROIDataset):
    def __init__(self,
                 dataset_dir,
                 roi_size=224):
        super().__init__(
            dataset_dir,
            roi_size
        )
        self.degradation_transform = \
            build_degradation_transform()

    def __getitem__(self, idx):
        roi = self.get_roi(idx)
        clean = self.image_transform(
            roi
        )
        degraded = self.degradation_transform(
            roi.copy()
        )
        degraded = self.image_transform(
            degraded
        )
        return {

            "clean": clean,
            "degraded": degraded
        }

################################################################################
# DATALOADER FACTORIES
################################################################################
def build_memory_dataloader(
    dataset_dir,
    batch_size=32,
    num_workers=4
):
    ds = MemoryBankDataset(
        dataset_dir
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

def build_patchcvt_dataloader(
    dataset_dir,
    batch_size=16,
    shuffle=True,
    num_workers=4
):
    ds = PatchCvTDataset(
        dataset_dir
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
   
#PatchCvT core architecture:   
#Frozen ResNet18
#↓
#MemoryBankBuilder
#↓
#Patch Retrieval
#↓
#Token Construction
#↓
#Transformer
#↓
#Robustness Head
#↓
#Semantic Gate
#↓
#Preservation Head

################################################################################
# IMPORTS
################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

################################################################################
# FROZEN RESNET18 BACKBONE
################################################################################
class FrozenResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


################################################################################
# MEMORY BANK
################################################################################
class MemoryBankBuilder:
    """
    Builds frozen semantic memory from
    representative clean ROIs.
    """
    def __init__(
        self,
        feature_dim=128,
        memory_size=512
    ):
        self.feature_dim = feature_dim
        self.memory_size = memory_size

    @torch.no_grad()
    def build(
        self,
        backbone,
        dataloader,
        device
    ):
        print("Building memory bank...")
        backbone.eval()
        all_features = []
        for batch in dataloader:
            batch = batch.to(device)
            fmap = backbone(batch)
            B, C, H, W = fmap.shape
            patches = (
                fmap
                .permute(0,2,3,1)
                .reshape(-1,C)
            )
            all_features.append(
                patches.cpu()
            )

        all_features = torch.cat(
            all_features,
            dim=0
        )
        print(
            "Collected patches:",
            len(all_features)
        )

        ####################################################################
        # Random subset
        #
        # v1
        #
        # Later:
        # PatchCore coreset
        ####################################################################
        if len(all_features) > self.memory_size:
            idx = torch.randperm(
                len(all_features)
            )[:self.memory_size]
            all_features = all_features[idx]
        memory_bank = F.normalize(
            all_features,
            dim=1
        )
        print(
            "Memory bank:",
            memory_bank.shape
        )
        return memory_bank


################################################################################
# MEMORY RETRIEVAL
################################################################################
class MemoryRetrieval(nn.Module):
    def __init__(
        self,
        tau=0.1
    ):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        q,
        memory_bank
    ):
        ####################################################################
        # q
        #
        # [B,N,C]
        #
        ####################################################################
        distances = torch.cdist(
            q,
            memory_bank.unsqueeze(0)
        )
        weights = torch.softmax(
            -distances / self.tau,
            dim=-1
        )
        q_tilde = torch.matmul(
            weights,
            memory_bank
        )
        return q_tilde


################################################################################
# TOKEN CONSTRUCTION
################################################################################
class SemanticTokenBuilder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        q_tilde
    ):
        ####################################################################
        # residual
        ####################################################################
        r = q - q_tilde

        ####################################################################
        # activation energy
        ####################################################################
        a = torch.norm(
            q,
            dim=-1,
            keepdim=True
        )

        ####################################################################
        # Eq.13
        ####################################################################
        z = torch.cat(
            [
                q,
                q_tilde,
                r,
                a
            ],
            dim=-1
        )
        return z, r, a


################################################################################
# TRANSFORMER ENCODER
################################################################################
class PatchCvTTransformer(nn.Module):
    def __init__(
        self,
        input_dim=385,
        d_model=256,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        ####################################################################
        # token projection
        ####################################################################
        self.proj = nn.Linear(
            input_dim,
            d_model
        )

        ####################################################################
        # transformer
        ####################################################################
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(
        self,
        tokens
    ):
        tokens = self.proj(tokens)
        tokens = self.encoder(tokens)
        return tokens

################################################################################
# ROBUSTNESS HEAD
################################################################################
class RobustnessHead(nn.Module):
    def __init__(
        self,
        d_model=256
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                d_model,
                128
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                1
            ),
            nn.Sigmoid()
        )

    def forward(
        self,
        z
    ):
        return self.net(z)


################################################################################
# PRESERVATION HEAD
################################################################################
class PreservationHead(nn.Module):
    def __init__(
        self,
        d_model=256
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                d_model,
                128
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                1
            ),
            nn.Sigmoid()
        )

    def forward(
        self,
        embedding
    ):
        return self.net(
            embedding
        )


################################################################################
# PATCHCVT
################################################################################
class PatchCvT(nn.Module):
    def __init__(
        self,
        memory_bank
    ):
        super().__init__()

        ####################################################################
        # frozen backbone
        ####################################################################
        self.backbone = FrozenResNet18()

        ####################################################################
        # frozen memory
        ####################################################################
        self.register_buffer(
            "memory_bank",
            memory_bank
        )

        ####################################################################
        # modules
        ####################################################################
        self.retrieval = MemoryRetrieval()
        self.token_builder = (
            SemanticTokenBuilder()
        )
        self.transformer = (
            PatchCvTTransformer()
        )
        self.robustness_head = (
            RobustnessHead()
        )
        self.preservation_head = (
            PreservationHead()
        )

        ####################################################################
        # Eq.16
        ####################################################################
        self.Wr = nn.Parameter(
            torch.tensor(1.0)
        )
        self.We = nn.Parameter(
            torch.tensor(1.0)
        )

    def extract_patches(
        self,
        x
    ):
        fmap = self.backbone(x)
        B,C,H,W = fmap.shape
        patches = (
            fmap
            .permute(0,2,3,1)
            .reshape(B,H*W,C)
        )
        return patches

    def forward(
        self,
        x
    ):
        ####################################################################
        # patch embeddings
        ####################################################################
        q = self.extract_patches(x)

        ####################################################################
        # memory retrieval
        ####################################################################
        q_tilde = self.retrieval(
            q,
            self.memory_bank
        )

        ####################################################################
        # tokens
        ####################################################################
        z, r, a = self.token_builder(
            q,
            q_tilde
        )

        ####################################################################
        # transformer
        ####################################################################
        z_prime = self.transformer(
            z
        )

        ####################################################################
        # robustness
        ####################################################################
        robustness = (
            self.robustness_head(
                z_prime
            )
        )

        ####################################################################
        # residual magnitude
        ####################################################################
        residual_mag = torch.norm(
            r,
            dim=-1,
            keepdim=True
        )

        ####################################################################
        # Eq.16
        ####################################################################
        gate = torch.sigmoid(

            self.Wr * robustness

            -

            self.We * residual_mag
        )

        ####################################################################
        # Eq.17
        ####################################################################
        z_hat = gate * z_prime

        ####################################################################
        # global semantic representation
        ####################################################################
        embedding = z_hat.mean(
            dim=1
        )

        ####################################################################
        # preservation score
        ####################################################################
        preservation = (
            self.preservation_head(
                embedding
            )
        )

        return {
            "embedding": embedding,
            "preservation": preservation,
            "robustness": robustness,
            "gate": gate,
            "residual": residual_mag,
            "q": q,
            "q_tilde": q_tilde
        }
        
################################################################################
# LOSSES
################################################################################
class PatchCvTLoss(nn.Module):
    def __init__(
        self,
        lambda_semantic=1.0,
        lambda_robust=0.25,
        lambda_gate=0.25
    ):
        super().__init__()
        self.lambda_semantic = lambda_semantic
        self.lambda_robust = lambda_robust
        self.lambda_gate = lambda_gate

    def semantic_loss(
        self,
        clean_embedding,
        degraded_embedding
    ):
        cosine = F.cosine_similarity(
            clean_embedding,
            degraded_embedding,
            dim=1
        )
        return (1.0 - cosine).mean()

    def robustness_loss(
        self,
        robustness,
        residual
    ):
        target = torch.exp(
            -residual.detach()
        )
        return F.mse_loss(
            robustness,
            target
        )

    def gate_loss(
        self,
        gate,
        residual
    ):
        target = torch.exp(
            -residual.detach()
        )
        return F.l1_loss(
            gate,
            target
        )

    def forward(
        self,
        clean_out,
        degraded_out
    ):
        semantic = self.semantic_loss(
            clean_out["embedding"],
            degraded_out["embedding"]
        )
        robust = self.robustness_loss(
            degraded_out["robustness"],
            degraded_out["residual"]
        )

        gate = self.gate_loss(
            degraded_out["gate"],
            degraded_out["residual"]
        )

        total = (
            self.lambda_semantic * semantic
            +
            self.lambda_robust * robust
            +
            self.lambda_gate * gate
        )

        return {
            "total": total,
            "semantic": semantic,
            "robust": robust,
            "gate": gate
        }
        
################################################################################
# TRAIN STEP
################################################################################
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()
    running = {
        "total":0,
        "semantic":0,
        "robust":0,
        "gate":0
    }
    count = 0
    for batch in loader:
        clean = batch["clean"].to(device)
        degraded = batch["degraded"].to(device)
        optimizer.zero_grad()
        clean_out = model(clean)
        degraded_out = model(degraded)
        losses = criterion(
            clean_out,
            degraded_out
        )
        losses["total"].backward()
        optimizer.step()
        for k in running:
            running[k] += (
                losses[k].item()
            )

        count += 1

    for k in running:
        running[k] /= count

    return running
    
################################################################################
# VALIDATION
################################################################################
@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):
    model.eval()
    running = {
        "total":0,
        "semantic":0,
        "robust":0,
        "gate":0
    }
    count = 0
    for batch in loader:
        clean = batch["clean"].to(device)
        degraded = batch["degraded"].to(device)
        clean_out = model(clean)
        degraded_out = model(degraded)
        losses = criterion(
            clean_out,
            degraded_out
        )
        for k in running:
            running[k] += (
                losses[k].item()
            )
            
        count += 1
    for k in running:
        running[k] /= count

    return running
    
################################################################################
# CHECKPOINTS
################################################################################
def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metric
):
    torch.save(
        {
            "epoch":epoch,
            "metric":metric,
            "model_state":
                model.state_dict(),
            "optimizer_state":
                optimizer.state_dict()
        },
        path
    )

def load_checkpoint(
    path,
    model,
    optimizer=None
):
    ckpt = torch.load(
        path,
        map_location="cpu"
    )
    model.load_state_dict(
        ckpt["model_state"]
    )
    if optimizer is not None:
        optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )

    return ckpt
    
################################################################################
# INFERENCE
################################################################################
@torch.no_grad()
def predict_roi(
    model,
    roi_tensor
):
    model.eval()
    if roi_tensor.ndim == 3:
        roi_tensor = roi_tensor.unsqueeze(0)

    out = model(roi_tensor)
    preservation = (
        out["preservation"]
        .squeeze()
        .cpu()
        .item()
    )
    robustness = (
        out["robustness"]
        .mean()
        .cpu()
        .item()
    )
    residual = (
        out["residual"]
        .mean()
        .cpu()
        .item()
    )

    return {
        "preservation":
            preservation,
        "robustness":
            robustness,
        "residual":
            residual
    }
    
################################################################################
# TRAIN
################################################################################
def train_patchcvt(
    train_dir,
    test_dir,
    memory_dir,
    epochs=50,
    batch_size=16,
    lr=1e-4,
    device="cuda"
):

    ########################################################################
    # dataloaders
    ########################################################################
    memory_loader = (
        build_memory_dataloader(
            memory_dir,
            batch_size=batch_size
        )
    )

    train_loader = (
        build_patchcvt_dataloader(
            train_dir,
            batch_size=batch_size
        )
    )

    test_loader = (
        build_patchcvt_dataloader(
            test_dir,
            batch_size=batch_size,
            shuffle=False
        )
    )

    ########################################################################
    # build memory bank
    ########################################################################
    backbone = (
        FrozenResNet18()
        .to(device)
    )
    builder = MemoryBankBuilder(
        feature_dim=128,
        memory_size=512
    )

    memory_bank = builder.build(
        backbone,
        memory_loader,
        device
    )

    memory_bank = (
        memory_bank.to(device)
    )

    ########################################################################
    # model
    ########################################################################
    model = PatchCvT(
        memory_bank
    ).to(device)

    ########################################################################
    # optimizer
    ########################################################################
    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=lr,
        weight_decay=1e-4
    )

    ########################################################################
    # loss
    ########################################################################
    criterion = PatchCvTLoss()

    ########################################################################
    # training
    ########################################################################
    best_loss = 1e9
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        val_metrics = validate(
            model,
            test_loader,
            criterion,
            device
        )
        print()
        print(f"Epoch {epoch+1}/{epochs}")
        print("TRAIN", train_metrics)
        print("VALID", val_metrics)

        ####################################################################
        # save last
        ####################################################################
        save_checkpoint(
            "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics["total"]
        )

        ####################################################################
        # save best
        ####################################################################
        if val_metrics["total"] < best_loss:
            best_loss = (
                val_metrics["total"]
            )
            save_checkpoint(
                "best.pt",
                model,
                optimizer,
                epoch,
                best_loss
            )
            print("BEST MODEL SAVED")

    return model
    
################################################################################
# MAIN
################################################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory_dataset",
        type=str,
        required=True
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    args = parser.parse_args()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)
    model = train_patchcvt(
        train_dir=args.train_dataset,
        test_dir=args.test_dataset,
        memory_dir=args.memory_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )