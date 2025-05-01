"""
PACE training and testing script.
"""

# === Standard library ===
import os
import random
from argparse import Namespace
from math import pi

# === Third-party ===
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torchvision.transforms as transforms
from augment import image_augment
# === Local modules ===
from config import parser
from datasets import Image, load_dataset
from model import PACE, ViTImageClassifier
from numpy.linalg import det, inv
from scipy.special import gammaln, psi
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
# === Transformers ===
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments
from utils import (Adam, DictImageDataset, MyImageDataset, accuracy_score,
                   compute_metrics, dirichlet_expectation, posterior_mu_sigma,
                   read_tsv_file, setup_seed)


class PACEEarlyStoppingCallback(EarlyStoppingCallback):
    """Custom early-stopping callback that also snapshots PACE parameters."""

    def __init__(
        self,
        pace_args: Namespace,
        model: torch.nn.Module,
        early_stopping_patience: int = 1,
        early_stopping_threshold: float = 0.0,
    ):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.epochs = 0
        self.pace_args = pace_args
        self.model = model
        #  self.flag = True

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Check the configured metric, apply early stopping, and save model +
        PACE state whenever evaluation runs.
        """
        metric_key = args.metric_for_best_model
        if not metric_key.startswith("eval_"):
            metric_key = f"eval_{metric_key}"
        metric_value = metrics.get(metric_key)

        # Standard patience check
        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

        self.epochs += 1
        # self.flag = True

        # Save model + PACE snapshots
        if PACE is None:
            return
        save_dir = self.pace_args.save_path
        torch.save(
            self.model.state_dict(),
            os.path.join(save_dir, f"{self.pace_args.task}_epoch{self.epochs}.pt"),
        )
        np.save(
            os.path.join(save_dir, f"{self.pace_args.task}_mus-epoch{self.epochs}.npy"),
            PACE._mus,
        )
        np.save(
            os.path.join(save_dir, f"{self.pace_args.task}_sigmas-epoch{self.epochs}.npy"),
            PACE._sigmas,
        )
        np.save(
            os.path.join(save_dir, f"{self.pace_args.task}_eta-epoch{self.epochs}.npy"),
            PACE._eta,
        )


class PACETrainer(Trainer):
    """Trainer subclass with custom loss to integrate PACE EM steps."""
    
    def __init__(self, pace_args, *args, **kwargs):
        """
        pace_args: the namespace returned by parser.parse_args()
        *args, **kwargs: everything else that Trainer needs
        """
        super().__init__(*args, **kwargs)
        self.pace_args = pace_args
        
    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        enc = inputs["encodings"]
        logits, states, att = model(enc)

        # Augmented pass
        aug = image_augment(enc)
        logits_t, states_t, att_t = model(aug)

        # Base ViT loss
        criterion = torch.nn.CrossEntropyLoss()
        vit_loss = criterion(logits, inputs["labels"]).sum()

        # One-time flag reset
        # if PACEcallback.flag:
        #     PACEcallback.flag = False

        if PACE is not None and not return_outputs:
            # PACE E-step on augmented
            gamma_t, phi_t = PACE.do_e_step(states_t, att_t[self.pace_args.layer + 1])
            PACE._phi_trans = PACE._phi

            # PACE EM-step on original
            probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
            gamma, phi = PACE.do_em_step(
                states, att[self.pace_args.layer + 1], cl=True, y=probs
            )
            PACE.update_eta(logits)
            loss = PACE._delta
        else:
            loss = vit_loss

        if return_outputs:
            preds = torch.argmax(logits, dim=-1)
            return loss, {"label_ids": inputs["labels"], "predictions": preds}

        return loss


def main():
    # --- Parse args and prepare output dir ---
    pace_args = parser.parse_args()
    pace_args.save_path = os.path.normpath(os.path.join(pace_args.save_path, pace_args.name))
    os.makedirs(pace_args.save_path, exist_ok=True)

    # --- Seeding ---
    setup_seed(pace_args.seed)

    # 1. Load full dataset
    dataset = load_dataset("imagefolder", data_dir="../dataset/Color")

    # 2. Cast image column to PIL
    dataset = dataset.cast_column("image", Image())

    # 3. Print to verify keys (they should include 'image' and 'label')
    print("Example keys before transform:", dataset["train"][0].keys())  # should be image + label

    # 4. Split first — before adding transform
    splits = dataset["train"].train_test_split(test_size=0.2, seed=pace_args.seed)
    train_ds = splits["train"].cast_column("image", Image())
    test_ds  = splits["test"].cast_column("image", Image())

    # 5. Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x[:3, ...]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def make_encodings(example):
        img = example["image"].convert("RGB")
        return {
            "encodings": transform(img),
            "labels": example["label"],
        }

    dataset = dataset.map(
        make_encodings,
        remove_columns=["image", "label"],  # drop raw columns if you like
        batched=False,
    )

    # 4) Split into train/test
    splits = dataset["train"].train_test_split(test_size=0.2, seed=pace_args.seed)
    train_ds = splits["train"]
    test_ds  = splits["test"]

    # 5) Tell HuggingFace to return PyTorch tensors
    train_ds.set_format(type="torch", columns=["encodings", "labels"])
    test_ds .set_format(type="torch", columns=["encodings", "labels"])
    
    # --- Model & PACE init ---
    ViT = ViTImageClassifier(
        in_dim=pace_args.b_dim, 
        out_dim=pace_args.out_dim, 
        hid_dim=pace_args.c_dim, 
        layer=pace_args.layer
    ).cuda()
    if not pace_args.require_grad:
        ViT.backbone.requires_grad_(False)
        
    global PACE
    PACE = (
        PACE(d=pace_args.c_dim, 
             K=pace_args.K, 
             D=pace_args.D, 
             N=pace_args.N, 
             alpha=pace_args.alpha,
             C=pace_args.out_dim)
        if "PACE" in pace_args.name
        else None
    )

    # --- Trainer setup ---
    training_args = TrainingArguments(
        output_dir="../results",
        num_train_epochs=pace_args.num_epochs,
        per_device_train_batch_size=pace_args.train_batch_size,
        per_device_eval_batch_size=pace_args.eval_batch_size,
        warmup_steps=0,
        weight_decay=pace_args.weight_decay,
        logging_dir="./logs",
        logging_steps=10,
        seed=pace_args.seed,
        load_best_model_at_end=True,
        metric_for_best_model=pace_args.metric,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=pace_args.lr,
    )

    global PACEcallback
    PACEcallback = PACEEarlyStoppingCallback(pace_args=pace_args, 
                                           model=ViT,
                                           early_stopping_patience=10,)

    trainer = PACETrainer(
        pace_args=pace_args,
        model=ViT,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[PACEcallback],
        # data_collator=default_data_collator,
    )

    # --- Train or eval ---
    print(f"Train size: {len(train_ds)}, Eval size: {len(test_ds)}")

    if pace_args.train:
        print("Training...")
        if not pace_args.require_grad:
            ckpt = os.path.join("..", "ckpt", "ViT-base", f"{pace_args.task}_epoch5.pt")
            ViT.load_state_dict(torch.load(ckpt))
        trainer.train()
        torch.save(
            ViT.state_dict(),
            os.path.join(pace_args.save_path, f"{pace_args.task}_epoch{pace_args.num_epochs}.pt"),
        )
    else:
        print("Evaluating...")
        ckpt = os.path.join(pace_args.save_path, f"{pace_args.task}_epoch{pace_args.num_epochs}.pt")
        ViT.load_state_dict(torch.load(ckpt))

    # --- Save or load final PACE parameters ---
    if PACE is not None:
        mus_path = os.path.join(
            pace_args.save_path, f"{pace_args.task}_mus-epoch{pace_args.num_epochs}.npy"
        )
        sig_path = os.path.join(
            pace_args.save_path, f"{pace_args.task}_sigmas-epoch{pace_args.num_epochs}.npy"
        )
        eta_path = os.path.join(
            pace_args.save_path, f"{pace_args.task}_eta-epoch{pace_args.num_epochs}.npy"
        )

        if pace_args.train:
            np.save(mus_path, PACE._mus)
            np.save(sig_path, PACE._sigmas)
            np.save(eta_path, PACE._eta)
        else:
            PACE._mus = np.load(mus_path)
            PACE._sigmas = np.load(sig_path)
            PACE._eta = np.load(eta_path)

            # Explain ViT with PACE
            print("PACE is explaining ViT...")
            test_loader = DataLoader(test_ds, batch_size=pace_args.eval_batch_size, shuffle=False)
            for inputs in test_loader:
                enc = inputs["encodings"].cuda()
                logits, states, att = ViT(enc)
                gamma, phi = PACE.do_e_step(states, att[pace_args.layer + 1])
                E_log_theta = dirichlet_expectation(gamma)


if __name__ == "__main__":
    main()
