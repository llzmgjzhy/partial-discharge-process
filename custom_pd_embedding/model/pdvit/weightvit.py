# pipeline to process partial discharge data
# conclude a vit model to learn a set of weights assigned to the classifier and the classifiers
import torch
from torch import nn
from einops import repeat
from custom_pd_embedding.model.vit.vit_base.vit import (
    Transformer,
    ViT,
)
from custom_pd_embedding.model.mlp.mlp import MLP
from custom_pd_embedding.model.resnet.resnet18.resnet import RESNET18
from custom_pd_embedding.train.util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class weightVit(nn.Module):
    def __init__(
        self,
        *,
        num_classifiers,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        mlp_path,
        resnet_path,
        vit_path,
    ):
        super().__init__()
        # vit pipeline config
        patch_dim = num_classifiers * num_classes

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.weight_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classifiers)

        # three classifier

        # manmade feature classifier
        self.mlp = MLP(input_size=MLP_INPUT_DIM, output_size=num_classes)
        self.mlp.load_state_dict(torch.load(mlp_path))

        # neural network feature classifier
        self.resnet18 = RESNET18(n_classes=num_classes)
        self.resnet18.load_state_dict(torch.load(resnet_path))

        # vit classifier
        self.vit_classifier = ViT(
            image_size=16,
            patch_size=16,
            num_classes=num_classes,
            dim=64,
            depth=4,
            heads=4,
            mlp_dim=64,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.vit_classifier.load_state_dict(torch.load(vit_path))

    def forward(self, x):
        # process data for different classifier
        # x shape is (batch_size,trace_steps,h=100,w=100)
        x_mlp = process_data_for_mlp_tensor(x)
        x_resnet = process_data_for_resnet_tensor(x)
        x_vit = process_data_tensor(x)

        # classifiers inference
        with torch.no_grad():
            mlp_pre = self.mlp(x_mlp)
            resnet_pre = self.resnet18(x_resnet)
            vit_pre = self.vit_classifier(x_vit)

        weightVit_input = torch.cat([mlp_pre, resnet_pre, vit_pre], dim=1)
        weightVit_input = weightVit_input.unsqueeze(1)

        # backbone inference
        weightVit_input = self.to_patch_embedding(weightVit_input)
        b, n, _ = weightVit_input.shape

        weight_tokens = repeat(self.weight_token, "1 1 d -> b 1 d", b=b)
        weightVit_input = torch.cat((weight_tokens, weightVit_input), dim=1)
        weightVit_input = self.dropout(weightVit_input)

        weightVit_input = self.transformer(weightVit_input)

        weightVit_input = (
            weightVit_input.mean(dim=1)
            if self.pool == "mean"
            else weightVit_input[:, 0]
        )

        # get weights and allocate to the classifiers
        weightVit_input = self.to_latent(weightVit_input)
        weights = torch.softmax(self.mlp_head(weightVit_input), dim=-1)
        w1, w2, w3 = torch.chunk(weights, 3, dim=1)
        final_pre = w1 * mlp_pre + w2 * resnet_pre + w3 * vit_pre

        return final_pre


class multiWeightVit(nn.Module):
    def __init__(
        self,
        *,
        num_classifiers,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        mlp_path,
        resnet_path,
        vit_path,
    ):
        super().__init__()
        self.num_classes = num_classes
        # vit pipeline config
        patch_dim = num_classifiers * num_classes

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.weight_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, patch_dim)

        # three classifier

        # manmade feature classifier
        self.mlp = MLP(input_size=MLP_INPUT_DIM, output_size=num_classes)
        self.mlp.load_state_dict(torch.load(mlp_path))

        # neural network feature classifier
        self.resnet18 = RESNET18(n_classes=num_classes)
        self.resnet18.load_state_dict(torch.load(resnet_path))

        # vit classifier
        self.vit_classifier = ViT(
            image_size=16,
            patch_size=16,
            num_classes=num_classes,
            dim=64,
            depth=4,
            heads=4,
            mlp_dim=64,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.vit_classifier.load_state_dict(torch.load(vit_path))

    def forward(self, x):
        # process data for different classifier
        # x shape is (batch_size,trace_steps,h=100,w=100)

        # cause data process code has existed,but is numpy type.for code reuse,firstly turn x into numpy,and then turn into tensor when process is finished
        x_np = x.cpu().numpy()
        x_mlp = process_data_for_mlp(x_np)
        x_resnet = process_data_for_resnet(x_np)
        x_vit = process_data(x_np)

        # turn into tensor
        x_mlp = torch.from_numpy(x_mlp).float().to(device)
        x_resnet = torch.from_numpy(x_resnet).float().to(device)
        x_vit = torch.from_numpy(x_vit).float().to(device)

        # classifiers inference
        with torch.no_grad():
            mlp_pre = self.mlp(x_mlp)
            resnet_pre = self.resnet18(x_resnet)
            vit_pre = self.vit_classifier(x_vit)

        weightVit_input = torch.cat([mlp_pre, resnet_pre, vit_pre], dim=1)
        weightVit_input = weightVit_input.unsqueeze(1)

        # backbone inference
        weightVit_input = self.to_patch_embedding(weightVit_input)
        b, n, _ = weightVit_input.shape

        weight_tokens = repeat(self.weight_token, "1 1 d -> b 1 d", b=b)
        weightVit_input = torch.cat((weight_tokens, weightVit_input), dim=1)
        weightVit_input = self.dropout(weightVit_input)

        weightVit_input = self.transformer(weightVit_input)

        weightVit_input = (
            weightVit_input.mean(dim=1)
            if self.pool == "mean"
            else weightVit_input[:, 0]
        )

        # get weights and allocate to the classifiers
        weightVit_input = self.to_latent(weightVit_input)
        weights = torch.softmax(self.mlp_head(weightVit_input), dim=-1)
        w1, w2, w3 = torch.chunk(weights, 3, dim=1)
        final_pre = w1 * mlp_pre + w2 * resnet_pre + w3 * vit_pre

        return final_pre
