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
        is_multi=False,
        trace_steps=1,
    ):
        super().__init__()
        self.trace_steps = trace_steps
        self.num_classes = num_classes
        self.weights = []
        # vit pipeline config
        # classifier num + one label
        patch_dim = (num_classifiers + 1) * num_classes

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

        mlp_head_dim = num_classifiers
        if is_multi:
            mlp_head_dim = num_classifiers * num_classes
        self.mlp_head = nn.Linear(dim, mlp_head_dim)

        # three classifier

        # manmade feature classifier
        self.mlp = MLP(input_size=MLP_INPUT_DIM, output_size=num_classes).to(device)
        self.mlp.load_state_dict(torch.load(mlp_path))

        # neural network feature classifier
        self.resnet18 = RESNET18(n_classes=num_classes).to(device)
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
        ).to(device)
        self.vit_classifier.load_state_dict(torch.load(vit_path))

    def forward(self, x, label):
        # process data for different classifier
        # x shape is (batch_size,trace_steps,11171=403+768+10000)

        x_mlp = x[:, :, :403]
        x_resnet = x[:, :, 403:10403]
        x_resnet = x_resnet.reshape(x_resnet.shape[0], x_resnet.shape[1], 100, 100)
        x_vit = x[:, :, 10403:]

        mlp_pre = torch.empty(
            self.trace_steps, x.shape[0], self.num_classes, device=device
        )
        resnet_pre = torch.empty(
            self.trace_steps, x.shape[0], self.num_classes, device=device
        )
        vit_pre = torch.empty(
            self.trace_steps, x.shape[0], self.num_classes, device=device
        )
        # classifiers inference
        for i in range(self.trace_steps):
            with torch.no_grad():
                mlp_pre[i] = self.mlp(x_mlp[:, i, :])
                resnet_pre[i] = self.resnet18(x_resnet[:, i, :, :].unsqueeze(1))
                vit_pre[i] = self.vit_classifier(x_vit[:, i, :].unsqueeze(1))

        # concat three classifier output and transpose input into (batch_size,trace_steps,3*num_classes)
        weightVit_input = torch.cat([mlp_pre, resnet_pre, vit_pre], dim=2)
        weightVit_input = torch.transpose(weightVit_input, 0, 1)
        # one hot code for label
        # one hot coding
        label_one_hot_arr = torch.nn.functional.one_hot(
            label.long(), num_classes=self.num_classes
        ).to(device)
        # mask
        label_one_hot_arr[:, 0, :] = 0
        weightVit_input = torch.cat([weightVit_input, label_one_hot_arr], dim=2)

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
        # weights = torch.softmax(self.mlp_head(weightVit_input), dim=-1)
        weights = self.mlp_head(weightVit_input)
        self.weights = weights
        w1, w2, w3 = torch.chunk(weights, 3, dim=1)
        final_pre = w1 * mlp_pre[0] + w2 * resnet_pre[0] + w3 * vit_pre[0]

        return final_pre

    def inference(self, x):
        b, n, _ = x.shape
        x_mlp = x[:, :, :403]
        x_mlp = x_mlp.squeeze(1)
        x_resnet = x[:, :, 403:10403]
        x_resnet = x_resnet.reshape(x_resnet.shape[0], x_resnet.shape[1], 100, 100)
        x_vit = x[:, :, 10403:]

        with torch.no_grad():
            mlp_pre = self.mlp(x_mlp)
            resnet_pre = self.resnet18(x_resnet)
            vit_pre = self.vit_classifier(x_vit)

        weightVit_input = torch.cat([mlp_pre, resnet_pre, vit_pre], dim=1)
        weightVit_input = repeat(weightVit_input, "b c -> b t c", t=self.trace_steps)
        pseudo_label = torch.zeros(b, self.trace_steps, self.num_classes, device=device)
        weightVit_input = torch.cat([weightVit_input, pseudo_label], dim=2)
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
        # weights = torch.softmax(self.mlp_head(weightVit_input), dim=-1)
        weights = self.mlp_head(weightVit_input)
        w1, w2, w3 = torch.chunk(weights, 3, dim=1)
        final_pre = w1 * mlp_pre + w2 * resnet_pre + w3 * vit_pre

        return final_pre
