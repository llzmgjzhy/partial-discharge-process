# pipeline to process partial discharge data
# conclude a vit model to learn a set of weights assigned to the classifier and the classifiers
import torch
from torch import nn
from custom_pd_embedding.model.vit.vit_base.vit import (
    FeedForward,
    Attention,
    Transformer,
    ViT,
)
from custom_pd_embedding.model.mlp.mlp import MLP
from custom_pd_embedding.model.resnet.resnet18.resnet_in_vit16_custom import (
    RESNET18InVit16Custom,
)

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class weightVit(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()

        # vit pipeline config
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.weight_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        # classifier

        # manmade feature classifier
        self.mlp = MLP(
            in_features=num_classes,
            hidden_dim=512,
            out_features=6,
            dropout=0.1,
        )

        # neural network feature classifier
        self.resnet18 = RESNET18InVit16Custom(
            n_classes=6,
            hidden_dim=512,
        )

        # vit classifier
        self.vit_classifier = ViT(
            image_size=16,
            patch_size=16,
            num_classes=6,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
        )
