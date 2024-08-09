from torch import nn


class ModifiedViT(nn.Module):
    def __init__(self, original_model):
        super(ModifiedViT, self).__init__()

        # Extract parts other than PatchEmbed layer
        self.features = nn.Sequential(
            *list(original_model.children())[1:]  # Skip the first PatchEmbed layer
        )

    def forward(self, x):
        x = self.features(x)
        return x
