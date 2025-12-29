import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import *


class SimpleModel(nn.Module):

    def __init__(self, input_d: int, hidden_d: int, output_d: int):

        super(SimpleModel, self).__init__()

        self.input_d = input_d
        self.hidden_d = hidden_d
        self.output_d = output_d


        self.layer1 = nn.Linear(input_d, hidden_d)
        self.layer2 = nn.Linear(hidden_d, hidden_d)
        self.output_layer = nn.Linear(hidden_d, output_d)

        self.activation = nn.ReLU()


    def forward(self, x, return_embedding: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        emb = self.activation(x)
        x = self.output_layer(emb)
        if return_embedding:
            return x, emb
        return x

class DeeperModel(nn.Module):
    def __init__(self, input_d, hidden_d, output_d): 
        super().__init__()
        self.layer1 = nn.Linear(input_d, hidden_d)
        self.layer2 = nn.Linear(hidden_d, hidden_d)
        self.layer3 = nn.Linear(hidden_d, hidden_d)
        self.layer4 = nn.Linear(hidden_d, hidden_d)
        self.output_layer = nn.Linear(hidden_d, output_d)
        self.activation = nn.ReLU()

    def forward(
        self, x, return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.view(x.size(0), -1)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        emb = self.activation(self.layer4(x)) 
        x = self.output_layer(emb)
        if return_embedding:
            return x, emb
        return x
    
class PyramidalMLP(nn.Module):
    """
    Width decreases layer by layer (pyramidal).
    Example: input -> 600 -> 300 -> 150 -> 75 -> output
    """

    def __init__(
        self,
        input_d: int,
        hidden_dims: Tuple[int, int, int, int],
        output_d: int,
    ):
        super(PyramidalMLP, self).__init__()

        self.input_d = input_d
        self.hidden_dims = hidden_dims
        self.output_d = output_d

        h1, h2, h3, h4 = hidden_dims

        self.layer1 = nn.Linear(input_d, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, h3)
        self.layer4 = nn.Linear(h3, h4)

        self.output_layer = nn.Linear(h4, output_d)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor, return_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.view(x.size(0), -1)

        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        emb = self.activation(self.layer4(x))
        x = self.output_layer(x)

        if return_embedding:
            return x, emb
        return x


class BottleneckMLP(nn.Module):
    """
    Width decreases to a bottleneck and then expands
    Example: input -> 600 -> 300 -> 150 -> 300 -> 600 -> output
    """

    def __init__(
        self,
        input_d: int,
        hidden_dims: Tuple[int, int, int, int, int],
        output_d: int,
    ):
        super(BottleneckMLP, self).__init__()

        self.input_d = input_d
        self.hidden_dims = hidden_dims
        self.output_d = output_d

        h1, h2, hb, h4, h5 = hidden_dims  # hb = bottleneck dim

        self.layer1 = nn.Linear(input_d, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, hb)   # bottleneck layer
        self.layer4 = nn.Linear(hb, h4)
        self.layer5 = nn.Linear(h4, h5)

        self.output_layer = nn.Linear(h5, output_d)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        embedding_from: str = "bottleneck",  # "bottleneck" or "last"
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.view(x.size(0), -1)

        x1 = self.activation(self.layer1(x))
        x2 = self.activation(self.layer2(x1))
        xb = self.activation(self.layer3(x2))   # bottleneck representation
        x4 = self.activation(self.layer4(xb))
        x5 = self.activation(self.layer5(x4))

        out = self.output_layer(x5)

        if return_embedding:
            if embedding_from == "bottleneck":
                emb = xb
            elif embedding_from == "last":
                emb = x5
            else:
                raise ValueError(f"Unknown embedding_from='{embedding_from}'")
            return out, emb

        return out


class DiamondMLP(nn.Module):
    """
    Width increases to gain dimensionality and then -> bottleneck
    Example: input -> 150 -> 300 -> 150 -> 300 -> 600 -> output
    """

    def __init__(
        self,
        input_d: int,
        hidden_dims: Tuple[int, int, int, int, int],
        output_d: int,
    ):
        super(BottleneckMLP, self).__init__()

        self.input_d = input_d
        self.hidden_dims = hidden_dims
        self.output_d = output_d

        h1, h2, hb, h4, h5 = hidden_dims  # hb = bottleneck dim

        self.layer1 = nn.Linear(input_d, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, hb)   # bottleneck layer
        self.layer4 = nn.Linear(hb, h4)
        self.layer5 = nn.Linear(h4, h5)

        self.output_layer = nn.Linear(h5, output_d)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        embedding_from: str = "bottleneck",  # "bottleneck" or "last"
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.view(x.size(0), -1)

        x1 = self.activation(self.layer1(x))
        x2 = self.activation(self.layer2(x1))
        xb = self.activation(self.layer3(x2))   # bottleneck representation
        x4 = self.activation(self.layer4(xb))
        x5 = self.activation(self.layer5(x4))

        out = self.output_layer(x5)

        if return_embedding:
            if embedding_from == "bottleneck":
                emb = xb
            elif embedding_from == "last":
                emb = x5
            else:
                raise ValueError(f"Unknown embedding_from='{embedding_from}'")
            return out, emb

        return out


class PatchMLP(nn.Module):
    def __init__(self, input_c, input_h, input_w, patch_size, hidden_d, output_d):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches_h = input_h // patch_size
        self.num_patches_w = input_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        patch_input_d = input_c * patch_size * patch_size
        
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_input_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(hidden_d, output_d)

    def forward(self, x, return_embedding=False):
        # x shape: [Batch, 3, H, W]
        
        # patches shape: [Batch, 3, num_patches_h, num_patches_w, patch_size, patch_size]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # patches shape final: [Batch, Num_Patches, Patch_Vector_Size]
        patches = patches.contiguous().view(x.size(0), -1, 3 * self.patch_size * self.patch_size)
        
        batch_size, num_patches, patch_dim = patches.size()
        patches_flat = patches.view(batch_size * num_patches, patch_dim)
        
        # patch_embeddings shape: [Batch * Num_Patches, Hidden_D]
        patch_embeddings = self.patch_encoder(patches_flat)
        
        patch_embeddings = patch_embeddings.view(batch_size, num_patches, -1)
        
        image_embedding = patch_embeddings.mean(dim=1)
        
        out = self.output_layer(image_embedding)

        if return_embedding:
            return out, patch_embeddings
            
        return out


class PatchMLP2(nn.Module):
    def __init__(self, input_c, input_h, input_w, patch_size, hidden_d, output_d):
        super().__init__()
        
        self.patch_size = patch_size
        patch_input_d = input_c * patch_size * patch_size
        
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_input_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU()
        )
        
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_d, hidden_d // 2),
            nn.ReLU(),
            nn.Linear(hidden_d // 2, 1)
        )
        
        self.output_layer = nn.Linear(hidden_d, output_d)

    def forward(self, x, return_embedding=False):
        b, c, h, w = x.shape
        p = self.patch_size
        
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(b, -1, c * p * p)
        
        patch_embeddings = self.patch_encoder(patches)
        
        # The network looks at each patch and assigns it a "score"
        # scores shape: [Batch, Num_Patches, 1]
        scores = self.attention_net(patch_embeddings)
        
        # Convert scores to probabilities (Softmax)
        # We ensure that the sum of all importances is 1 (e.g.: 0.1, 0.8, 0.1)
        # This tells us what percentage of the final decision comes from each patch.
        weights = F.softmax(scores, dim=1)
        
        # Weighted Sum instead of Mean
        # We multiply each patch by its importance and sum them.
        # Irrelevant patches (weight close to 0) disappear.
        image_embedding = (patch_embeddings * weights).sum(dim=1)
        
        
        out = self.output_layer(image_embedding)
        
        if return_embedding:
            return out, image_embedding, weights
            
        return out