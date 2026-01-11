
import torch.nn as nn
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import models
import matplotlib.pyplot as plt

from typing import *
from torchview import draw_graph
from graphviz import Source

from PIL import Image
import torchvision.transforms.v2  as F
import numpy as np 

import pdb

from helpers import freeze_all, unfreeze_module, add_dropout_after_conv, add_dropout_after_block, add_batchnorm_after_block


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


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)

        x = self.output_layer(x)
        
        return x
    


class WraperModel(nn.Module):
    def __init__(
            self,
            num_classes: int,
            unfreeze_blocks: int = 0,
            dropout_blocks: int = 0,
            dropout_value: float = 0.5,
            use_batchnorm_blocks: bool = False

        ):
        """
        experiment_mode (str): Experiment configuration:
            - 'baseline': Frozen VGG16 + Simple linear layer.
            - 'multilayer': Frozen VGG16 + Hidden layers (more capacity).
            - 'dropout': Same as multilayer but with Dropout (less overfitting).
            - 'batchnorm': Adds Batch Normalization for stability.
            - 'finetune': Unfreezes last convolutional block + Dropout.
        """
        super(WraperModel, self).__init__()

        # Load pretrained VGG16 model
        self.backbone = models.mnasnet1_0(weights=None)
        self.backbone.load_state_dict(torch.load("/data/uabmcv2526/mcvstudent29/Week3/mnasnet1_0-IMAGENET1K_V1.pth"))
        
        # ----- experiment anar descongelant blocs de darrere cap endavant -----
        # Congelar-ho tot
        freeze_all(self.backbone)

        # Descongelar progressivament
        backbone_blocks = list(self.backbone.layers)
        total_blocks = len(backbone_blocks)
        if unfreeze_blocks > 0:
            for block in backbone_blocks[-unfreeze_blocks:]:
                unfreeze_module(block)
                
        # ----- experiment: BatchNorm en bloques descongelados -----
        if use_batchnorm_blocks and unfreeze_blocks > 0:
            backbone_blocks = list(self.backbone.layers)

            start_idx = len(backbone_blocks) - unfreeze_blocks

            for i in range(start_idx, len(backbone_blocks)):
                backbone_blocks[i] = add_batchnorm_after_block(
                    backbone_blocks[i]
                )

            # Write back modified blocks
            self.backbone.layers = nn.Sequential(*backbone_blocks)

        # Afegir classificador 
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )
        unfreeze_module(self.backbone.classifier)

        print(
            f"MODEL SETUP - Unfrozen backbone blocks: "
            f"{max(0, total_blocks - unfreeze_blocks)} → {total_blocks - 1}"
        )

        # ----- experiment afegir dropout a cada convolucional (de darrere cap endavant) -----
        # if dropout_blocks > 0:
        #     backbone_blocks = list(self.backbone.layers)
        #     for i in range(len(backbone_blocks) - dropout_blocks, len(backbone_blocks)):
        #         backbone_blocks[i] = add_dropout_after_conv(
        #             backbone_blocks[i],
        #             p=dropout_value
        #         )
        #     # write back
        #     self.backbone.layers = nn.Sequential(*backbone_blocks)

        # ----- experiment afegir un sol dropout a cada block (de darrere cap endavant) -----
        # if dropout_blocks > 0:
        #     backbone_blocks = list(self.backbone.layers)
        #     for i in range(len(backbone_blocks) - dropout_blocks, len(backbone_blocks)):
        #         backbone_blocks[i] = add_dropout_after_block(
        #             backbone_blocks[i],
        #             p=dropout_value
        #         )
        #     self.backbone.layers = nn.Sequential(*backbone_blocks)


    def forward(self, x):
        return self.backbone(x)
    


    def extract_feature_maps(self, input_image:torch.Tensor):

        conv_weights =[]
        conv_layers = []
        total_conv_layers = 0

        for module in self.backbone.layers.children():
            if isinstance(module, nn.Conv2d):
                total_conv_layers += 1
                conv_weights.append(module.weight)
                conv_layers.append(module)


        print("TOTAL CONV LAYERS: ", total_conv_layers)
        feature_maps = []  # List to store feature maps
        layer_names = []  # List to store layer names
        x= torch.clone(input=input_image)
        for layer in conv_layers:
            x = layer(x)
            feature_maps.append(x)
            layer_names.append(str(layer))

        return feature_maps, layer_names


    def extract_features_from_hooks(self, x, layers: List[str]):
        """
        Extract feature maps from specified layers.
        Args:
            x (torch.Tensor): Input tensor.
            layers (List[str]): List of layer names to extract features from.
        Returns:
            Dict[str, torch.Tensor]: Feature maps from the specified layers.
        """
        outputs = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                outputs[name] = output
            return hook

        # Register hooks for specified layers
        #for layer_name in layers:
        dict_named_children = {}
        for name, layer in self.backbone.named_children():
            for n, specific_layer in layer.named_children():
                dict_named_children[f"{name}.{n}"] = specific_layer

        for layer_name in layers:
            layer = dict_named_children[layer_name]
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))

        # Perform forward pass
        _ = self.forward(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs


    def modify_layers(self, modify_fn: Callable[[nn.Module], nn.Module]):
        """
        Modify layers of the model using a provided function.
        Args:
            modify_fn (Callable[[nn.Module], nn.Module]): Function to modify a layer.
        """
        self.vgg16 = modify_fn(self.vgg16)


    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if feature_extracting:
            for param in self.backbone.parameters():
                param.requires_grad = False


    def extract_grad_cam(self, input_image: torch.Tensor, 
                         target_layer: List[Type[nn.Module]], 
                         targets: List[Type[ClassifierOutputTarget]]) -> Type[GradCAMPlusPlus]:

        

        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:

            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam


def build_transforms(use_flip: bool = False,
                     use_color: bool = False,
                     use_geometric: bool = False,
                     use_translation: bool = False,
                     aug_ratio: float = 1.0) -> F.Compose:
    transforms = [
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.Resize((256, 256)),
        ]
    
    if use_flip:
        transforms += [
            F.RandomHorizontalFlip(p=0.5 * aug_ratio)
        ]
    if use_color:
        transforms += [
            F.RandomApply([
                F.ColorJitter(
                    brightness=0.2,   # + - 20%
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                )
            ], p=aug_ratio)
        ]
    if use_geometric:
        transforms += [
            F.RandomApply([
                F.RandomAffine(
                    degrees=10,          # rotation + - 10°
                    scale=(0.9, 1.1),    # zoom in/out
                    shear=5              # shear + - 5°
                )
            ], p=aug_ratio)
        ]
    if use_translation:
        transforms += [
            F.RandomApply([
                F.RandomAffine(
                    degrees=0,           # no rotation
                    translate=(0.1, 0.1) # + - 10% width/height
                )
            ], p=aug_ratio)
        ]

    return F.Compose(transforms)


# Example of usage
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load a pretrained model and modify it
    model = WraperModel(num_classes=8, feature_extraction=False)

    transformation = build_transforms(
        use_flip=False,
        use_color=False,
        use_geometric=False,
        use_translation=False
    )


    # Example GradCAM usage
    dummy_input = Image.open("/home/cboned/data/Master/MIT_split/test/highway/art803.jpg")#torch.randn(1, 3, 224, 224)
    input_image = transformation(dummy_input).unsqueeze(0)

    target_layers = [model.backbone.features[26]]
    targets = [ClassifierOutputTarget(6)]
    
    image = torch.from_numpy(np.array(dummy_input)).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min()) ## Image needs to be between 0 and 1 and be a numpy array (Remember that if you have norlized the image you need to denormalize it before applying this (image * std + mean))

    ## VIsualize the activation map from Grad Cam
    ## To visualize this, it is mandatory to have gradients.
    
    grad_cams = model.extract_grad_cam(input_image=input_image, target_layer=target_layers, targets=targets)

    visualization = show_cam_on_image(image, grad_cams, use_rgb=True)

    # Plot the result
    plt.imshow(visualization)
    plt.axis("off")
    plt.show()

    # Display processed feature maps shapes
    feature_maps, layer_names = model.extract_feature_maps(input_image)

                                                                 ### Aggregate the feature maps
    # Process and visualize feature maps
    processed_feature_maps = []  # List to store processed feature maps
    for feature_map in feature_maps:
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        min_feature_map, min_index = torch.min(feature_map, 0) # Get the min across channels
        processed_feature_maps.append(min_feature_map.data.cpu().numpy())
    
    
    # Plot All the convolution feature maps separately
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed_feature_maps)):
        ax = fig.add_subplot(5, 4, i + 1)
        ax.imshow(processed_feature_maps[i], cmap="hot", interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"{layer_names[i].split('(')[0]}_{i}", fontsize=10)


    plt.show()

    ## Plot a concret layer feature map when processing a image thorugh the model
    ## Is not necessary to have gradients

    with torch.no_grad():
        feature_map = (model.extract_features_from_hooks(x=input_image, layers=["features.28"]))["features.28"]
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        print(feature_map.shape)
        processed_feature_map, _ = torch.min(feature_map, 0) 

    # Plot the result
    plt.imshow(processed_feature_map, cmap="gray")
    plt.axis("off")
    plt.show()



    ## Draw the model
    model_graph = draw_graph(model, input_size=(1, 3, 224, 224), device='meta', expand_nested=True, roll=True)
    model_graph.visual_graph.render(filename="test", format="png", directory="./Week3")