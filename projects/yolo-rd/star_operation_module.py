import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmyolo.registry import BACKBONES

@BACKBONES.register_module()
class StarOperationModule(BaseModule):
    """Star Operation Module (SOM) as described in the paper.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        expand_ratio (int): Expansion ratio for the intermediate features. 
                            Default is 3, as shown in the diagram (3C).
        norm_cfg (dict): Configuration for the normalization layer.
        act_cfg (dict): Configuration for the activation layer (for ConvModule).
        init_cfg (dict, optional): Initialization config dict. Defaults to None.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=3,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU', inplace=True), # Standard YOLOv8 activation
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Calculate expanded channels for the star operation branches
        # Note: The input to the branches is F_pre_split, which has in_channels.
        self.expanded_channels = in_channels * expand_ratio

        # First block: DWConv 1 + BN
        # This takes F_input (in_channels) and outputs F_pre_split (in_channels).
        # act_cfg=None because the diagram shows BN directly before branching,
        # and the activations are handled in the branches or later.
        self.dwconv1_block = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels, # Output has the same number of channels
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels, # Depthwise convolution
            norm_cfg=norm_cfg,
            act_cfg=None, # No activation here
            bias=False)

        # Star Operation Branches
        # Both branches take F_pre_split (in_channels) and expand to expanded_channels.
        # Branch 1 (Top): Conv_1x1
        self.branch1_conv = ConvModule(
            in_channels=in_channels,
            out_channels=self.expanded_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=None, # No activation here
            bias=False)

        # Branch 2 (Bottom): Conv_1x1 followed by ReLU6
        self.branch2_conv = ConvModule(
            in_channels=in_channels,
            #out_channels=self.expanded_channels,
            out_channels=self.expanded_channels, 
            #in channels multiplied by expand ratio (3) okay n e2. pa confirm na lang e2 "introduce nonlinearity by clamping values to the range [0, 6]""
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=None, # ReLU6 will be applied externally
            bias=False)
        self.relu6 = nn.ReLU6(inplace=True) # Explicit ReLU6 as per diagram
        # ReLU activation function; when call it in the forward method "branch2_out=self.relu6(...)"
        # it applies the exact clamping and nonlinearity.
        # Mathematical definition of ReLU6 function -> min(max(0,x), 6).
        # The '6" is built into its name and its operation.


        # Post-Star Block
        # 1. Conv_1x1 (reduction)
        # Takes F_fused (expanded_channels) and reduces to in_channels for residual.
        self.reduce_conv = ConvModule(
            in_channels=self.expanded_channels,
            out_channels=in_channels, # Reduce back to in_channels for residual connection #Produces a C channel tensor as output
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=None, # No activation here
            bias=False)
        
        # 2. DWConv 2 + BN
        # Takes in_channels and outputs in_channels for residual addition.
        self.dwconv2_block = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels, # Depthwise convolution
            # norm_cfg=norm_cfg, #This line INCLUDES Batch Normalization
            norm_cfg=None, #This line REMOVES Batch Normalization
            act_cfg=None, # No activation here
            bias=False)
        #Comment ^^ regarding the BN part
        "Version 1 (With BN) correctly implements the diagram (Figure 3) from the paper, "
        "which explicitly shows a BN layer.  The BN ensures that the features being refined by the DW-Conv are on a consistent scale, "
        "which is critical for learning the spatial details of small cracks without the training becoming unstable."


        # Final Conv_1x1 (output transformation)
        # Takes F_intermediate (in_channels after residual) and outputs out_channels.
        self.final_conv = ConvModule(
            in_channels=in_channels, # After residual, channels are still in_channels
            out_channels=out_channels, # Output channels of the SOM module
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg, 
            bias=False)

    def forward(self, x):
        identity = x # Store input for residual connection (F_input)

        # Initial Block: DWConv 1 + BN
        # Corresponds to BN (DWConv 1 (F_input)) in the formula.
        f_pre_split = self.dwconv1_block(x) 

        # Star Operation Branches
        # Branch 1: Conv_1x1 (F_pre_split)
        branch1_out = self.branch1_conv(f_pre_split)

        # Branch 2: ReLU6 (Conv_1x1 (F_pre_split))
        branch2_intermediate = self.branch2_conv(f_pre_split)
        branch2_out = self.relu6(branch2_intermediate)
        
        # F_fused = Branch1_output * Branch2_output (element-wise multiplication)
        f_fused = branch1_out * branch2_out 

        # Post-Star Block
        # First Conv_1x1 (reduction) + BN
        f_post_star_part1 = self.reduce_conv(f_fused)
        
        # Then DWConv 2 + BN
        f_post_star_part2 = self.dwconv2_block(f_post_star_part1)

        # Residual Connection: F_intermediate = F_post_star + F_input
        # The sum `f_post_star_part2 + identity` corresponds to `F_intermediate = DWConv 2 (BN (Conv_1x1 (F_fused))) + F_input`
        # Note: f_post_star_part2 has `in_channels` to match `identity`.
        f_intermediate = f_post_star_part2 + identity 

        # Final Output: Conv_1x1 (F_intermediate)
        output = self.final_conv(f_intermediate)

        return output