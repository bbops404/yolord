# projects/yolo-rd/som_backbone.py

from mmyolo.registry import MODELS
from mmyolo.models.backbones.yolov8_csp_darknet import YOLOv8CSPDarknet, YOLOv8CSPLayer
from mmdet.models.backbones.csp_darknet import CSPLayer
from .star_operation_module import StarOperationModule

@MODELS.register_module()
class SOM_YOLOv8CSPDarknet(YOLOv8CSPDarknet):
    """
    Custom YOLOv8 CSPDarknet backbone that integrates the StarOperationModule (SOM)
    into stages 3 and 4, as described in the YOLO-RD paper.
    """

    def __init__(self, *args, **kwargs):
        # We override the arch_settings to use our custom block types
        # Note: 'YOLOv8CSPLayer' with expand_ratio=3 is our placeholder for SOM
        self.arch_settings.update({
            'P5':
            # expand_ratio, block, num_blocks, args
            [[0.5, YOLOv8CSPLayer, 3, [2]],      # Stage 2
             [0.5, CSPLayer, 8, [False, 1.0]],  # Stage 3 with SOM
             [0.5, CSPLayer, 5, [False, 1.0]],  # Stage 4 with SOM
             [0.5, None, 0, []]]                # SPPF placeholder
        })
        print("--- INFO: Initializing Custom SOM-Backbone ---")
        super().__init__(*args, **kwargs)

    def _build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """
        Override the stage builder to inject the StarOperationModule.
        """
        expand_ratio, block, num_blocks, args = setting
        
        # Stages 0 and 1 (stem and stage 2 in diagram) are standard YOLOv8CSPLayer
        if stage_idx < 2:
            return super()._build_stage_layer(stage_idx, setting)

        # For Stages 3 and 4 (stage_idx 2 and 3), we build them with SOM
        block_name = f'stage{stage_idx + 1}'
        in_channels = self.channels[stage_idx]
        out_channels = int(self.channels[stage_idx + 1] * self.widen_factor)
        
        # This is where we create the custom block sequence
        # It consists of multiple SOM modules followed by one CSPLayer
        print(f"--- Building {block_name} with {num_blocks} StarOperationModules ---")
        
        layers = []
        # Create the repeated SOM blocks
        for i in range(num_blocks):
            layers.append(
                StarOperationModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    expand_ratio=3 # As per the SOM paper
                )
            )
        
        # The diagram shows a CSPLayer after the SOM blocks in Stage 3 and 4
        # We use the standard mmdet CSPLayer here
        layers.append(
            CSPLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                num_blocks=3, # A reasonable default, matching YOLOv8-s
                add_identity=True,
                use_depthwise=self.use_depthwise
            )
        )
        
        return [self.add_module(f'{block_name}', nn.Sequential(*layers))]