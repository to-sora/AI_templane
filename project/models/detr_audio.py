import torch
import torch.nn as nn
import math
from torchvision.models import resnet50 ,ResNet50_Weights
import math
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights

class DETRAudio(nn.Module):
    def __init__(self, config):
        super(DETRAudio, self).__init__()
        # Get the base dimension from config
        base_dimension = config["model_structure"].get("dimension", 128)
        num_encoder_layers = config["model_structure"].get("num_encoder_layers", 1)
        num_decoder_layers = config["model_structure"].get("num_decoder_layers", 1)
        self.num_queries = config['max_objects']
        self.debug = config.get('debug', False)
        self.log_scale_power = config["model_structure"].get("log_scale_power", False)
        if self.debug:
            print("----->","Debug mode enabled")

        # Check for 'add_time_dimension' in config
        self.add_time_dimension = config["model_structure"].get('add_time_dimension', False)
        if self.add_time_dimension:
            print("----->","Adding extra time dimension to embedding")
            dimension = base_dimension + 1  # Increase dimension by 1
        else:
            dimension = base_dimension
        self.dimension = dimension  # Update the model's dimension attribute

        # Backbone selection (unchanged)
        backbone_type = config["model_structure"].get('backbone_type', 'Resnet')
        pretrain = config.get('pretrain', False)
        if backbone_type == 'None':
            print("----->","Using Simple CNN backbone")
            self.backbone = SimpleCNN()
            backbone_output_dim = self.backbone.output_dim
        elif backbone_type == 'resnet18':
            print("-----> Using ResNet-18 backbone with pretrain:", pretrain)
            if pretrain:
                self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet18(weights=False)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            backbone_output_dim = 512
        elif backbone_type == 'resnet50':
            print("----->","Using ResNet-50 backbone with pretrain:", pretrain)
            if pretrain:
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_NO_TOP)
            else:
                self.backbone = resnet50(weights=False)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            backbone_output_dim = 2048
        elif backbone_type == 'TokenizedBackbone':
            print("----->","Using TokenizedBackbone")
            self.backbone = TokenizedBackbone(embed_dim=base_dimension)
            backbone_output_dim = self.backbone.output_dim
        elif backbone_type == 'CustomTokenizedBackbone':
            print("----->","Using CustomTokenizedBackbone")
            self.backbone = CustomTokenizedBackbone(embed_dim=base_dimension,CONFIG=config)
            backbone_output_dim = self.backbone.output_dim

        # Input projection to match transformer dimension (updated dimension)
        self.input_proj = nn.Conv2d(backbone_output_dim, base_dimension, kernel_size=1)

        # Positional embedding (unchanged)
        positional_embedding = config["model_structure"].get('positional_embedding', 'None')
        if positional_embedding == 'sinusoid':
            print("----->","Using sinusoidal positional encoding")
            self.positional_encoding = PositionalEncoding(base_dimension)
        elif positional_embedding == '2d':
            print("----->","Using 2D positional encoding")
            self.positional_encoding = TwoDPositionalEncoding(base_dimension)
        else:
            print("----->","No positional encoding")
            self.positional_encoding = None  # No positional encoding
        print("----->","Adding time dimension:", self.add_time_dimension)

        # Time series model selection (unchanged)
        time_series_type = config["model_structure"].get('time_series_type', 'default')
        if time_series_type == 'default':
            print("----->","Using default transformer")
            self.transformer = nn.Transformer(d_model=dimension, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        elif time_series_type == 'SparseFormer':
            print("----->","Using SparseFormer")
            self.transformer = SparseFormer(d_model=dimension, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        elif time_series_type == 'RNN':
            print("----->","Using RNN")
            self.transformer = StackedRNN(d_model=dimension, num_layers=num_encoder_layers)
        elif time_series_type == 'LSTM':
            print("----->","Using LSTM")
            self.transformer = StackedLSTM(d_model=dimension, num_layers=num_encoder_layers)
        else:
            raise ValueError(f"Unknown time_series_type: {time_series_type}")

        # Query embeddings for transformer decoder (updated dimension)
        self.query_embed = nn.Embedding(self.num_queries, dimension)

        # Classification and regression heads (updated dimension)
        number_of_layers = config["model_structure"].get('number_of_layers', 2)
        activation_last_layer = config["model_structure"].get('classification_activation_last_layer', 'sigmoid')
        num_classes_note_type = config['num_classes']['note_type'] + 1
        num_classes_instrument = config['num_classes']['instrument'] + 1
        num_classes_pitch = config['num_classes']['pitch'] + 1

        self.MLP_skip = config["model_structure"].get('MLP_skip', 0)
        self.debug = config.get('debug', False)
        self.class_embed_note_type = MLP(dimension, dimension, num_classes_note_type, number_of_layers, activation_last_layer,use_skip=self.MLP_skip,debug=self.debug)
        self.class_embed_instrument = MLP(dimension, dimension, num_classes_instrument, number_of_layers, activation_last_layer,use_skip=self.MLP_skip,debug=self.debug)
        self.class_embed_pitch = MLP(dimension, dimension, num_classes_pitch, number_of_layers, activation_last_layer,use_skip=self.MLP_skip,debug=self.debug)

        # Regression head with specified activation function (unchanged)
        regression_activation_last_layer = config["model_structure"].get('regression_activation_last_layer', 'relu')
        print("----->","Regression activation:", regression_activation_last_layer)

        self.start_time_head = MLP(
            dimension, dimension, 1, number_of_layers, regression_activation_last_layer, config.get("start_time_scaler", 20),use_skip=self.MLP_skip,debug=self.debug
        )
        self.duration_head = MLP(
            dimension, dimension, 1, number_of_layers, regression_activation_last_layer, config.get("duration_scaler", 2),use_skip=self.MLP_skip,debug=self.debug
        )
        self.velocity_head = MLP(
            dimension, dimension, 1, number_of_layers, regression_activation_last_layer, config.get("velocity_scaler", 200),use_skip=self.MLP_skip,debug=self.debug
        )
        print("----->","START TIME SCALER:", config.get("start_time_scaler", 20))
        print("----->","DURATION SCALER:", config.get("duration_scaler", 2))
        print("----->","VELOCITY SCALER:", config.get("velocity_scaler", 200))
        print("----->","DETRAudio model initialized")

    def forward(self, x):
        # x: [batch_size, 1, freq_bins, time_steps]
        bs = x.size(0)
        if self.log_scale_power:
            x = torch.log(torch.abs(x) + 1e-12)
            if self.debug:
                print("----->","Log scaling power spectrogram")
        if self.debug:
            print("----->","Input shape:", x.shape)
        x = self.backbone_conv(x)  # Backbone processing
        if self.debug:
            print("----->","Backbone output shape:", x.shape)
        x = self.input_proj(x)     # [batch_size, dimension, H, W]
        if self.debug:
            print("----->","Input projection shape:", x.shape)

        # Positional encoding (unchanged)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        if self.debug:
            print("----->","Positional encoding shape:", x.shape)

        # Flatten spatial dimensions and permute
        x = x.flatten(2).permute(2, 0, 1)  # [seq_len, batch_size, dimension]
        if self.debug:
            print("----->","Flattened shape:", x.shape)

        # Add time dimension if enabled
        if self.add_time_dimension:
            seq_len = x.size(0)
            # Create a linear interpolation from 0 to 1 across the sequence length
            time_emb = torch.linspace(0, 1, steps=seq_len, device=x.device).unsqueeze(1).unsqueeze(2)  # [seq_len, 1, 1]
            time_emb = time_emb.expand(-1, x.size(1), 1)  # [seq_len, batch_size, 1]
            x = torch.cat([x, time_emb], dim=2)  # Concatenate on embedding dimension
            if self.debug:
                print("----->","After adding time dimension, shape:", x.shape)

        # Time series model processing (unchanged)
        if isinstance(self.transformer, nn.Transformer):
            memory = self.transformer.encoder(x)
            hs = self.transformer.decoder(self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1), memory)
            if self.debug:
                print("----->","Transformer output shape:", hs.shape)
        else:
            hs = self.transformer(x)  # (seq_len, batch_size, dimension)
            # Uniformly sample `num_queries` indices from the sequence
            seq_len = hs.size(0)
            num_queries = min(self.num_queries, seq_len)
            indices = torch.linspace(0, seq_len - 1, steps=num_queries).long().to(x.device)
            hs = hs[indices]  # (num_queries, batch_size, dimension)
            if self.debug:
                print("-----> Time series model output shape:", hs.shape)
        hs = hs.permute(1, 0, 2)  # [batch_size, num_queries, dimension]

        # Classification and regression heads (unchanged)
        outputs_note_type = self.class_embed_note_type(hs)
        outputs_instrument = self.class_embed_instrument(hs)
        outputs_pitch = self.class_embed_pitch(hs)
        # Regression outputs
        outputs_start_time = self.start_time_head(hs)
        outputs_duration = self.duration_head(hs)
        outputs_velocity = self.velocity_head(hs)

        outputs_regression = torch.cat([outputs_start_time, outputs_duration, outputs_velocity], dim=-1)

        self.debug = False
        return {
            'pred_note_type': outputs_note_type,
            'pred_instrument': outputs_instrument,
            'pred_pitch': outputs_pitch,
            'pred_regression': outputs_regression
        }

    def backbone_conv(self, x):
        if hasattr(self.backbone, 'conv1'):
            # ResNet backbone
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
        else:
            # Custom backbone
            x = self.backbone(x)
        return x

    def freeze_layers(self, freeze_config):
        # Freezing layers (unchanged)
        if freeze_config.get("freeze_backbone", False):
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("-----> Backbone layers frozen.")

        if freeze_config.get("freeze_transformer", False):
            for param in self.transformer.parameters():
                param.requires_grad = False
            print("-----> Transformer layers frozen.")

        if freeze_config.get("freeze_heads", False):
            for param in self.class_embed_note_type.parameters():
                param.requires_grad = False
            for param in self.class_embed_instrument.parameters():
                param.requires_grad = False
            for param in self.class_embed_pitch.parameters():
                param.requires_grad = False
            for param in self.start_time_head.parameters():
                param.requires_grad = False
            for param in self.duration_head.parameters():
                param.requires_grad = False
            for param in self.velocity_head.parameters():
                param.requires_grad = False
            print("-----> MLP heads frozen.")

class TokenizedBackbone(nn.Module):
    def __init__(
        self,
        input_channels=1,
        embed_dim=256,
        patch_size=(16, 16),
        stride=(8, 8),
        padding=(0, 0),
        num_conv_layers=2,
        dropout=0.1
    ):
        """
        Tokenizes spectrogram input into overlapping patches suitable for transformers.

        Args:
            input_channels (int): Number of input channels (1 for grayscale spectrograms).
            embed_dim (int): Dimension of the token embeddings.
            patch_size (tuple): Size of each patch (height, width).
            stride (tuple): Stride for convolution (controls overlap).
            padding (tuple): Padding for convolution.
            num_conv_layers (int): Number of convolutional layers to process patches.
            dropout (float): Dropout rate for regularization.
        """
        super(TokenizedBackbone, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = embed_dim

        # Initial convolution to create patch embeddings
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )  # Output: (B, embed_dim, H_patch, W_patch)

        # Optional additional convolutional layers for deeper feature extraction
        conv_layers = []
        for _ in range(num_conv_layers):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            conv_layers.append(nn.BatchNorm2d(embed_dim))
            conv_layers.append(nn.ReLU(inplace=True))
        self.additional_convs = nn.Sequential(*conv_layers) if num_conv_layers > 0 else nn.Identity()

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms of shape (B, C, H, W).

        Returns:
            torch.Tensor: Token embeddings of shape (B, N_patches, embed_dim).
        """
        B, C, H, W = x.shape

        # Apply initial convolution to extract patches
        x = self.conv(x)  # Shape: (B, embed_dim, H_patch, W_patch)
        x = self.additional_convs(x)  # Shape: (B, embed_dim, H_patch, W_patch)

        # # Flatten spatial dimensions to create tokens
        # x = x.flatten(2)  # Shape: (B, embed_dim, N_patches)
        # x = x.transpose(1, 2)  # Shape: (B, N_patches, embed_dim)

        # Apply layer normalization and dropout
        # x = self.layer_norm(x)
        # x = self.dropout(x)

        return x



class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron with configurable last layer activation
    and multiple skip (residual) connections.

    Parameters:
    - input_dim (int): Dimension of the input features.
    - hidden_dim (int): Dimension of the hidden layers.
    - output_dim (int): Dimension of the output layer.
    - num_layers (int): Total number of Linear layers in the MLP.
    - activation_last_layer (str or None): Activation function after the last layer.
    - scaler (float): Scalar to multiply the final output.
    - use_skip (int): Number of skip (residual) connections to include.
                      Defaults to 0 (no skip connections).
    """
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        output_dim, 
        num_layers, 
        activation_last_layer=None, 
        scaler=1, 
        use_skip=0,
        debug=False
    ):
        super().__init__()
        self.scaler = scaler
        self.use_skip = use_skip
        self.debug = debug

        if num_layers < 2:
            raise ValueError("num_layers should be at least 2 (input and output layers).")

        # Initialize the layers list
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        print("MLP ----> using activation_last_layer:", activation_last_layer)

        # Add the specified activation function to the last layer, if any
        if activation_last_layer and activation_last_layer.lower() != 'none':
            activation = activation_last_layer.lower()
            if activation == 'relu':
                layers.append(nn.ReLU())

            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
                layers.append(AddOne())  # Ensure AddOne is defined
            elif activation == 'identity' or activation == 'none':
                pass
            else:
                raise ValueError(f"Unknown activation function: {activation_last_layer}")

        self.layers = nn.ModuleList(layers)

        # Define residual connections if use_skip > 0
        if use_skip > 0:
            if use_skip > num_layers - 1:
                raise ValueError(f"use_skip ({use_skip}) cannot exceed num_layers - 1 ({num_layers -1}).")

            # Determine after which layers to add residuals
            # Evenly distribute skip connections
            skip_points = self._calculate_skip_points(num_layers, use_skip)

            self.skip_points = skip_points  # List of layer indices after which to add residuals
            print("MLP ----> Skip points:", skip_points)

            # For each skip point, determine the input index to add
            # We'll save the input before the segment starts
            # and add it after the segment ends

            # To handle transformations, we need to define residual transformations
            # Only if the input dimension to the residual doesn't match output_dim
            self.residual_transforms = nn.ModuleList()
            for _ in range(use_skip):
                # Assuming the input to be added is of dimension hidden_dim
                # since after the first layer, the dimension is hidden_dim
                # except possibly for the first residual if input_dim != hidden_dim

                # To generalize, we'll handle residuals based on saved input dimensions
                # For simplicity, assume residuals are from hidden_dim to hidden_dim
                # unless it's the first residual from input_dim to hidden_dim

                # You might need to adjust this based on your architecture
                self.residual_transforms.append(nn.Identity())  # Placeholder
            # Adjust residual transforms based on actual saved inputs
            # We'll handle this in the forward pass
        else:
            self.skip_points = []
            self.residual_transforms = None

    def _calculate_skip_points(self, num_layers, use_skip):
        """
        Calculate the layer indices after which to add residual connections.
        Distribute skips as evenly as possible across the layers.

        Returns a list of layer indices.
        """
        skip_points = []
        total_segments = use_skip + 1
        layers_per_segment = num_layers // total_segments
        remainder = num_layers % total_segments

        current = 0
        for i in range(use_skip):
            # Distribute the remainder across the first few segments
            increment = layers_per_segment + (1 if i < remainder else 0)
            current += increment
            skip_points.append(current * 2 - 1)  # Each Linear and ReLU is a pair
        return skip_points

    def forward(self, x):
        debug = self.debug
        if self.use_skip > 0:
            residuals = []
            residual_transforms = []

            # Prepare residual connections
            current_layer = 0
            residual_idx = 0
            x_saved = x  # Initial input

            for i, layer in enumerate(self.layers):
                x = layer(x)

                # Check if the current layer is a skip point
                if i in self.skip_points:
                    # Save the current x as residual
                    residual = x_saved
                    if debug:
                        print(f"saving index at layer {i}")

                    # If input and output dimensions match, use identity
                    # Else, apply a transformation
                    if residual.shape[-1] != x.shape[-1]:
                        # Apply a linear transformation to match dimensions
                        # Assume residual was saved from a layer with matching dimension
                        # Here, we use the corresponding residual_transform
                        residual_transform = self.residual_transforms[residual_idx]
                        res = residual_transform(residual)
                    else:
                        res = residual

                    # Add residual to current x
                    if debug:
                        print(f"Adding residual at layer {i}")
                    x = x + res

                    residual_idx += 1

                    # Save the current x for the next residual
                    x_saved = x
            self.debug = False
            return x * self.scaler
        else:
            # No residual connections; standard forward pass
            for layer in self.layers:
                x = layer(x)
            return x * self.scaler

    def _initialize_residual_transforms(self):
        """
        Initialize residual transformations based on the skip points.
        This function should be called after the model is initialized.
        """
        for idx in range(len(self.skip_points)):
            # Determine the dimension of the residual to be added
            # For simplicity, assume residuals are from input_dim to hidden_dim or hidden_dim to output_dim
            # You might need to adjust this based on your architecture

            # Example: If residual comes from input layer
            # Apply a transformation to match hidden_dim
            # Else, use Identity
            # Here, we assume all residuals are from hidden_dim to hidden_dim
            pass  # No action needed since residual_transforms are set to Identity





