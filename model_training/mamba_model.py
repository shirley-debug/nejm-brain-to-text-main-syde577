import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("WARNING: mamba-ssm not installed. Please install with: pip install mamba-ssm")
    print("Falling back to a placeholder implementation.")

class MambaDecoder(nn.Module):
    '''
    Defines the Mamba decoder (bidirectional variant)
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0,
                 d_state = 16,
                 d_conv = 4,
                 expand = 2,
                 bidirectional = True,
                 use_gradient_checkpointing = True,
                 ):
        super(MambaDecoder, self).__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days
        self.bidirectional = bidirectional
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Day-specific input layers
        self.day_layer_activation = nn.Softsign()

        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # Build Mamba layers
        self.mamba_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Input projection to match n_units
        if self.input_size != self.n_units:
            self.input_proj = nn.Linear(self.input_size, self.n_units)
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        else:
            self.input_proj = nn.Identity()
        
        # Create Mamba layers
        for i in range(self.n_layers):
            if self.bidirectional:
                forward_mamba = Mamba(
                    d_model=self.n_units,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                backward_mamba = Mamba(
                    d_model=self.n_units,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.mamba_layers.append(nn.ModuleDict({
                    'forward_mamba': forward_mamba,
                    'backward_mamba': backward_mamba,
                }))
            else:
                mamba_block = Mamba(
                    d_model=self.n_units,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.mamba_layers.append(mamba_block)
            
            if i < self.n_layers - 1 and self.rnn_dropout > 0:
                self.dropout_layers.append(nn.Dropout(self.rnn_dropout))
            else:
                self.dropout_layers.append(nn.Identity())

        # Layer norms for stability (both bidirectional and unidirectional)
        self.pre_layer_norms = nn.ModuleList()
        self.post_layer_norms = nn.ModuleList()
        for i in range(self.n_layers - 1):
            # Pre-normalization before Mamba processing
            self.pre_layer_norms.append(nn.LayerNorm(self.n_units))
            # Post-normalization after Mamba processing
            self.post_layer_norms.append(nn.LayerNorm(self.n_units))
        
        # Projection layers for bidirectional intermediate connections
        if self.bidirectional:
            self.inter_projections = nn.ModuleList()
            for i in range(self.n_layers - 1):
                proj = nn.Linear(2 * self.n_units, self.n_units)
                # Initialize with smaller weights to prevent gradient explosion
                nn.init.xavier_uniform_(proj.weight, gain=0.5)
                nn.init.zeros_(proj.bias)
                self.inter_projections.append(proj)

        # Output projection
        output_dim = 2 * self.n_units if self.bidirectional else self.n_units
        self.out = nn.Linear(output_dim, self.n_classes)
        # Use smaller initialization for output layer
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        nn.init.zeros_(self.out.bias)

    def _forward_mamba_layer(self, x, layer_idx):
        """Helper function for gradient checkpointing - processes one bidirectional layer"""
        mamba_dict = self.mamba_layers[layer_idx]
        
        # Save residual connection
        residual = x
        
        # Apply pre-normalization before Mamba processing (if not last layer)
        if layer_idx < self.n_layers - 1:
            x = self.pre_layer_norms[layer_idx](x)
        
        # Forward direction
        x_fwd = mamba_dict['forward_mamba'](x)
        
        # Backward direction
        x_bwd = torch.flip(x, dims=[1])
        x_bwd = mamba_dict['backward_mamba'](x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])
        
        # Concatenate both directions
        x_out = torch.cat([x_fwd, x_bwd], dim=-1)
        
        # Apply dropout
        x_out = self.dropout_layers[layer_idx](x_out)
        
        # Project back to n_units for next layer (if not last layer)
        if layer_idx < self.n_layers - 1:
            x_out = self.inter_projections[layer_idx](x_out)
            # Apply post-normalization after projection
            x_out = self.post_layer_norms[layer_idx](x_out)
            # Add residual connection for gradient stability
            x_out = x_out + residual
        
        return x_out

    def forward(self, x, day_idx, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example
        states   (optional) - not used for Mamba
        return_state (bool) - if True, return None for states
        '''
        # Apply day-specific layer
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        
        # Ensure day weights and biases match input dtype (for mixed precision compatibility)
        day_weights = day_weights.to(x.dtype)
        day_biases = day_biases.to(x.dtype)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the output of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)
        
        # Convert back to float32 for rest of model (handles BFloat16 inputs during eval)
        x = x.float()

        # (Optionally) Perform input concat operation
        if self.patch_size > 0: 
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dim, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Project input to n_units if needed
        x = self.input_proj(x)
        
        # Pass through bidirectional Mamba layers LAYER BY LAYER
        if self.bidirectional:
            for i in range(self.n_layers):
                if self.training and self.use_gradient_checkpointing:
                    # Use gradient checkpointing to save memory during training
                    x = checkpoint(self._forward_mamba_layer, x, i, use_reentrant=False)
                else:
                    # Regular forward pass (used during validation/testing)
                    x = self._forward_mamba_layer(x, i)
        else:
            # Unidirectional processing with residual connections
            for i, mamba_block in enumerate(self.mamba_layers):
                # Save residual
                residual = x
                
                # Pre-normalization (if not last layer)
                if i < self.n_layers - 1:
                    x = self.pre_layer_norms[i](x)
                
                # Mamba processing
                if self.training and self.use_gradient_checkpointing:
                    x = checkpoint(mamba_block, x, use_reentrant=False)
                else:
                    x = mamba_block(x)
                
                # Dropout
                x = self.dropout_layers[i](x)
                
                # Post-normalization and residual (if not last layer)
                if i < self.n_layers - 1:
                    x = self.post_layer_norms[i](x)
                    x = x + residual

        # Compute logits
        logits = self.out(x)
        
        if return_state:
            return logits, None
        
        return logits


class MambaDecoderUnidirectional(nn.Module):
    '''
    Unidirectional Mamba decoder (for comparison or when bidirectional is not needed)
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0,
                 d_state = 16,
                 d_conv = 4,
                 expand = 2,
                 use_gradient_checkpointing = True,
                 ):
        super(MambaDecoderUnidirectional, self).__init__()
        
        # Create a MambaDecoder with bidirectional=False
        self.decoder = MambaDecoder(
            neural_dim=neural_dim,
            n_units=n_units,
            n_days=n_days,
            n_classes=n_classes,
            rnn_dropout=rnn_dropout,
            input_dropout=input_dropout,
            n_layers=n_layers,
            patch_size=patch_size,
            patch_stride=patch_stride,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=False,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    
    def forward(self, x, day_idx, states = None, return_state = False):
        return self.decoder(x, day_idx, states, return_state)