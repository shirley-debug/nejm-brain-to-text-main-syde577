import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("WARNING: mamba-ssm not installed")

class MambaDecoder(nn.Module):
    '''
    Memory-efficient Mamba decoder with gradient checkpointing
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
                 use_gradient_checkpointing = True,  # NEW PARAMETER
                 ):
        super(MambaDecoder, self).__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required")
        
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

        # Build bidirectional Mamba layers
        self.mamba_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Input projection
        if self.input_size != self.n_units:
            self.input_proj = nn.Linear(self.input_size, self.n_units)
            nn.init.xavier_uniform_(self.input_proj.weight)
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

        # Projection layers for bidirectional intermediate connections
        if self.bidirectional:
            self.inter_projections = nn.ModuleList()
            for i in range(self.n_layers - 1):
                proj = nn.Linear(2 * self.n_units, self.n_units)
                nn.init.xavier_uniform_(proj.weight)
                self.inter_projections.append(proj)

        # Output projection
        output_dim = 2 * self.n_units if self.bidirectional else self.n_units
        self.out = nn.Linear(output_dim, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def _forward_mamba_layer(self, x, layer_idx):
        """Helper function for gradient checkpointing"""
        mamba_dict = self.mamba_layers[layer_idx]
        
        # Forward direction
        x_fwd = mamba_dict['forward_mamba'](x)
        
        # Backward direction
        x_bwd = torch.flip(x, dims=[1])
        x_bwd = mamba_dict['backward_mamba'](x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])
        
        # Concatenate
        x_out = torch.cat([x_fwd, x_bwd], dim=-1)
        
        # Apply dropout
        x_out = self.dropout_layers[layer_idx](x_out)
        
        # Project back to n_units for next layer (if not last layer)
        if layer_idx < self.n_layers - 1:
            x_out = self.inter_projections[layer_idx](x_out)
        
        return x_out

    def forward(self, x, day_idx, states = None, return_state = False):
        # Apply day-specific layer
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # Perform input concat operation
        if self.patch_size > 0: 
            x = x.unsqueeze(1)
            x = x.permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2)
            x_unfold = x_unfold.permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Project input
        x = self.input_proj(x)
        
        # Pass through Mamba layers with gradient checkpointing
        if self.bidirectional:
            for i in range(self.n_layers):
                if self.training and self.use_gradient_checkpointing:
                    # Use gradient checkpointing to save memory
                    x = checkpoint(self._forward_mamba_layer, x, i, use_reentrant=False)
                else:
                    x = self._forward_mamba_layer(x, i)
        else:
            # Unidirectional processing
            output = x
            for i, mamba_block in enumerate(self.mamba_layers):
                if self.training and self.use_gradient_checkpointing:
                    # Checkpoint the mamba block
                    output = checkpoint(mamba_block, output, use_reentrant=False)
                else:
                    output = mamba_block(output)
                # Apply dropout outside checkpoint (dropout doesn't need gradients)
                output = self.dropout_layers[i](output)
            x = output

        # Compute logits
        logits = self.out(x)
        
        if return_state:
            return logits, None
        
        return logits