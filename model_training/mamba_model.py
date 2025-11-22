import torch 
from torch import nn
import torch.nn.functional as F

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
    
    This class combines day-specific input layers, bidirectional Mamba blocks, 
    and an output classification layer. Mamba is a state space model that 
    efficiently models long-range dependencies.
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
                 ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each Mamba layer (d_model)
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes 
        rnn_dropout    (float) - dropout rate applied between Mamba layers
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of Mamba layers 
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step 
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input 
        d_state     (int)      - state space dimension for Mamba (default: 16)
        d_conv      (int)      - convolution kernel size for Mamba (default: 4)
        expand      (int)      - expansion factor for Mamba inner dimension (default: 2)
        bidirectional (bool)  - whether to use bidirectional Mamba (default: True)
        '''
        super(MambaDecoder, self).__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days
        self.bidirectional = bidirectional

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # Build bidirectional Mamba layers
        self.mamba_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Input projection to match n_units
        if self.input_size != self.n_units:
            self.input_proj = nn.Linear(self.input_size, self.n_units)
            nn.init.xavier_uniform_(self.input_proj.weight)
        else:
            self.input_proj = nn.Identity()
        
        # Create Mamba layers
        for i in range(self.n_layers):
            if self.bidirectional:
                # Forward and backward Mamba blocks
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
                # Combine into a module that processes both directions
                self.mamba_layers.append(nn.ModuleDict({
                    'forward': forward_mamba,
                    'backward': backward_mamba,
                }))
            else:
                # Unidirectional Mamba
                mamba_block = Mamba(
                    d_model=self.n_units,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.mamba_layers.append(mamba_block)
            
            # Dropout between layers (except after last layer)
            if i < self.n_layers - 1 and self.rnn_dropout > 0:
                self.dropout_layers.append(nn.Dropout(self.rnn_dropout))
            else:
                self.dropout_layers.append(nn.Identity())

        # Output projection
        # If bidirectional, concatenate forward and backward outputs, so output dim is 2 * n_units
        output_dim = 2 * self.n_units if self.bidirectional else self.n_units
        self.out = nn.Linear(output_dim, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, day_idx, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        states   (optional) - not used for Mamba (Mamba doesn't maintain explicit hidden states like RNNs)
        return_state (bool) - if True, return None for states (Mamba doesn't have explicit hidden states)
        '''
        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the output of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0: 
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Project input to n_units if needed
        x = self.input_proj(x)
        
        # Pass through bidirectional Mamba layers
        if self.bidirectional:
            # Process forward direction
            x_forward = x
            for i, mamba_dict in enumerate(self.mamba_layers):
                x_forward = mamba_dict['forward'](x_forward)
                x_forward = self.dropout_layers[i](x_forward)
            
            # Process backward direction (reverse sequence)
            x_backward = torch.flip(x, dims=[1])  # Reverse along time dimension
            for i, mamba_dict in enumerate(self.mamba_layers):
                x_backward = mamba_dict['backward'](x_backward)
                x_backward = self.dropout_layers[i](x_backward)
            x_backward = torch.flip(x_backward, dims=[1])  # Reverse back to original order
            
            # Concatenate forward and backward outputs
            output = torch.cat([x_forward, x_backward], dim=-1)
        else:
            # Unidirectional processing
            output = x
            for i, mamba_block in enumerate(self.mamba_layers):
                output = mamba_block(output)
                output = self.dropout_layers[i](output)

        # Compute logits
        logits = self.out(output)
        
        if return_state:
            # Mamba doesn't maintain explicit hidden states like RNNs, so return None
            return logits, None
        
        return logits


class MambaDecoderUnidirectional(nn.Module):
    '''
    Unidirectional Mamba decoder (for comparison or when bidirectional is not needed)
    
    Same as MambaDecoder but with bidirectional=False by default.
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
                 ):
        '''
        Same parameters as MambaDecoder, but bidirectional is always False.
        '''
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
        )
    
    def forward(self, x, day_idx, states = None, return_state = False):
        return self.decoder(x, day_idx, states, return_state)

