import torch
from torch.autograd import Variable
import torch.nn.functional as F


class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total - log_det_W_total
        return loss/(z.size(0)*z.size(1)*z.size(2))


class InvertibleConv(torch.nn.Module):
    def __init__(self, channels):
        super(InvertibleConv, self).__init__()
        print("Channels: ", channels)
        self.conv = torch.nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        W = torch.qr(torch.FloatTensor(channels, channels).normal_())[0]
        
        if torch.det(W) < 0:
            W[:, 0] = -1*W[:,0]
        W = W.view(channels, channels, 1)
        self.conv.weight.data = W
        
    def forward(self, z, reverse=False):
        batch_size, group_size, n_of_groups = z.size()
        # Here, group size refers to the channel dimension of the data, and each matrix of channel dimension (i.e. [:, i, :]) is
        # going to be multiplied by the W matrix. The n_of_groups is number of groups, and that's *how many* matrices there are
        # in the channel dimension, and each one will be multiplied by W
        # The larger the group_size value is, the more thorough the "mixing" of the variables is before going back to the AC layer
        # In the extreme case of group_size=1, the variables are never permuted before going back to AC, and the flow would work
        # terribly because nothing would change order, so the same values would go into the WN each step of flow. In the other
        # extreme, where n_of_groups=1, mixing is maximized, but we run the risk of our W matrix being too big and possibly getting
        # numerical instability when we try to invert it, since we're not using very high precision to represent it (float32).
        
        W = self.conv.weight.squeeze()
        
        if reverse:
            if not hasattr(self, 'W_inv'):
                W_inv = W.float().inverse()
                W_inv = Variable(W_inv[..., None])
                self.W_inv = W_inv
            z = F.conv1d(z, self.W_inv, bias=None, stride=1, padding=0)
            return z
        else:
            log_det_w = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_w
        
class AffineCoupling(torch.nn.Module):
    def __init__(self, n_in_channels, n_context_channels, n_layers, dilation_list, n_channels, kernel_size):
        super(AffineCoupling, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_context_channels = n_context_channels
        self.n_layers = n_layers
        self.dilation_list = dilation_list
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.WN = WN(n_in_channels, n_context_channels, n_layers, n_channels, kernel_size, dilation_list)
    
    def forward(self, forecast, context, reverse=False):
        """
        context: batch x ? x ?
        forecast: batch x time
        """
        if reverse:
            n_half = int(forecast.size(1)/2)
            forecast_0 = forecast[:, :n_half, :]
            forecast_1 = forecast[:, n_half:, :]

            output = self.WN(forecast_0, context)

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            forecast_1 = (forecast_1 - b)/torch.exp(s)
            forecast = torch.cat([forecast_0, forecast_1], 1)

            return forecast
        else:
            n_half = int(forecast.size(1)/2)
            forecast_0 = forecast[:, :n_half, :]
            forecast_1 = forecast[:, n_half:, :]

            output  = self.WN(forecast_0, context)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            forecast_1 = torch.exp(log_s)*forecast_1 + b  # Might want to use sigmoid or clip the input to the exp for stability

            forecast = torch.cat([forecast_0, forecast_1], 1)

            return forecast, log_s        
        
class WaveGlow(torch.nn.Module):
    def __init__(self, n_context_channels, n_flows, n_group, n_early_every, n_early_size, n_layers, dilation_list, n_channels, kernel_size, cuda=True):
        super(WaveGlow, self).__init__()

        assert(n_layers == len(dilation_list))
        self.n_flows = n_flows                  # Number of steps of flow
        self.n_group = n_group                  # 
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.n_channels = n_channels
        self.IC = torch.nn.ModuleList()
        self.AC = torch.nn.ModuleList()
        self.cuda = cuda
        
        n_half = int(n_group/2)
        
        n_remaining_channels = n_group

        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.IC.append(InvertibleConv(n_remaining_channels))
            # In original code, things used to instantiate the WN here (since they don't use AC class):
            # WN(n_half, n_mel_channels*n_group, **WN_config)
            self.AC.append(AffineCoupling(n_half, n_context_channels, n_layers, dilation_list, n_channels, kernel_size))

        self.n_remaining_channels = n_remaining_channels # Apparently will be useful at inference, according to authors
        
            
    def forward(self, forecast, context):
        '''
        Transform a forecast with a given context into the latent space (so a spherical gaussian sample)
        
        forecast: torch FloatTensor of shape [b, N], where b is batch dimension and N is length of forecast (usually 96)
        context: torch FloatTensor of shape [b, M], where b is batch dimension and M is num features (currently probably just =N,
            use past 24 hours of data to predict next 24 hours of data)

        '''
         
        # context = context.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        # context = context.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        
        # Not sure why we do this, applying it blindly from original code
        # unfold(dimension, size, step): returns tensor which contains all slices of size "size" from the tensor
        # in the dimension "dimension". Step between two slices is given by step.
        # Ex: [1,2,3,4].unfold(0, 2, 1) = [1,2], [2,3], [3,4]
        # In effect, THI is the reshape operation which moves things from the spatial dimension into the channel dimension
        forecast = forecast.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        
        output_forecast = []
        log_s_list = []
        log_det_W_list = []
        
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_forecast.append(forecast[:, :self.n_early_size, :])
                forecast = forecast[:, self.n_early_size:, :]
                
            forecast, log_det_W = self.IC[k](forecast)
            log_det_W_list.append(log_det_W)
            
            forecast, log_s = self.AC[k](forecast, context)
            log_s_list.append(log_s)

            # print("Shape of forecast in forward: ", forecast.shape)
            
        output_forecast.append(forecast)
        # This keeps track of what shapes were assigned early, so that if we want to re-use these generated
        # latent samples in generate, we can properly assign the shapes
        early_assignment_shapes = [] 
        for early_output in output_forecast:
            early_assignment_shapes.append(early_output.shape)
        return torch.cat(output_forecast, 1), log_s_list, log_det_W_list, early_assignment_shapes
    
    def generate(self, context, sigma=1.0, latent_z=None, early_assignment_shapes=None):
        if latent_z is not None:
            assert early_assignment_shapes is not None, "If using latent_z, must also give early_assignment_shapes"
        if early_assignment_shapes is not None:
            assert latent_z is not None, "If giving early_assignment_shapes, specify latent_z as well"

        # if we don't give specifc points in the latent space to transform, sample random ones. Otherwise, use the specified points
        if latent_z is None:
            if self.cuda:
                forecast = torch.cuda.FloatTensor(context.size(0), self.n_remaining_channels, int(self.n_channels / self.n_group)).normal_()
            else:
                forecast = torch.FloatTensor(context.size(0), self.n_remaining_channels, int(self.n_channels / self.n_group)).normal_()
            # forecast = torch.autograd.Variable(sigma*forecast) # why does this have autograd in original paper? Never trains in this dir
            forecast = sigma*forecast # why does this have autograd in original paper? Never trains in this dir
        else:
            latent_z_parts = []
            for i in range(len(early_assignment_shapes)):
                if i==0:
                    latent_z_parts.append(latent_z[:, :early_assignment_shapes[i][1], :])
                else:
                    start = early_assignment_shapes[i-1][1]
                    finish = start + early_assignment_shapes[i][1]
                    latent_z_parts.append(latent_z[:, start:finish, :])


            latent_z_parts = latent_z_parts[::-1]
            forecast = latent_z_parts[0]

    
        # To keep track of how many early_every pieces we've appended, when we're using a given latent_z
        num_early_every_inserted = 1 
        for k in reversed(range(self.n_flows)):
            forecast = self.AC[k](forecast, context, reverse=True)
            forecast = self.IC[k](forecast, reverse=True)
            
            if k % self.n_early_every == 0 and k > 0:
                if latent_z is None:
                    if self.cuda:
                        z = torch.cuda.FloatTensor(context.size(0), self.n_early_size, int(self.n_channels / self.n_group)).normal_()
                    else:
                        z = torch.FloatTensor(context.size(0), self.n_early_size, int(self.n_channels / self.n_group)).normal_()
                    forecast = torch.cat((sigma*z, forecast), 1)
                else:
                    z = latent_z_parts[num_early_every_inserted]
                    forecast = torch.cat((z, forecast), 1)
                    num_early_every_inserted += 1
        
        # check dimensions and shit
        forecast = forecast.permute(0, 2, 1).contiguous().view(forecast.size(0), -1).data
        return forecast
    
# Copied directly from waveglow github
# Work this out on paper, it's related to the process used in the Table on pg 4 of glow paper
# I believe that separated it out into its own function so they could use jit on it for a speed up

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts




class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, n_context_channels, n_layers, n_channels,
                 kernel_size, dilation_list):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        assert(len(dilation_list) == n_layers)
        # number of layers in the neural network
        self.n_layers = n_layers
        # Number of channels in the data (not sure why doesn't match n_in_channels)
        self.n_channels = n_channels        
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.dilation_list = dilation_list

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        # Self.end is the final layer, which outputs something with 2*n_channels
        # We need this to have double the channels because it is computing both
        # the shift and scale factor, and each one of those is n_channels big

        # This is the conditioning layer, the layer which accepts our context
        # data as input, and outputs the data which gets mixed into the network
        # as it processes the samples. Specifically, the mixing happens in the
        # fused_add_tanh_sigmoid_multiply function, where output from this
        # layer is added to the sample being transformed by the network before
        # doing the tanh and sigmoid activations.
        
        # Note the output size of this layer, 2*n_channels*nlayers
        # This is enough output that, at each layer of this NN (and therefore at each
        # call to the fused_add_tanh_sigmoid_multiply function), there is a unique
        # bit of output from this layer that is mixed with the sample in that layer
        # of the neural network. If we need to regularize the model more, we could
        # consider not having it be unique for every layer
        cond_layer = torch.nn.Conv1d(n_context_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
#             dilation = 2 ** i # gonna want to change this since our data is much lower dimensional
            dilation = self.dilation_list[i]
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)


            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forecast, context):
        forecast = self.start(forecast)
        output = torch.zeros_like(forecast)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        context = self.cond_layer(context)

        for i in range(self.n_layers):
            context_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](forecast),
                context[:,context_offset:context_offset+2*self.n_channels,:],
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                forecast = forecast + res_skip_acts[:,:self.n_channels,:]
                output = output + res_skip_acts[:,self.n_channels:,:]
            else:
                output = output + res_skip_acts

        return self.end(output)
    
    