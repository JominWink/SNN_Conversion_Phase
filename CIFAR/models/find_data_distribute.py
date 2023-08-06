import torch
from torch.utils.data import DataLoader
from CIFAR.models import spiking_layer
import torch.nn as nn
import numpy as np
from spiking_layer import SpikeModel, SpikeModule
from typing import Union

class ActivationSaverHook:
    """
    This hook can save output of a layer.
    Note that we have to accumulate T times of the output
    if the model spike state is TRUE.
    """

    def __init__(self):
        self.stored_output = None
        self.stored_input = None

    def __call__(self, module, input_batch, output_batch):
        if self.stored_output is None:
            self.stored_output = output_batch

        if self.stored_input is None:
            self.stored_input = input_batch[0]

    def reset(self):
        self.stored_output = None
        self.stored_input = None


class GetLayerInputOutput:
    def __init__(self, model, target_module: nn.Module):
        self.model = model
        self.module = target_module
        self.data_saver = ActivationSaverHook()

    @torch.no_grad()
    def __call__(self, input):
        # do not use train mode here (avoid bn update)
        self.model.eval()
        h = self.module.register_forward_hook(self.data_saver)
        # note that we do not have to check model spike state here,
        # because the SpikeModel forward function can already do this
        _ = self.model(input)
        h.remove()
        return self.data_saver.stored_input, self.data_saver.stored_output

def quantile(tensor: torch.Tensor, p: float):
    try:
        return torch.quantile(tensor, p)
    except:
        tensor_np = tensor.cpu().detach().numpy()
        return torch.tensor(np.percentile(tensor_np, q=p*100)).type_as(tensor)


@torch.no_grad()
def find_activation_percentile (train_loader: torch.utils.data.DataLoader,
                           model: SpikeModel,
                           percentile: Union[float, None] = None,
                           percentile2:Union[float, None] = None,
                           iter:int = 30):
    print('test')
    # model.eval()
    num_cali_samples = 1024
    batch_size = 64
    data_sample = []
    for (input, target) in train_loader:
        data_sample += [input]
        if len(data_sample) * data_sample[-1].shape[0] >= num_cali_samples:
            break
    data_sample = torch.cat(data_sample, dim=0)[:num_cali_samples]
    for name, module in model.named_modules():
        if isinstance(module, SpikeModule):
            # print('\nAdjusting threshold for layer {}:'.format(name))
            device = next(module.parameters()).device
            data_sample = data_sample.to(device)
            model = model.to(device)
            get_out = GetLayerInputOutput(model, module)
            cached_batches = []
            # for i,(images,targets) in enumerate(data_sample):
            for i in range(int(data_sample.size(0) / batch_size)):
                # compute the original output
                _, cur_out = get_out(data_sample[i * batch_size:(i + 1) * batch_size])
                get_out.data_saver.reset()
                cached_batches.append(cur_out)

            cur_outs = torch.cat([x for x in cached_batches])
            sum_data = torch.sum(cur_outs)
            x = torch.nonzero(cur_outs)
            num = x.shape[0]
            per_data = sum_data / num * 0.9999
            print(per_data)

            # percentile = 0.9999
            # percentile2 = 0.9999
            # per_data = quantile(cur_outs,percentile)
            # max_data = cur_outs.max()
            # min_data = quantile(cur_outs,percentile2)
            # print(per_data,max_data,min_data)

