import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

### Standard Classification test function
def test(net, testloader, device: str, accuracy_scale=1., freeze_bn_buffer=True, jit_augment=None):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0.0, 0.0, 0.0
    if freeze_bn_buffer:
        net.eval()
    else:
        net.train()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            if jit_augment is not None:
                images = jit_augment(images)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    loss = loss / total
    
    # converting to the required range
    accuracy = accuracy * accuracy_scale

    return loss, accuracy, total


def epochs_to_batches(num_epochs, dataset_len, batch_size, drop_last=False):
    if drop_last:
        fb_per_epoch = np.floor(dataset_len / int(batch_size))
    else:
        fb_per_epoch = np.ceil(dataset_len / int(batch_size))
    return int(fb_per_epoch * num_epochs)


def update_module(module, updates=None, memo=None):
    """
    Taken from https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py
    """
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p is not None and hasattr(p, 'update') and p.update is not None:
            if p in memo:
                module._parameters[param_key] = memo[p]
            else:
                updated = p + p.update
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff is not None and hasattr(buff, 'update') and buff.update is not None:
            if buff in memo:
                module._buffers[buffer_key] = memo[buff]
            else:
                updated = buff + buff.update
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module

class ReLUSTE(torch.autograd.Function):
    def __init__(self):
        super(ReLUSTE, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Identity(torch.autograd.Function):
    # just identity, this class is defined for compatibility/scalability reasons
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Step(torch.autograd.Function):
    def __init__(self):
        super(Step, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return (input > 0.).long().float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class SoftArgmax(torch.autograd.Function):
    def __init__(self):
        super(SoftArgmax, self).__init__()

    @staticmethod
    def forward(ctx, input):
        t = torch.argmax(F.softmax(input, dim=0), dim=0, keepdims=True)
        return torch.zeros_like(input).scatter_(0, t, 1.)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def split_weight_decay_parameters(model):
    wd_params = []
    wd_params_names = []
    for n, m in model.named_modules():
        if allow_weight_decay(m):
            wd_params.append(m.weight)
            wd_params_names.append(f'{n}.weight')

    no_wd_params = [p for n, p in model.named_parameters() if n not in wd_params_names]
    assert len(wd_params) + len(no_wd_params) == len(list(model.parameters())), "Sanity check failed."
    return wd_params_names, wd_params, no_wd_params

def allow_weight_decay(module):
    return isinstance(module,
                        (nn.Linear,
                        nn.Conv1d,
                        nn.Conv2d,
                        nn.Conv3d,
                        nn.ConvTranspose1d,
                        nn.ConvTranspose2d,
                        nn.ConvTranspose3d)
    )


class Hypergrad:
    """
    Credit: "Optimizing Millions of Hyperparameters by Implicit Differentiation"
    (https://arxiv.org/pdf/1911.02590.pdf)
    """

    def __init__(self, learning_rate=.1, truncate_iter=3):
        self.learning_rate = learning_rate
        self.truncate_iter = truncate_iter

    def grad(self, loss_val, loss_train, meta_params, params):
        dloss_val_dparams = torch.autograd.grad(
            loss_val,
            params,
            retain_graph=True,
            allow_unused=True
        )
        
        dloss_train_dparams = torch.autograd.grad(
                loss_train,
                params,
                allow_unused=True,
                create_graph=True,
        )

        v2 = self._approx_inverse_hvp(dloss_val_dparams, dloss_train_dparams, params)

        v3 = torch.autograd.grad(
            dloss_train_dparams,
            meta_params,
            grad_outputs=v2,
            allow_unused=True
        )

        return list(-g for g in v3)

    def _approx_inverse_hvp(self, dloss_val_dparams, dloss_train_dparams, params):
        p = v = dloss_val_dparams

        for _ in range(self.truncate_iter):
            grad = torch.autograd.grad(
                    dloss_train_dparams,
                    params,
                    grad_outputs=v,
                    retain_graph=True,
                    allow_unused=True
                )

            grad = [g * self.learning_rate for g in grad]  # scale: this a is key for convergence

            v = [curr_v - curr_g for (curr_v, curr_g) in zip(v, grad)]
            p = [curr_p + curr_v for (curr_p, curr_v) in zip(p, v)]

        return list(pp for pp in p)
