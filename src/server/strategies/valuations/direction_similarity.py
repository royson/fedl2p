
import os
import torch
import torch.nn.functional as F
import numpy as np

from src.server.strategies.utils import get_layer
from src.server.strategies.valuations import ClientValuation
from src.data import cycle
from src.models.model_utils import set_weights, get_updates

from typing import List
from flwr.common import Weights
from src.utils import get_func_from_config

import logging
logger = logging.getLogger(__name__)

class DirectionSimilarity(ClientValuation):
    def __init__(self, *args, 
             partition='val', 
             combined=True, 
             calc_weighting_layers=None, 
             val_batch_size=32, 
             per_class=False, 
             **kwargs):
        '''
            `calc_weighting_layers`: determine the layers of your model to be used for the metric
            `combined`: determines whether dot product happens per layer or on all layers
            `val_batch_size`: the batch size used for separate dataset evaluation
            `per_class`: compares the metric per class. **Only last FC layer is supported if set to true**  
        '''
        super().__init__(*args, **kwargs)

        # overwrite default valloader in ClientValuation
        if self.validation_set:
            self.valloader = iter(cycle(self.dataloader(
                data_pool='train',
                partition=partition, 
                batch_size=val_batch_size, 
                num_workers=0,
                shuffle=True
            )))
        self.combined = combined

        self.calc_weighting_layers = self.init_layer_names(calc_weighting_layers)
        self.per_class = per_class
        self.key_metrics = ['w1_mag', 'w2_mag', '|w1_mag-w2_mag|', 'w1_mag+w2_mag', 'l2-N', 'l2-N*w2_mag', 'cos_sim', 'l2_distance']

        # logging
        logger.info(f'Layers used to compute weightings: {self.calc_weighting_layers}.')
        logger.info(f'Computing metric per class: {self.per_class}')

    def init_layer_names(self, layer_names):
        if layer_names is not None:
            assert len(layer_names) > 0
            for layer_name in layer_names:
                try:
                    layer = get_layer(self.model, layer_name)
                except AttributeError:
                    logger.error(f'{layer_name} not found.')
                    import pdb
                    pdb.set_trace()
            return layer_names
        else:
            return [name for name, param in self.model.named_parameters() if param.requires_grad]

    def compute_metrics(self, w1_update, w2_update):
        l2 = (w1_update - w2_update).norm(2).item()

        ### Splitting angular and magnitude factors
        # Computing magnitudes
        w1_update_mag = w1_update.norm(2).item()
        w2_update_mag = w2_update.norm(2).item()
        w1_update_uv = w1_update / w1_update_mag
        w2_update_uv = w2_update / w2_update_mag

        cos_sim = torch.dot(w1_update_uv, w2_update_uv).item()

        ed = (w1_update_uv - w2_update_uv).norm(2).item()

        return w1_update_mag, w2_update_mag, abs(w1_update_mag - w2_update_mag), w1_update_mag + w2_update_mag, ed, ed*w2_update_mag, cos_sim, l2

    def evaluate(self, current_weights: Weights, lr: float, weights_1: List[Weights], weights_2: Weights=None, use_val=False, **kwargs):
        '''
            Comparing all weights in `weights_1` with `weights_2`

            Inputs:
            `current_weights`: is the last round weights used to compute updates
            `lr`: round's lr used to compute the updates during training
            `weights_1` and `weights_2`: comparing the weights in `weights_1` with `weights_2` or validation set depending on `use_val`
            `use_val`: uses update from validation set instead of `weights_2`

            Returns: 
            `all_metrics` Dict[List[List or Float]]: a dictionary of lists with see compute_metrics function.
                           each list element corresponds to each weights in `weights_1` 

        '''
        assert weights_2 is not None or use_val

        ### take only weights from net
        current_weights = current_weights[:self.net_length]
        weights_1 = [_w[:self.net_length] for _w in weights_1]
        if weights_2 is not None:
            weights_2 = weights_2[:self.net_length]

        all_metrics = {}
        for km in self.key_metrics:
            all_metrics[km] = []

        if use_val:
            assert self.validation_set, 'validation set does not exist.'
            # use validation dataset to compute weights_2 updates
            images, labels = next(self.valloader)
            images = images.to(self.device)
            labels = labels.to(self.device)
            all_metrics['val_batch_loss'] = []
        else:
            w2_updates = get_updates(original_weights=current_weights, updated_weights=weights_2)
            w2_updates = [(update / lr).to(self.device) for update in w2_updates]
            w2_updates = dict(
                    zip(
                        list(self.model.state_dict().keys()),
                        w2_updates
                    )
            )        

        for w in weights_1:
            w1_updates = get_updates(original_weights=current_weights, updated_weights=w) 
            w1_updates = [(update / lr).to(self.device) for update in w1_updates]
            w1_updates = dict(
                    zip(
                        list(self.model.state_dict().keys()),
                        w1_updates
                    )
                )

            if use_val:
                set_weights(self.model, w)
                self.model.eval()

                oup = self.model(images)
                loss = F.cross_entropy(oup, labels)
                self.model.zero_grad()
                loss.backward()
                all_metrics['val_batch_loss'].append(loss.item())

                w2_updates = {
                    n: -p.grad for n, p in self.model.named_parameters()
                }

            if self.combined:
                if self.per_class:       
                    class_metrics = []             
                    for class_id in range(self.no_of_classes):
                        w1_update = torch.Tensor([]).to(self.device)
                        w2_update = torch.Tensor([]).to(self.device)
                        for layer_name in self.calc_weighting_layers:
                            assert w1_updates[layer_name].size(0) == self.no_of_classes
                            assert w2_updates[layer_name].size(0) == self.no_of_classes

                            _w1_update = w1_updates[layer_name][class_id]
                            _w2_update = w2_updates[layer_name][class_id]
                            if len(_w1_update.size()) == 0:
                                _w1_update = _w1_update.unsqueeze(0)
                                _w2_update = _w2_update.unsqueeze(0)
                            
                            w1_update = torch.cat([w1_update, _w1_update])
                            w2_update = torch.cat([w2_update, _w2_update])
                        
                        class_metrics.append(self.compute_metrics(w1_update, w2_update))

                    for km, m in zip(all_metrics, list(zip(*class_metrics))):
                        all_metrics[km].append(m)

                else:
                    w1_update = torch.Tensor([]).to(self.device)
                    w2_update = torch.Tensor([]).to(self.device)
                    for layer_name in self.calc_weighting_layers:
                        w1_update = torch.cat([w1_update, w1_updates[layer_name].flatten()])
                        w2_update = torch.cat([w2_update, w2_updates[layer_name].flatten()])
                    
                    metrics = self.compute_metrics(w1_update, w2_update)

                    for km, m in zip(all_metrics, metrics):
                        all_metrics[km].append(m)
            else:
                if self.per_class:
                    class_metrics = []
                    for class_id in range(self.no_of_classes):
                        tmp = {}
                        for km in self.key_metrics:
                            tmp[km] = []
                        for layer_name in self.calc_weighting_layers:
                            assert w1_updates[layer_name].size(0) == self.no_of_classes
                            assert w2_updates[layer_name].size(0) == self.no_of_classes
                            _w1_update = w1_updates[layer_name][class_id]
                            _w2_update = w2_updates[layer_name][class_id]
                            if len(_w1_update.size()) == 0:
                                _w1_update = _w1_update.unsqueeze(0)
                                _w2_update = _w2_update.unsqueeze(0)
                            
                            metrics = self.compute_metrics(_w1_update, _w2_update)
                            for km, m in zip(self.key_metrics, metrics):
                                tmp[km].append(m)
    
                        class_metrics.append([np.sum(tmp[km]) for km in self.key_metrics])

                    for km, m in zip(all_metrics, list(zip(*class_metrics))):
                        all_metrics[km].append(m)
                        
                else:
                    tmp = {}
                    for km in self.key_metrics:
                        tmp[km] = []

                    for layer_name in self.calc_weighting_layers:
                        metrics = self.compute_metrics(w1_updates[layer_name].flatten(), \
                                                    w2_updates[layer_name].flatten())
                        for km, m in zip(self.key_metrics, metrics):
                            tmp[km].append(m)

                    for km in self.key_metrics:
                        all_metrics[km].append(np.sum(tmp[km]))
        return all_metrics
