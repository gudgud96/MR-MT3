from torch.optim import AdamW
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5Config
from models.t5_detr import T5DETR
from utils import get_cosine_schedule_with_warmup
import pandas as pd
from .misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

class DETR(pl.LightningModule):

    def __init__(self, config, hungarian, optim_cfg):
        super().__init__()
        self.config = config
        self.hungarian = hungarian
        self.optim_cfg = optim_cfg
        T5config = T5Config.from_dict(OmegaConf.to_container(self.config))
        self.model: nn.Module = T5DETR(T5config)

        matcher = HungarianMatcher(**hungarian.matcher)
        
        self.criterion = SetCriterion(**hungarian.criterion,
                                      matcher = matcher)        

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # input.shape = (B, framesize, n_mels) = (B, 256, 512)
        # targets.shape = (B, n_tokens, 4)
        lm_logits = self.forward(inputs=inputs, labels=targets)

        loss_dict = self.criterion(lm_logits, targets)

        # the reason that we have the keys "labels" and "boxes" is to keep the code consistent with DETR
        total_loss = loss_dict['labels']['loss_ce'] * self.criterion.weight_dict['loss_ce'] + \
                     (loss_dict['boxes']['loss_pitch'] + \
                      loss_dict['boxes']['loss_onset'] + \
                      loss_dict['boxes']['loss_offset']) * self.criterion.weight_dict['loss_bbox']
                      
        self.log('train/pitch_loss', loss_dict['boxes']['loss_pitch'], prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/program_loss', loss_dict['labels']['loss_ce'], prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/onset_loss', loss_dict['boxes']['loss_onset'], prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/offset_loss', loss_dict['boxes']['loss_offset'], prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/loss', total_loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        if batch_idx == 0:
            if self.current_epoch == 0:
                self._log_text(targets, "train/targets", max_sentences=4, logger=self)
            # get top1 predictions out of the lm_logits tuples
            pitch_preds = lm_logits[0].argmax(-1)
            program_preds = lm_logits[1].argmax(-1)
            onset_preds = lm_logits[2].argmax(-1)
            offset_preds = lm_logits[3].argmax(-1)
            pred = torch.stack((pitch_preds, program_preds, onset_preds, offset_preds), dim=2)
            self._log_text(pred, "train/preds", max_sentences=4, logger=self)
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch 
        # input.shape = (B, framesize, n_mels) = (B, 256, 512)
        # targets.shape = (B, n_tokens, 4)
        lm_logits = self.forward(inputs=inputs, labels=targets)

        loss_dict = self.criterion(lm_logits, targets)

        # the reason that we have the keys "labels" and "boxes" is to keep the code consistent with DETR
        total_loss = loss_dict['labels']['loss_ce'] * self.criterion.weight_dict['loss_ce'] + \
                     (loss_dict['boxes']['loss_pitch'] + \
                      loss_dict['boxes']['loss_onset'] + \
                      loss_dict['boxes']['loss_offset']) * self.criterion.weight_dict['loss_bbox']

        self.log('val/pitch_loss', loss_dict['boxes']['loss_pitch'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/program_loss', loss_dict['labels']['loss_ce'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/onset_loss', loss_dict['boxes']['loss_onset'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/offset_loss', loss_dict['boxes']['loss_offset'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            if self.current_epoch == 0:
                self._log_text(targets, "val/targets", max_sentences=4, logger=self)
            pitch_preds = lm_logits[0].argmax(-1)
            program_preds = lm_logits[1].argmax(-1)
            onset_preds = lm_logits[2].argmax(-1)
            offset_preds = lm_logits[3].argmax(-1)
            pred = torch.stack((pitch_preds, program_preds, onset_preds, offset_preds), dim=2)
            self._log_text(pred, "val/preds", max_sentences=4, logger=self)
        # no need to use it in this stage
        # return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.optim_cfg.lr)
        warmup_step = int(self.optim_cfg.warmup_steps)
        print('warmup step: ', warmup_step)
        schedule = {
            'scheduler': get_cosine_schedule_with_warmup(
                optimizer=optimizer, 
                num_warmup_steps=warmup_step, 
                num_training_steps=self.optim_cfg.num_steps_per_epoch * self.optim_cfg.num_epochs,
                min_lr=self.optim_cfg.min_lr
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [schedule]

        # we follow MT3 to use fixed learning rate
        # NOTE: we find this to not work :(
        # return AdamW(self.model.parameters(), self.config.lr)

    def _log_text(self, token_seqs, tag, max_sentences, logger):       
        plugin_list = []
        for idx, token_seq in enumerate(token_seqs):
            if idx < max_sentences: 
                token_str = ''
                for token in token_seq:
                    pitch = token[0].item()
                    program = token[1].item()
                    onset = token[2].item()
                    offset = token[3].item()
                    token_str = token_str + \
                    f"<{pitch}, {program}, {onset}, {offset}>" + ', '
                plugin_list.append(token_str)        
        s = pd.Series(plugin_list, name="token sequence")
        logger.logger.experiment.add_text(tag, s.to_markdown(), global_step=self.global_step)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_classes, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_classes = num_classes
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def out_of_bound_assignment(self, tgt, outbound_idx):
        tgt[tgt < 0] = outbound_idx
        tgt[tgt > outbound_idx] = outbound_idx        

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        pitch_pred = outputs[0]
        instrument_pred = outputs[1]
        onset_pred = outputs[2]
        offset_pred = outputs[3]

        bs, num_queries = pitch_pred.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = instrument_pred.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox_pitch = pitch_pred.flatten(0, 1)  # [batch_size * num_queries, 128+1]
        out_bbox_onset = onset_pred.flatten(0, 1)  # [batch_size * num_queries, 256+1]
        out_bbox_offset = offset_pred.flatten(0, 1)  # [batch_size * num_queries, 256+1]

        # Also concat the target labels and boxes
        tgt = torch.cat([v for v in targets])
        tgt_pitch = tgt[:, 0].long()
        tgt_instru = tgt[:, 1].long()
        tgt_onset = tgt[:, 2].long()
        tgt_offset = tgt[:, 3].long()

        # convert out of bound frames to a special class
        self.out_of_bound_assignment(tgt_onset, self.num_classes[2])
        self.out_of_bound_assignment(tgt_offset, self.num_classes[3])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_instru]

        # It was originally L1 loss
        # But for our case, the boxes (pitch, onset, offset) are all classifications
        # will try the same classification cost first
        cost_pitch = -out_bbox_pitch[:, tgt_pitch]
        cost_onset = -out_bbox_onset[:, tgt_onset]
        cost_offset = -out_bbox_offset[:, tgt_offset]

        cost_bbox = cost_pitch + cost_onset + cost_offset

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_pitch_weight = torch.ones(self.num_classes[0]+1) # added one class for "not found"
        empty_program_weight = torch.ones(self.num_classes[1]+1)
        empty_onset_weight = torch.ones(self.num_classes[2]+1)
        empty_offset_weight = torch.ones(self.num_classes[3]+1)

        empty_pitch_weight[-1] = self.eos_coef[0]
        empty_program_weight[-1] = self.eos_coef[1]
        empty_onset_weight[-1] = self.eos_coef[2]
        empty_offset_weight[-1] = self.eos_coef[3]

        self.register_buffer('empty_pitch_weight', empty_pitch_weight)
        self.register_buffer('empty_program_weight', empty_program_weight)
        self.register_buffer('empty_onset_weight', empty_onset_weight)
        self.register_buffer('empty_offset_weight', empty_offset_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs[1]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[:,1][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes[1],
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_program_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        pitch_logits = outputs[0]
        onset_logits = outputs[2]
        offset_logits = outputs[3]

        idx = self._get_src_permutation_idx(indices)
        pitch_logits = pitch_logits[idx]
        onset_logits = onset_logits[idx]
        offset_logits = offset_logits[idx]

        target_pitch = torch.cat([t[:,0][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_onset = torch.cat([t[:,2][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_offset = torch.cat([t[:,3][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # assign out of bound frames to a special class
        target_onset[target_onset < 0] = self.num_classes[2]
        target_offset[target_offset > self.num_classes[3]] = self.num_classes[3]

        loss_pitch = F.cross_entropy(pitch_logits, target_pitch, reduction='none')
        loss_onset = F.cross_entropy(onset_logits, target_onset, reduction='none')
        loss_offset = F.cross_entropy(offset_logits, target_offset, reduction='none')

        losses = {}
        losses['loss_pitch'] = loss_pitch.sum() / num_boxes
        losses['loss_onset'] = loss_onset.sum() / num_boxes
        losses['loss_offset'] = loss_offset.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=targets[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        losses['labels'] = self.loss_labels(outputs, targets, indices, num_boxes)
        losses['boxes'] = self.loss_boxes(outputs, targets, indices, num_boxes)

        return losses