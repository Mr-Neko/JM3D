'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

class ULIPWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)  

        if 'origin_image_embed' in outputs:
            origin_image_embed = outputs['origin_image_embed']

            origin_image_embed = F.normalize(origin_image_embed, dim=-1, p=2)
            score = torch.einsum('bnc, bc -> bn', origin_image_embed, text_embed).squeeze()
            score = F.softmax(score, dim=-1) # b, n
            image_embed = torch.sum(image_embed * score.unsqueeze(dim=-1), dim=1)

        # image_embed = torch.mean(image_embed, dim=1)
        # gather features from all GPUs
        pc_embed_all, text_embed_all, image_embed_all = \
            utils.all_gather_batch([pc_embed, text_embed, image_embed])

        # cosine similarity as logits
        logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()
        # logits_per_pc_image = logit_scale * torch.einsum('bc, anc -> ban', pc_embed, image_embed_all)
        # logits_per_image_pc = logit_scale * torch.einsum('bnc, ac -> bna', image_embed, pc_embed_all).permute(0, 2, 1)

        logits_per_text_image = logit_scale * text_embed @ image_embed_all.t()
        logits_per_image_text = logit_scale * image_embed @ text_embed_all.t()
        # logits_per_text_image = logit_scale * torch.einsum('bc, anc -> ban', text_embed, image_embed_all)
        # logits_per_image_text = logit_scale * torch.einsum('bnc, ac -> bna', image_embed, text_embed_all).permute(0, 2, 1)

        ulip_loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
                F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
                (F.cross_entropy(logits_per_pc_image, self.labels) + \
                F.cross_entropy(logits_per_image_pc, self.labels)) / 2 + \
                (F.cross_entropy(logits_per_text_image, self.labels) + \
                F.cross_entropy(logits_per_image_text, self.labels)) / 2
                # ((F.cross_entropy(logits_per_pc_image, self.labels.unsqueeze(dim=-1).repeat(1, score.shape[1]), reduction='none')).mean() + \
                #   (F.cross_entropy(logits_per_image_pc, self.labels.unsqueeze(dim=-1).repeat(1, score.shape[1]), reduction='none')).mean()) / 2 + \
                # ((F.cross_entropy(logits_per_text_image, self.labels.unsqueeze(dim=-1).repeat(1, score.shape[1]), reduction='none')).mean() + \
                #    (F.cross_entropy(logits_per_image_text, self.labels.unsqueeze(dim=-1).repeat(1, score.shape[1]), reduction='none')).mean()) / 2
                # (F.cross_entropy(logits_per_text_image, self.labels) + F.cross_entropy(logits_per_image_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_pc_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_pc_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_image_acc = 100 * correct / local_batch_size

        loss_classed = 0
        soft_loss = 0
        if 'labels' in outputs:

            loss_classed = F.cross_entropy(outputs['class_output'], outputs['labels']) / 2

        if 'soft_labels' in outputs:

            soft_labels = F.log_softmax(outputs['soft_labels'] + 0.001, dim=-1)

            soft_loss = (F.kl_div(F.log_softmax(logits_per_pc_text, dim=-1), soft_labels, reduction='batchmean') + \
                     F.kl_div(F.log_softmax(logits_per_text_pc, dim=-1), soft_labels, reduction='batchmean')) / 2 + \
                     (F.kl_div(F.log_softmax(logits_per_pc_image, dim=-1), soft_labels, reduction='batchmean') + \
                     F.kl_div(F.log_softmax(logits_per_image_pc, dim=-1), soft_labels, reduction='batchmean')) / 2 + \
                     (F.kl_div(F.log_softmax(logits_per_text_image, dim=-1), soft_labels, reduction='batchmean') + \
                     F.kl_div(F.log_softmax(logits_per_image_text, dim=-1), soft_labels, reduction='batchmean')) / 2
        
        loss = ulip_loss + soft_loss + loss_classed

        if 'labels' in outputs:

            if 'soft_labels' in outputs:
                return {'loss': loss, 'ulip_loss': ulip_loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc, 'soft_loss': soft_loss, 'class_loss': loss_classed}
            
            else:
                return {'loss': loss, 'ulip_loss': ulip_loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc, 'class_loss': loss_classed}
        
        else:

            if 'soft_labels' in outputs:
                return {'loss': loss, 'ulip_loss': ulip_loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc, 'soft_loss': soft_loss}
            
            else:
                return {'loss': loss, 'ulip_loss': ulip_loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc}


