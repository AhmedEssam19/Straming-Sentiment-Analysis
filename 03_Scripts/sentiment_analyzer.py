import torch
import torchmetrics

import pytorch_lightning as pl

from torch.nn import functional as F


class SentimentAnalyzer(pl.LightningModule):
    def __init__(self, model_name, output_units, dropout):
        super().__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers',
                                    'modelForSequenceClassification',
                                    model_name,
                                    num_labels=output_units,
                                    hidden_dropout_prob=dropout,
                                    )

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, targets = batch
        preds = self(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(preds.view(-1), targets)
        self.log('train_loss', loss)

        self.train_acc(preds.view(-1), targets.type(torch.int64))
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, targets = batch
        preds = self(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(preds.view(-1), targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_acc(preds.view(-1), targets.type(torch.int64))
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
