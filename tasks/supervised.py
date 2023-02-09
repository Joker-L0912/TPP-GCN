import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses
from utils import metrics
from utils import losses
import numpy as np
import pandas as pd


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            regressor="linear",
            loss="mse",
            pre_len: int = 3,
            learning_rate: float = 1e-3,
            weight_decay: float = 1.5e-3,
            feat_max_val: float = 1.0,
            **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self._loss = loss
        self.feat_max_val = feat_max_val
        self.turth_num = []
        self.pre_num = []
        self.i = 0

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y = batch
        num_nodes = x.size(2)
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        # y = y * self.feat_max_val
        # predictions = predictions * self.feat_max_val
        # -------------------
        # self.i = self.i + 1
        # if self.i >= 6590 and self.i <= 6600:
        # print(self.i)
        # print('追加....')
        # # for a in range(len(y)):
        # self.turth_num.append(y[3].cpu())
        # # for b in range(len(predictions)):
        # self.pre_num.append(predictions[3].cpu())
        # if self.i == 6600:
        #     print('保存.....')
        #     pd.DataFrame(self.pre_num).to_csv('./res_csv/test_pre.csv')
        #     pd.DataFrame(self.turth_num).to_csv('./res_csv/test_y.csv')
        # _-------------------

        # if self.i % 10999 == 0:
        #     pd.DataFrame(predictions).to_csv('./res_csv/pre.csv')
        #     pd.DataFrame(y).to_csv('./res_csv/y.csv')
        # if i % 56 == 0:
        #     ep = 0
        #     pd.DataFrame(predictions).to_csv('./res_csv/pre-'+str(ep)+'.csv')
        #     pd.DataFrame(y).to_csv('./res_csv/y-' + str(ep) + '.csv')
        # # print("--------------------")
        # print("predictions",predictions)
        # print('\n')
        # print("y", y)
        # print('\n')
        # print("loss", loss)
        # print('--------------------')
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val
        y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        mape = utils.metrics.mape(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        # print('val_loss:',loss)
        metrics = {
            "MAPE": mape,
            "RMSE": rmse,
            "MAE": mae,
            # "R2": r2,
            "loss": loss
        }
        self.log_dict(metrics)
        self.i += 1
        if self.i == 200:
            pd.DataFrame(predictions.cpu().numpy()).to_csv(r'data/predictions_TGCN_12.csv', index=False, header=False)
            pd.DataFrame(y.cpu().numpy()).to_csv(r'data/y_TGCN_12.csv', index=False, header=False)

        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=0)
        parser.add_argument("--loss", type=str, default="mse")
        return parser
