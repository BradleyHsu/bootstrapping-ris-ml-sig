from pytorch_lightning import Trainer, LightningModule
import torch
from torch import nn
from torchmetrics import Accuracy
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

""" In Multi GPU training, we need to define training_step_end and training_epoch_end
    because the training_step is called on each GPU and the training_step_end and
    training_epoch_end are called on the main GPU. These two functions are used to
    aggregate the results from each GPU. This is done automatically in single GPU
    training. Same applies to validation and test steps.
"""

class MyModule(LightningModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, **kwargs):
        super().__init__()
        self.layer = nn.Linear(28*28, 10)
        self.out = nn.Softmax(dim=1)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = kwargs.get('batch_size', 64)
        self.num_workers = kwargs.get('num_workers', 12)
        self.lr = kwargs.get('lr', 0.02)
        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy()

    def forward(self, x):
        return self.out(self.layer(x))
    
    def shared_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self(x)
        loss = self.criterion(pred, y)
        return loss, pred, y

    def training_step(self, batch, batch_idx):
        loss, pred, y = self.shared_step(batch, batch_idx)
        return {"train_loss": loss, "pred": pred, "y": y}
    
    def validation_step(self, batch, batch_idx):
        loss, pred, y = self.shared_step(batch, batch_idx)
        return {"val_loss": loss, "pred": pred, "y": y}
    
    def test_step(self, batch, batch_idx):
        loss, pred, y = self.shared_step(batch, batch_idx)
        return {"test_loss": loss, "pred": pred, "y": y}
    
    def training_step_end(self, outputs):
        loss = outputs['train_loss'].mean()
        pred = outputs['pred']
        y = outputs['y']
        acc = self.acc(pred, y)
        return {"loss": loss, "train_acc": acc}
    
    def validation_step_end(self, outputs):
        loss = outputs['val_loss'].mean()
        pred = outputs['pred']
        y = outputs['y']
        acc = self.acc(pred, y)
        return {"loss": loss, "val_acc": acc}
    
    def test_step_end(self, outputs):
        loss = outputs['test_loss'].mean()
        pred = outputs['pred']
        y = outputs['y']
        acc = self.acc(pred, y)
        return {"loss": loss, "test_acc": acc}
    
    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)
    
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc, prog_bar=True, sync_dist=True)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        shuffle=True,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=True,
                        pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                        shuffle=False,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=True,
                        pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                        shuffle=False,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=True,
                        pin_memory=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


""" Non Module level code should be defined in main function when using multi GPU training.
    Otherwise, it will be called on each GPU and will cause memory overflow error.
"""
if __name__ == '__main__':
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    val_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    val_dataset, test_dataset = random_split(val_dataset, [5000, 5000])
    model = MyModule(train_dataset, val_dataset, test_dataset)
    trainer = Trainer(accelerator='cuda', strategy='ddp_find_unused_parameters_false', devices=2, max_epochs=5, num_nodes=1)
    trainer.fit(model)
    trainer.test(model)
