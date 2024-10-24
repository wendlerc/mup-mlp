import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import argparse
import random
import wandb


class CIFAR10MLP(pl.LightningModule):
    def __init__(self, 
                 input_size=3072, 
                 hidden_size=512, 
                 num_classes=10, 
                 lr=1e-3,
                 use_mup=False,
                 prefactor=2**0.5,
                 num_epochs=None,
                 optimizer='sgd',
                 follow_table=False):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.use_mup = use_mup
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.optimizier = optimizer
        self.follow_table = follow_table

        if use_mup:
            self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
            self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.fc3 = nn.Linear(hidden_size, num_classes, bias=False)
            
            std_dev1 = prefactor/input_size**0.5 * min(1, hidden_size**0.5 / input_size**0.5)
            std_dev2 = prefactor/hidden_size**0.5 * min(1, hidden_size**0.5 / hidden_size**0.5)
            std_dev3 = prefactor/hidden_size**0.5 * min(1, num_classes**0.5 / hidden_size**0.5)
            nn.init.normal_(self.fc1.weight, mean=0, std=std_dev1)
            nn.init.normal_(self.fc2.weight, mean=0, std=std_dev2)
            nn.init.normal_(self.fc3.weight, mean=0, std=std_dev3)

        else:
            self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
            self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.fc3 = nn.Linear(hidden_size, num_classes, bias=False)

        self.a1_0 = None
        self.h1_0 = None
        self.a2_0 = None
        self.h2_0 = None
        self.a3_0 = None
        self.x_0 = None

        self.W0_0 = self.fc1.weight.detach().cpu()
        self.W1_0 = self.fc2.weight.detach().cpu()
        self.W2_0 = self.fc3.weight.detach().cpu()
        self.sigma_max_0 = torch.svd(self.W0_0).S[0]
        self.sigma_max_1 = torch.svd(self.W1_0).S[0]
        self.sigma_max_2 = torch.svd(self.W2_0).S[0]



    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        a1 = self.fc1(x)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        h2 = F.relu(a2)
        a3 = self.fc3(h2)
        if self.a1_0 is None:
            self.a1_0 = a1.detach().cpu()
            self.h1_0 = h1.detach().cpu()
            self.a2_0 = a2.detach().cpu()
            self.h2_0 = h2.detach().cpu()
            self.a3_0 = a3.detach().cpu()
            self.x_0 = x.detach().cpu()
        return a3, {"a1": a1, "h1": h1, "a2": a2, "h2": h2, "a3": a3}

    def log_changes(self, run):
        W0 = self.fc1.weight.detach().cpu()
        W1 = self.fc2.weight.detach().cpu()
        W2 = self.fc3.weight.detach().cpu()

        # log spectral norm of W0 - W0_0
        run.summary['spectral_norm_W0'] = torch.svd(W0 - self.W0_0).S[0]/self.sigma_max_0
        run.summary['spectral_norm_W1'] = torch.svd(W1 - self.W1_0).S[0]/self.sigma_max_1
        run.summary['spectral_norm_W2'] = torch.svd(W2 - self.W2_0).S[0]/self.sigma_max_2

        # log changes in the activations of the first batch
        y_hat_first, latents_first = self(self.x_0.to(self.device))
        for latent_name, latent in latents_first.items():
            change = torch.norm(latent.detach().cpu() - getattr(self, latent_name + '_0'), dim=-1)
            change /= torch.norm(getattr(self, latent_name + '_0'), dim=-1)
            run.summary[f'act_change_{latent_name}'] = change.mean().item()


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, latents = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, sync_dist=True)

        # on the logging step we need to do some additional computations
        if self.global_step % self.trainer.log_every_n_steps == 0:

            for latent_name, latent in latents.items():
                self.log(f'act norm {latent_name}', latent.norm(dim=-1).mean(), sync_dist=True)
                self.log(f'act mean(abs({latent_name}))', torch.mean(torch.abs(latent)), sync_dist=True)
            
            # Log gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.log(f'grad norm {name}', param.grad.norm(), sync_dist=True)
                    self.log(f'grad mean(abs(grad_{name}))', torch.mean(torch.abs(param.grad)), sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, sync_dist=True, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test_acc', acc, sync_dist=True, on_step=False, on_epoch=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        if self.use_mup:
            hidden_size = self.hidden_size
            input_size = self.input_size
            num_classes = self.num_classes
            if self.optimizier == 'sgd':
                lr_fc1 = self.lr * hidden_size/input_size
                lr_fc2 = self.lr * hidden_size/hidden_size
                lr_fc3 = self.lr * num_classes/hidden_size

                param_groups = [
                    {'params': self.fc1.parameters(), 'lr': lr_fc1},
                    {'params': self.fc2.parameters(), 'lr': lr_fc2},
                    {'params': self.fc3.parameters(), 'lr': lr_fc3},
                ]

                optimizer = torch.optim.SGD(param_groups)
            elif self.optimizier == 'adam':
                # just using the numbers in the table
                if self.follow_table:
                    lr_fc1 = self.lr * 1.
                    lr_fc2 = self.lr * 1./hidden_size
                    lr_fc3 = self.lr * 1./hidden_size
                else:
                    lr_fc1 = self.lr * 1./input_size
                    lr_fc2 = self.lr * 1./hidden_size
                    lr_fc3 = self.lr * num_classes/hidden_size

                param_groups = [
                    {'params': self.fc1.parameters(), 'lr': lr_fc1},
                    {'params': self.fc2.parameters(), 'lr': lr_fc2},
                    {'params': self.fc3.parameters(), 'lr': lr_fc3},
                ]
                optimizer = torch.optim.Adam(param_groups)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizier}")
        else:
            if self.optimizier == 'sgd':
                optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            elif self.optimizier == 'adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizier}")
        return optimizer


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train CIFAR10 MLP model')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--width', type=int, default=1024, help='Network width (hidden layer size)')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--use_mup', action='store_true', help='Use μP scaling')
    parser.add_argument('--follow_table', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='cifar10-mlp', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default='chrisxx', help='Weights & Biases entity name')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use (sgd or adam)')
    args = parser.parse_args()

    # Data preparation
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model training
    model = CIFAR10MLP(hidden_size=args.width, 
                       lr=args.lr, 
                       use_mup=args.use_mup, 
                       num_epochs=args.n_epochs,
                       optimizer=args.optimizer,
                       follow_table=args.follow_table)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints',
        filename='cifar10-mlp-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
    )


    run_name = f"{'mup_' if args.use_mup else 'std_'}{args.optimizer}_w{args.width}_lr{args.lr}_e{args.n_epochs}_b{args.batch_size}_{random.randint(1000, 9999)}"
    
    if args.use_wandb:
        logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, name=run_name)
    else:
        logger = TensorBoardLogger("tb_logs", name=run_name)

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)
    print(logger.experiment)
    if args.use_wandb:
        model.log_changes(logger.experiment)

    # Test the model
    # Use a single device for testing to ensure each sample is evaluated once
    test_trainer = pl.Trainer(devices=1, num_nodes=1, logger=logger)
    test_trainer.test(model, val_loader)

