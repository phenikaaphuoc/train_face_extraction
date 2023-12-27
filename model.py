import torchvision.models as models
import torch.nn as nn
from utils import  read_yaml_file,caculate_metrix
import os
import glob
from data import get_dataloader
import torch.optim as optim
import torch
from torch.optim import lr_scheduler
import logging

logging.basicConfig(level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')
class Model(nn.Module):
    def __init__(self,opt_path:str):
        super(Model,self).__init__()

        self.opt = read_yaml_file(opt_path)
        self.model = models.resnet50(pretrained=True)
        self.num_class =  len(glob.glob(os.path.join(self.opt["train"]["data_dir"],"*")))
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1028),
            nn.BatchNorm1d(1028),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(1028, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, self.num_class)
        )

        if "pretrained_path" in self.opt:
            self.load_model(self.opt["pretrained_path"])
        self.dataloader = get_dataloader(self.opt['train'])
        self.val_dataloader = get_dataloader(self.opt['val'],train=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt['train']["learning_rate"],weight_decay =self.opt['train']['weight_decay'])
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.opt['train']['learning_schedule'], gamma=0.5)

        if "train" in self.opt:
            self.train()
    def train(self):
        count = 0
        save_dir = self.opt["val"]["save_dir"]
        os.makedirs(save_dir,exist_ok=True)
        device = self.opt["device"]
        logging.info("train on: ",device)
        self.model = self.model.to(device)
        total_epochs =self.opt["train"]["num_epochs"]
        self.model.train()
        for epoch in range(total_epochs):
            for i, (inputs, labels) in enumerate(self.dataloader):
                count+=1
                self.optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = self.model(inputs)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()
                del inputs
                del labels
                if count % self.opt['train']['print_freq'] == 0:
                    logging.info(f"epoch {epoch+1}/{total_epochs} , iter {count} train loss: {loss.item()}")
                if count% self.opt['val']['print_freq'] ==0:
                    loss_valid = caculate_metrix(self.model, self.val_dataloader, device, self.criterion)
                    logging.info(f"epoch {epoch + 1}/{total_epochs} , iter {count} val loss: {loss_valid}")
                    if count % self.opt['val']['save_freq'] == 0:
                        file_name = str(count)+"_"+str(loss_valid)+"_"+".pth"
                        file_path = os.path.join(save_dir,file_name)
                        self.save_model(file_path)
                self.scheduler.step()
    def save_model(self,file_path):
        torch.save(self.model.state_dict(), file_path)
        logging.info(f'Model state_dict saved to {file_path}')

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        logging.info(f'Model state_dict loaded from {file_path}')
