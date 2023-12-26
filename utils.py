import torch
import yaml
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file {file_path}: {e}")
            return None
def caculate_metrix(model,dataloader,device,loss_fn):
    model = model.to(device)
    loss = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for x,y in dataloader:
            count += x.shape[0]
            x,y = x.to(device),y.to(device)
            y_pred = model(x)
            loss+= loss_fn(y_pred,y).item()
            del x
            del y
    model.train()
    return loss/count