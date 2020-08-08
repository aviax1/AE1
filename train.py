!pip install wandb
import torch,wandb,os,warnings,csv
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from tensorflow.keras.datasets import mnist
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

(xtrain,ytrain), (xtest,ytest) = mnist.load_data()
hyperparameter_defaults = dict(
    dropout = 0.5,
    hd1 = 16,
    bt = 100,
    learning_rate = 0.001,
    epochs = 200,
)
wandb.init(config=hyperparameter_defaults, project="ae1")

config = wandb.config
num_epochs=config.epochs        #
batch_size =config.bt  #
image_size=784         #
hidden_size=config.hd1 #
lv_size = 48           # Latent Variable 
learning_rate=config.learning_rate     #
cret = nn.MSELoss()    # criterion
warnings.filterwarnings('ignore')

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, hidden_size), 
            nn.ReLU(True), nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True), nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True), nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True), nn.Linear(hidden_size, lv_size))
        self.dropout = nn.Dropout(p=config.dropout)
        self.decoder = nn.Sequential(
            nn.Linear(lv_size, hidden_size),nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),nn.ReLU(True),
            nn.Linear(hidden_size, image_size), nn.Tanh())

    def forward(self, x):
        return self.decoder(self.encoder(x))
      
model = autoencoder()
wandb.watch(model)
tmodel=autoencoder()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

class DigitDataSet(Dataset):
  def __init__(self, dataset):
      self.dataset = dataset
      self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

  def __len__(self):
      return len(self.dataset)

  def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.transform( self.dataset[idx,:,:])
    

def model_name(digit):
  return './ae_'+str(digit)+'.pth'

def get_prediction(data=xtest):
  nn=len(data)
  dataloader = DataLoader(DigitDataSet(data), batch_size=nn,shuffle=0 , num_workers=4)
  diff = np.zeros( (nn,10),dtype=np.float32 )
  for i in range(10):
    for data in dataloader:
      input_imgs = data
      imgs = Variable(input_imgs.view(input_imgs.size(0), -1))
      tmodel.load_state_dict(torch.load(model_name(i)))
      tmodel.eval()
      output_imgs = tmodel(imgs)
      for i2 in range(len( output_imgs[:,0])):
        im_pred=output_imgs.detach().numpy()[i2,:]
        im_org=imgs.numpy()[i2,:]
        difmat=np.abs(im_pred.reshape(28,28)-im_org.reshape(28,28))
        diff[i2,i]=np.sum( np.sum( difmat ))
  return np.argmin(diff, axis=1)


def testmodel(): 
  nn=len(ytest)
  min_index =get_prediction()
  seccess =  min_index == ytest
  counts, bins = np.histogram(ytest[ min_index != ytest ])
  plt.hist(bins[:-1], bins, weights=counts)
  plt.title("error by digit")
  plt.show()
  accurcy =int(10000*np.sum(seccess))/(nn*100)
  error_rate = int(10000*np.sum(min_index != ytest))/(nn*100)
  print(str(accurcy) + "% accuracy or "+str(error_rate)+"% error rate")
  return counts, bins ,len(ytest[min_index != ytest]) , len(ytest)



def save_model(digit,model):
  mn=model_name(digit)
  torch.save(model.state_dict(),mn )
  wandb.save(mn)
  print("save model "+ mn)

def load_model_ifexist(digit,model):
  mn=model_name(digit)
  if os.path.isfile(mn):
    model.load_state_dict(torch.load(mn))
    model.eval()
  return model

def train_by_digit(by_digit,model,ne=num_epochs,opt=optimizer):
  model=load_model_ifexist( by_digit,model)
  print("*****\nstart traning Model for digit " +str(by_digit) +"\n")
  dataloader = DataLoader(DigitDataSet(xtrain[ytrain==by_digit]), batch_size=config.bt,shuffle=True, num_workers=6)
  for epoch in range(ne):
    run=  epoch%25==0
    run2= epoch%125==0 and epoch >0
    for data in dataloader:
      imgs = Variable(data.view(data.size(0), -1))
      output_imgs = model(imgs)
      loss = cret(output_imgs, imgs)
      opt.zero_grad()
      loss.backward()
      opt.step()
      if run:
        run=0
        im=data[0,0,:,:].reshape(28,28)
        pred=model(imgs).detach().numpy()[0,:].reshape(28,28)
        wandb.log({"img": [wandb.Image(pred, caption="preidciton"),wandb.Image(im, caption="original")]})
      if run2:
        run2=0
        save_model(by_digit,model)
        testmodel()
        
    print('epoch [{}/{}], loss:{:.4f}' .format(epoch + 1, ne, loss.data))
    wandb.log({"loss": loss.data})
  save_model(by_digit,model)
  print("\nfinish traning Model Number " +str(by_digit) +"\n*****\n")
  

for by_digit in range(10):
  train_by_digit(by_digit,model,120)
