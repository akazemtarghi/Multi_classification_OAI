import torch
import random
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import GroupKFold
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import torchvision.transforms as transforms
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from functools import partial
from threading import Thread
from tornado import gen
import sys
import torch
from tensorboardX import SummaryWriter
import torchvision

def tensorboardx(train_dataset, writer, model):
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              pin_memory=False)
    data = next(iter(trainloader))
    images = data['image']
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    images = images.to(device)
    writer.add_graph(model, images, verbose=False)

def set_ultimate_seed(base_seed=777):
    import random
    random.seed(base_seed)

    try:
        import numpy as np
        np.random.seed(base_seed)
    except ModuleNotFoundError:
        print('Module `numpy` has not been found')
    try:
        import torch
        torch.manual_seed(base_seed + 1)
        torch.cuda.manual_seed_all(base_seed + 2)
    except ModuleNotFoundError:
        print('Module `torch` has not been found')

def SplittingData (root='C:/Users/Amir Kazemtarghi/Documents/INTERNSHIP/data/IDnKL.csv',
                   Ratio = 0.2):

    """ This function split the data into train and test with rate of 4:1
        data from same ID remain in same group of train or test.
    """

    file = pd.read_csv(root)
    file1 = file.copy()
    file1.drop_duplicates(subset=['ID'], inplace=True)
    file1 = file1.reset_index(drop=True)
    file2 = file1['ID']
    random.shuffle(file2)
    Dataset_size = len(file2)
    split = int(np.floor(Ratio * Dataset_size))
    train_indices, test_indices = file2[split:], file2[:split]
    Train_split = file.loc[file['ID'].isin(train_indices)]
    Train_split = Train_split.reset_index(drop=True)
    Train_split = Train_split.drop(columns=['Unnamed: 0'])
    test_split = file.loc[file['ID'].isin(test_indices)]
    test_split = test_split.reset_index(drop=True)
    test_split = test_split.drop(columns=['Unnamed: 0'])

    return Train_split, test_split

def GroupKFold_Amir(input, n_splits):
    X = input
    y = X.landmarks_frame.KL[:]
    y = y.reset_index(drop=True)
    groups = X.landmarks_frame.ID[:]
    group_kfold = GroupKFold(n_splits)
    group_kfold.get_n_splits(X, y, groups)
    print(group_kfold)
    return group_kfold.split(X, y, groups)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

class OAIdataset(Dataset):

    """datasetA."""

    def __init__(self, csv_file, root_dir, transform):
        """
        Args:
            csv_file (string): Path to the csv file with KL grade and ID.
            root_dir (string): Directory with all the images.

        """
        self.landmarks_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,
                                str(self.landmarks_frame['ID'].iloc[idx]))

        img_name = img_name + '.npy'
        patches, p_id = np.load(img_name)

        if self.landmarks_frame['SIDE'].iloc[idx] == 1:
            image = patches['R']
        else:
            image = patches['L']
        imageID = self.landmarks_frame['ID'].iloc[idx]
        landmarks = self.landmarks_frame['KL'].iloc[idx]
        if self.transform:
           image = self.transform(image)

        sample = {'image': image, 'landmarks': landmarks, 'imageID': imageID}
        return sample

class Amir(nn.Module):
    def __init__(self, nclass):

        super(Amir,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU())

        self.fc = nn.Sequential(nn.Dropout(),
            nn.Linear(8192, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, nclass))


    def forward(self, x):
        t = self.layer1(x)
        t = self.layer2(t)
        t = self.layer3(t)
        t = self.layer4(t)
        t = t.view(x.size(0), -1)
        t = self.fc(t)

        return t


def Training_dataset(data_loaders, model, patience, n_epochs, namefold):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)


    for epoch in range(n_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            # Iterate over data.
            for data in data_loaders[phase]:
                optimizer.zero_grad()
                # get the input images and their corresponding labels
                images = data['image']
                key_pts = data['landmarks']
                images = images.to(device)
                key_pts = key_pts.to(device)

                # wrap them in a torch Variable
                output_pts = model(images)

                # calculate the loss between predicted and target keypoints
                loss = criterion(output_pts, key_pts)
                # zero the parameter (weight) gradients
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()
                    train_losses.append(loss.item())

                # print loss statistics
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)


        writer.add_scalars(namefold, {'Train loss': train_loss,
                                      'Valid loss': valid_loss}, epoch)
        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, avg_train_losses, avg_valid_losses


def Testing_dataset(testloader, model,y_score_sum):
    y_score = torch.ones(1, 5)
    y = []
    y_score = y_score.cpu().numpy()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images = data['image']
            label = data['landmarks']
            images = images.to(device)
            label1 = label.to(device)
            output1 = model(images)
            softmax = nn.Softmax()
            output2 = softmax(output1)
            _, predicted = torch.max(output2.data, 1)
            output2 = output2.cpu().numpy()
            label2 = label1.cpu().numpy()
            y_score = np.append(y_score, output2, axis=0)
            y = np.append(y, label2, axis=0)
            total += label1.size(0)
            correct += (predicted == label1).sum().item()

        y_score = np.delete(y_score, 0, axis=0)
        y_score_sum = y_score_sum + y_score
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        return y_score_sum, y

def roc_curve(y_score_sum, y):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # creating boolean matrix instead of one array
    y_score_ave = y_score_sum / 5
    t1_indice = np.where(y == 0)
    t2_indice = np.where(y == 1)
    t3_indice = np.where(y == 2)
    t4_indice = np.where(y == 3)
    t5_indice = np.where(y == 4)

    Y = np.zeros((len(y), 5))
    Y[t1_indice, 0] = 1
    Y[t2_indice, 1] = 1
    Y[t3_indice, 2] = 1
    Y[t4_indice, 3] = 1
    Y[t5_indice, 4] = 1
    # drop the fist row which is ones
    # y_score = np.delete(y_score_ave, 0, axis=0)
    # Computing ROC and ROC AUC
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_score_ave[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc, fpr, tpr


writer = SummaryWriter('TensorboardX')

Transforms = transforms.Compose([transforms.ToPILImage(),
                                 transforms.RandomRotation(5),
                                 transforms.RandomAffine(5),
                                 transforms.ToTensor()])

Transforms1 = transforms.Compose([transforms.ToTensor()])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_tensor_type(torch.cuda.FloatTensor)

batch_size = 50
nclass = 5
Epoch = 100
learning_rate = 0.001

model = Amir(nclass).to(device)
A = Amir(nclass)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_Csv, test_Csv = SplittingData()

train_dataset = OAIdataset(csv_file=train_Csv,
                           root_dir='C:/Users/Amir Kazemtarghi/Documents/INTERNSHIP/data/DatabaseA/',
                           transform=Transforms)

test_dataset = OAIdataset(csv_file=test_Csv,
                          root_dir='C:/Users/Amir Kazemtarghi/Documents/INTERNSHIP/data/DatabaseA/',
                          transform=Transforms1)

testloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         num_workers=0,
                                         pin_memory=False)

Groupkfold = GroupKFold_Amir(train_dataset, n_splits=5)


y_score_sum = np.zeros((1778, 5))
patience = 3
tensorboardx(train_dataset, writer, A)
nfold = 1

for train_index, test_index in Groupkfold:
    namefold = 'Fold' + str(nfold)
    model = Amir(nclass).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_subset = torch.utils.data.Subset(train_dataset, train_index)
    valid_subset = torch.utils.data.Subset(train_dataset, test_index)

    trainloader = torch.utils.data.DataLoader(train_subset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              pin_memory=False)
    validloader = torch.utils.data.DataLoader(valid_subset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              pin_memory=False)

    data_loaders = {"train": trainloader, "val": validloader}
    data_lengths = {"train": len(trainloader), "val": len(validloader)}
    model, train_loss, valid_loss = Training_dataset(data_loaders, model, patience, Epoch, namefold)
    y_score_sum, y = Testing_dataset(testloader, model, y_score_sum)
    nfold = nfold + 1

# Computing ROC
roc_auc, fpr, tpr = roc_curve(y_score_sum, y)

# plotting ROC
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','gray'])
plt.switch_backend('agg')
fig1 = plt.figure()
for i, color in zip(range(5), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
writer.add_figure('roc', fig1)


# visualize the loss as the network trained
fig = plt.figure()
plt.plot(range(1,len(train_loss)+1), train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1), valid_loss, label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')
writer.add_figure('minimum loss', fig)







