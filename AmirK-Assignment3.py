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
torch.cuda.current_device()




transforms = transforms.Compose([transforms.ToTensor()])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 40
nclass = 5
Epoch = 10
learning_rate = 0.001

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
            nn.Sigmoid(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Sigmoid())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Sigmoid())

        self.fc = nn.Sequential(nn.Dropout(),
            nn.Linear(7200, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, nclass))


    def forward(self, x):
        t = self.layer1(x)
        t = self.layer2(t)
        t = self.layer3(t)
        t = self.layer4(t)
        t = t.view(x.size(0), -1)
        t = self.fc(t)

        return t

model = Amir(nclass).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_Csv, test_Csv = SplittingData()

train_dataset = OAIdataset(csv_file=train_Csv,
                           root_dir='C:/Users/Amir Kazemtarghi/Documents/INTERNSHIP/data/DatabaseA/',
                           transform=transforms)
test_dataset = OAIdataset(csv_file=test_Csv,
                          root_dir='C:/Users/Amir Kazemtarghi/Documents/INTERNSHIP/data/DatabaseA/',
                          transform=transforms)
testloader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=0,
                                          pin_memory=False)

g = GroupKFold_Amir(train_dataset,n_splits=5)

Data = {'epochs': [],'trainlosses': [],'vallosses': [] }
source = ColumnDataSource(Data)

plot = figure()
plot.line(x= 'epochs', y='trainlosses',
 color='green', alpha=0.8, legend='Train loss', line_width=2,
 source=source)
plot.line(x= 'epochs', y='vallosses',
 color='red', alpha=0.8, legend='Val loss', line_width=2,
 source=source)

doc = curdoc()
# Add the plot to the current document
doc.add_root(plot)
@gen.coroutine
def update(new_data):
    source.stream(new_data)
valid_loss = []

for train_index, test_index in g:
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

    for epoch in range(Epoch):
        print('Epoch {}/{}'.format(epoch, Epoch - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for data in data_loaders[phase]:
                optimizer.zero_grad()

                # get the input images and their corresponding labels
                images = data['image']
                key_pts = data['landmarks']
                images = images.to(device)
                key_pts = key_pts.to(device)
                # wrap them in a torch Variable
                images, key_pts = Variable(images), Variable(key_pts)
                output_pts = model(images)
                # calculate the loss between predicted and target keypoints
                loss = criterion(output_pts, key_pts)
                # zero the parameter (weight) gradients
                optimizer.zero_grad()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()

            epoch_loss = running_loss / data_lengths[phase]

            if phase == 'train':
                train_loss = epoch_loss # Set model to training mode
            else:
                valid_loss = epoch_loss

            new_data = {'epochs': [epoch],
                        'trainlosses': [train_loss],
                        'vallosses': [valid_loss]}
            doc.add_next_tick_callback(partial(update, new_data))


            print('{} Loss: {:.4f}'.format(phase, epoch_loss))




fpr = dict()
tpr = dict()
roc_auc = dict()
y = []
y_score = torch.ones(1, 5)
y_score = y_score.cpu().numpy()
n=0
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    i=0
    for data in testloader:
        images = data['image']
        key_pts = data['landmarks']
        images = images.to(device)
        key_pts = key_pts.to(device)
        outputs = model(images)
        _ , predicted = torch.max(outputs.data, 1)
        OUTPUT = outputs.cpu().numpy()
        KEY_PTS = key_pts.cpu().numpy()
        y_score = np.append(y_score, OUTPUT, axis=0)
        y = np.append(y, KEY_PTS, axis=0)
        total += key_pts.size(0)
        correct += (predicted == key_pts).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))


# creating boolean matrix instead of one array
T1_indice = np.where(y == 0)
T2_indice = np.where(y == 1)
T3_indice = np.where(y == 2)
T4_indice = np.where(y == 3)
T5_indice = np.where(y == 4)

Y = np.zeros((len(y), 5))
Y[T1_indice, 0] = 1
Y[T2_indice, 1] = 1
Y[T3_indice, 2] = 1
Y[T4_indice, 3] = 1
Y[T5_indice, 4] = 1

# drop the fist row which is ones
y_score = np.delete(y_score, 0, axis=0)

# Computing ROC and ROC AUC
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(Y[:,i], y_score[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# plotting ROC
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','gray'])
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














