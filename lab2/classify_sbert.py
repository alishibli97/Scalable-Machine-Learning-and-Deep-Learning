import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertConfig, BertTokenizer
from scipy import spatial

# QUESTION: 
## SHOULD WE TRAIN FROM SCRATCH OR CAN WE START FROM bert-base-uncased or any other starting point?
##### DONE: BertConfig() starts from random weights

## TODO: Add validation set and test on testing set for SBERT classification
##### keep the model running for a while and then test .....

## TODO: After finishing training for classification, save the model please to stop training everytime <3
##### sure.. but we can just run the model 

## TODO: After training on the classification data, check how to "fine-tune" on the STS regression data (maybe only the head without the bert itself?)
## Check my question please <3 https://docs.google.com/document/d/1YeohuAr55fKF2nI1RiCgpq_Wa3Yn-CLAwfPoBmViNIM/edit

class SBERT(nn.Module):
    def __init__(self):
        super(SBERT, self).__init__()
        
        # self.model = BertModel.from_pretrained("bert-base-uncased")
        configuration = BertConfig()
        self.model = BertModel(configuration)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.pooling = nn.AvgPool1d(kernel_size=3,stride=1)
        # 768 / 3 -> 256
        self.linear = nn.Linear(in_features=2298, out_features=3) # 2298=(768-2)*3; 153 is the embedding dimension after pooling and stuff..
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sent1, sent2=None, objective="embedding"):
        encoded_input1 = self.tokenizer(sent1, padding=True, truncation=True, return_tensors='pt')
        output1 = self.model(**encoded_input1)
        output1 = self.pooling(output1["pooler_output"])
        
        if objective=="embedding":
            return output1

        encoded_input2 = self.tokenizer(sent2, padding=True, truncation=True, return_tensors='pt')
        output2 = self.model(**encoded_input2)
        output2 = self.pooling(output2["pooler_output"])
                        
        if objective == "regression":
            return torch.cosine_similarity(output1, output2)

        if objective == "classification":
            diff = abs(torch.subtract(output1,output2))
            concat = torch.cat([output1,output2,diff],axis=1)            
            result = self.linear(concat)
            out = self.softmax(result)
            return out

sbert = SBERT()

#classify
import json
json_list_train = list(open("datasets/snli_1.0/snli_1.0_train.jsonl","r"))
json_list_val = list(open("datasets/snli_1.0/snli_1.0_dev.jsonl","r"))

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
import pandas as pd

data_train = {'sentence1': [], 'sentence2': [], 'gold_label': []}
for json_str in json_list_train:
    try:
        result = json.loads(json_str)
        result['gold_label']=label2int[result['gold_label']]
        for key in data_train:
            data_train[key].append(result[key])
    except:
        pass
df_train = pd.DataFrame.from_dict(data_train)#.head()

data_val = {'sentence1': [], 'sentence2': [], 'gold_label': []}
for json_str in json_list_val:
    try:
        result = json.loads(json_str)
        result['gold_label']=label2int[result['gold_label']]
        for key in data_val:
            data_val[key].append(result[key])
    except:
        pass
df_val = pd.DataFrame.from_dict(data_val)#.head()

print("Training data:",len(df_train))
print("Testing data:",len(df_val))

from torch.utils.data import Dataset,DataLoader

class SNLI_Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sent1 = row['sentence1']
        sent2 = row['sentence2']
        label = row['gold_label']
        return (sent1, sent2), label

import torch.optim as optim
import time

# from torchsample.callbacks import EarlyStopping
# callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
# model.set_callbacks(callbacks)


n_epochs = 10
n_batch = 16
lr=0.001


training_data = SNLI_Dataset(df_train)
train_dataloader = DataLoader(training_data, batch_size=n_batch, shuffle=False)

validation_data = SNLI_Dataset(df_val)
val_dataloader = DataLoader(validation_data, batch_size=n_batch, shuffle=True)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(sbert.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(sbert.parameters(), lr=lr)

sbert.train()
train_losses = []
val_losses = []
for epoch in range(n_epochs):

    for i,((sent1,sent2),label) in enumerate(train_dataloader):
        
        start = time.time()
        
        optimizer.zero_grad()
        
        output = sbert(sent1,sent2,objective="classification")
        loss = criterion(output, label)
        train_losses.append(loss)
        
#         (sent1_val,sent2_val),label_val = next(iter(val_dataloader))
#         output_val = sbert(sent1_val,sent2_val,"classification")
#         val_loss = criterion(output_val,label_val)
#         val_losses.append(val_loss)
        
        message = "epoch={}/{} iteration={}/{} train_loss={:.4f} took {:.4f} secs" \
            .format(epoch+1,n_epochs,i+1,len(train_dataloader),loss.detach().numpy(),time.time()-start)
        
        print(message)
        
        loss.backward()
        optimizer.step()
        
        
print('Finished Training')

PATH = "models/classification.pt"
torch.save(sbert.state_dict(), PATH)
PATH = "models/classification.pt"

sbert = SBERT()
sbert.load_state_dict(torch.load(PATH))
sbert.train()

import pandas as pd

df_train = pd.read_csv("Stsbenchmark/sts-train.csv",header=0,names=["main-caption","genre","filename","year","score","sentence1","sentence2"])#,usecols=['score','sentence1','sentence2'])
df_test = pd.read_csv("Stsbenchmark/sts-test.csv",header=0,names=["main-caption","genre","filename","year","score","sentence1","sentence2"])#,usecols=['score','sentence1','sentence2'])

df_train = df_train[['score','sentence1','sentence2']]
df_test = df_test[['score','sentence1','sentence2']]

df_train.head()
# df_test.head()

def map_score(value, leftMin=0, leftMax=5, rightMin=-1, rightMax=1):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

df_train['score'] = df_train['score'].apply(map_score)


from torch.utils.data import Dataset,DataLoader

class STS_Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sent1 = row['sentence1']
        sent2 = row['sentence2']
        label = row['score']
        return (sent1, sent2), label

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.SGD(sbert.parameters(), lr=0.001, momentum=0.9)

training_data_regression = STS_Dataset(df_train)
train_dataloader_regression = DataLoader(training_data_regression, batch_size=64, shuffle=False)

n_epochs = 1

losses = []
for epoch in range(n_epochs):

    for i,((sent1,sent2),label) in enumerate(train_dataloader_regression):
        
        output = sbert(sent1,sent2,objective="regression")
        optimizer.zero_grad()
        
        label = label.float()
        loss = criterion(output, label)
        
        losses.append(loss)
        
        print(f"epoch={epoch+1} iteration={i+1}/{len(train_dataloader_regression)} loss={loss.detach().numpy()}")
        
        loss.backward()
        optimizer.step()
        
        # break
print('Finished Training')

PATH = "models/classification_regression.pt"
torch.save(sbert.state_dict(), PATH)

PATH = "models/classification_regression.pt"
sbert = SBERT()
sbert.load_state_dict(torch.load(PATH))
sbert.eval()

import json
list_ = []
path = "datasets/News_Category_Dataset_v2.json"
with open(path) as files:
    for file in files:
        list_.append(json.loads(file))



df_news = pd.DataFrame(list_)
df_news.head()


def create_embeddings(sentences):
    embeddings = []
    for i,sentence in enumerate(sentences):
        print(f"Finished iteration {i}/{len(sentences)}",end="\r")
        
        embeddings.append(sbert(sentence,objective="embedding"))

    embeddings = torch.FloatTensor(embeddings)
    torch.save(x, 'datasets/embeddings.pt')