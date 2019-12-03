import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        #Assigning hidden dimension
        self.hidden_dim = hidden_size
        
        #getting embed from nn.Embedding()
        self.embed = nn.Embedding(vocab_size, embed_size)
        #Creating LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #Initializing Lineear linear to apply at last of RNN layer for further prediction
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        #Initializing valuews for hidden and cell state
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size)) 
        
    
    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), 1)
        
        #Getting output i.e score and hidden layer 
        lstm_out, self.hidden = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        
        
        return outputs
        

    def sample(self, inputs, states=None, hidden= None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        for i in range(max_len):
            outputs, hidden = self.lstm(inputs, hidden)
#             
            outputs = self.linear(outputs.squeeze(1))
#             
            target_index = outputs.max(1)[1]
#            
            res.append(target_index.item())
            inputs = self.embed(target_index).unsqueeze(1)
#             
        return res
        