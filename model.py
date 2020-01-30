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
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes image feature vectors and embedded word vectors (of size embed_size) as inputs 
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=0.5)
        #self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=0.5, batch_first=True)

        # the linear layer that maps the hidden state 
        # to the number of words we want as output, vocab_size
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # initialize the hidden state (see code below)
        self.hidden = self.init_hidden()
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.
            dimensions:
                features: batch_size x embed_size
                captions: batch_size x the length of the sequence (randomly chosen by data_loader)'''
        
        # drop the last word of each caption (i.e. the <END> word)
        #  since it not needed in the input sequence
        #   dimensions of captions are reduced to [batch_size, captions.shape[1]-1]
        captions = captions[:, :-1]
        
        # create embedded word vectors for each word in the captions
        #   dimensions of embeds are [batch_size, captions.shape[1]-1, embed_size]
        embeds = self.word_embeddings(captions)
        
        # Concat features and the embedded words into full sequences
        #   dimensions of lstm_input are [batch_size, captions.shape[1], embed_size]
        lstm_input = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)
                
        # run the sequences through the LSTM
        #   dimensions of lstm_output are [batch_size, captions.shape[1], hidden_size]
        lstm_out, self.hidden = self.lstm(lstm_input)
        
        # Pass features through a droupout layer
        lstm_out = self.dropout(lstm_out)
        
        # run the LSTM output through a fully connected layer
        #   dimensions of outputs are [batch_size, captions.shape[1], vocab_size]
        outputs = self.fc(lstm_out)
        #outputs = F.log_softmax(outputs, dim=1)
        
        return outputs
        
    def sample(self, features, states=None, max_len=20):
        ''' accepts pre-processed image tensor (features)
            and returns predicted sentence (list of tensor ids of length max_len)'''

        output = []

        for i in range(max_len):
            # run the embbeded vector (lstm_input) through the LSTM
            #   lstm_output has size [batch_size
            lstm_out, states = self.lstm(features, states)
            #lstm_out.squeeze(1)

            # run the LSTM output through the fully connected layer
            scores = self.fc(lstm_out)
            id_ = scores.max(2)[1]
            
            # add tensor id to the list
            output.append(id_.item())

            # update input
            features = self.word_embeddings(id_)

        return output
   
    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
            there will be none because the hidden state is formed based on previously seen data.
            So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_size)
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
