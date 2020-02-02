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

        # the embedding layer turns words into a vector of size embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes image feature vectors and embedded word vectors 
        # (size embed_size) as inputs and outputs hidden states (size hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                            dropout=0.5 if num_layers > 1 else 0,
                            batch_first=True)

        # the linear layer maps the hidden state (size hidden_size)
        # to the number of words we want as output (size vocab_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # dropout layer
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.
            dimensions:
                features: batch_size x embed_size
                captions: batch_size x the length of the sequence 
                                    (randomly chosen by data_loader)'''

        # drop the last word of each caption (i.e. the <END> word)
        # since it not needed in the input sequence
        #   dimensions of captions are reduced to [batch_size, captions.shape[1]-1]
        captions = captions[:, :-1]

        # create embedded word vectors for each word in the captions
        #   dimensions of embeds are [batch_size, captions.shape[1]-1, embed_size]
        embeds = self.embed(captions)

        # Concat features and the embedded words into full sequences
        #   dimensions of lstm_input are [batch_size, captions.shape[1], embed_size]
        features = features.unsqueeze(dim=1)
        lstm_input = torch.cat((features, embeds), dim=1)

        # run the sequences through the LSTM
        #   dimensions of lstm_output are [batch_size, captions.shape[1], hidden_size]
        lstm_out, _ = self.lstm(lstm_input, None)

        # Pass features through a droupout layer
        lstm_out = self.dropout(lstm_out)

        # run the LSTM output through a fully connected layer
        #   dimensions of outputs are [batch_size, captions.shape[1], vocab_size]
        outputs = self.fc(lstm_out)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        ''' accepts pre-processed image tensor (features)
            and returns predicted sentence (list of tensor ids of length max_len)'''

        sentence = []
        print(inputs.size())

        for _ in range(max_len):
            # run the embbeded vector (lstm_input) through the LSTM
            lstm_out, states = self.lstm(inputs, states)

            # run the LSTM output through the fully connected layer
            scores = self.fc(lstm_out)
            
            # find tensor id with highest score
            prediction = scores.max(2)[1]
            predicted_index = prediction.item()
            
            # add tensor id to the list
            sentence.append(predicted_index)
       
            # update input
            inputs = self.embed(prediction)


        return sentence
