import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


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
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, features, captions):

        # concatenate features and embeds to pass through lstm network:
        inputs = (features.unsqueeze(1), self.embed(captions))
        inputs = torch.cat(inputs, dim=1)

        # remove last tensor (<end> token) from inputs:
        inputs = torch.split(inputs, [captions.shape[1], 1], dim=1)[0]

        # clean out hidden state:
        hidden = (torch.randn(self.num_layers, inputs.shape[0], self.hidden_size, device=inputs.device), \
                  torch.randn(self.num_layers, inputs.shape[0], self.hidden_size, device=inputs.device))

        # pass features and word embeds (in this order) through lstm network:
        outputs, hidden = self.lstm(inputs, hidden)

        # # pass outputs through linear fully connected network with dropout:
        outputs = self.dropout(self.fc(outputs))

        # return outputs
        return outputs


    def sample(self, inputs, states=None, max_len=20):

         output = list()

         for i in range(max_len):

              # get scores from input:
              scores, states = self.lstm(inputs, states)
              scores = self.fc(scores)

              # convert probabilities tensor to numpy array:
              scores = scores.cpu().detach().numpy()[0][0].tolist()

              # get the index of max probability and add to sentence list:
              word_idx = scores.index(max(scores))

              # exit loop if got the <end> word:
              if word_idx == 1:
                  break

              # append index to sentence words list:
              if word_idx != 0:
                  output.append(word_idx)

              # embed the current word id to feed the lstm network:
              inputs = self.embed(torch.LongTensor([word_idx]).unsqueeze(1).to(inputs.device))

         # return list of integers representing the sentence words:
         return output
