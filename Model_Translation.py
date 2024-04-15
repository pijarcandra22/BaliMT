from Transformer.transformer import Transformer # this is the transformer.py file
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pickle
from datetime import datetime
import pandas as pd

import numpy as np
import os
import torch

class TextDataset(Dataset):
    def __init__(self, source_data_sentences, result_data_sentences):
        self.source_data_sentences = source_data_sentences
        self.result_data_sentences = result_data_sentences

    def __len__(self):
        return len(self.source_data_sentences)

    def __getitem__(self, idx):
        return self.source_data_sentences[idx], self.result_data_sentences[idx]

class Transformer_Model(TextDataset):
  def __init__(self,source_data,result_data,source_vocabulary,result_vocabulary,max_sequence_length = 300):
    self.source_data              = source_data
    self.result_data              = result_data
    self.source_vocabulary        = source_vocabulary
    self.result_vocabulary        = result_vocabulary
    self.max_sequence_length      = max_sequence_length

    self.index_to_result          = {k:v for k,v in enumerate(result_vocabulary)}
    self.result_to_index          = {v:k for k,v in enumerate(result_vocabulary)}
    self.index_to_source          = {k:v for k,v in enumerate(source_vocabulary)}
    self.source_to_index          = {v:k for k,v in enumerate(source_vocabulary)}

    self.source_data_sentences    = source_data
    self.result_data_sentences    = result_data

    self.validate_sentences()

    self.transformer              = None
    self.device                   = None
    self.NEG_INFTY                = None
    self.START_TOKEN              = None
    self.END_TOKEN                = None
    self.PADDING_TOKEN            = None

    self.dataset                  = TextDataset(self.source_data_sentences, self.result_data_sentences)


  def is_valid_tokens(self,sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

  def is_valid_length(self,sentence, max_sequence_length):
      return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space

  def validate_sentences(self):
    valid_sentence_indicies  = []
    for index in range(len(self.result_data_sentences)):
      result_data_sentence, source_data_sentence = self.result_data_sentences[index], self.source_data_sentences[index]
      if self.is_valid_length(result_data_sentence, self.max_sequence_length) \
        and self.is_valid_length(source_data_sentence, self.max_sequence_length) \
        and self.is_valid_tokens(result_data_sentence, self.result_vocabulary):
          valid_sentence_indicies.append(index)

    self.result_data_sentences = [self.result_data_sentences[i] for i in valid_sentence_indicies]
    self.source_data_sentences = [self.source_data_sentences[i] for i in valid_sentence_indicies]

  def create_masks(self,eng_batch, kn_batch,NEG_INFTY):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([self.max_sequence_length, self.max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)

    for idx in range(num_sentences):
      source_sentence_length, result_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
      eng_chars_to_padding_mask = np.arange(source_sentence_length + 1, self.max_sequence_length)
      kn_chars_to_padding_mask = np.arange(result_sentence_length + 1, self.max_sequence_length)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

  def describe(self,START_TOKEN,END_TOKEN,PADDING_TOKEN,d_model = 512,batch_size = 30,ffn_hidden = 2048,num_heads = 8,drop_prob = 0.1,num_layers = 1,max_sequence_length = 200, lr=1e-4, NEG_INFTY = -1e9, total_loss = 0, num_epochs = 100):
    self.NEG_INFTY      = NEG_INFTY
    self.START_TOKEN    = START_TOKEN
    self.END_TOKEN      = END_TOKEN
    self.PADDING_TOKEN  = PADDING_TOKEN
    self.num_epochs     = num_epochs

    self.kn_vocab_size = len(self.result_vocabulary)

    self.transformer = Transformer(d_model,
                              ffn_hidden,
                              num_heads,
                              drop_prob,
                              num_layers,
                              max_sequence_length,
                              self.kn_vocab_size,
                              self.source_to_index,
                              self.result_to_index,
                              START_TOKEN,
                              END_TOKEN,
                              PADDING_TOKEN)

    self.train_loader    = DataLoader(self.dataset, batch_size)
    self.iterator        = iter(self.train_loader)

    self.criterian       = nn.CrossEntropyLoss(ignore_index=self.result_to_index[PADDING_TOKEN],
                                    reduction='none')

    # When computing the loss, we are ignoring cases when the label is the padding token
    for params in self.transformer.parameters():
        if params.dim() > 1:
            nn.init.xavier_uniform_(params)

    self.optim = torch.optim.Adam(self.transformer.parameters(), lr)
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    self.transformer.train()
    self.transformer.to(self.device)
  
  def fit(self,folder_cehckpoint=''):
    dataHist = {
      'epoch' : [],
      'loss'  : []
    }
    for epoch in range(self.num_epochs):
        print(f"Epoch {epoch}")

        meanloss = []

        self.iterator = iter(self.train_loader)
        for batch_num, batch in enumerate(self.iterator):
            self.transformer.train()
            eng_batch, kn_batch = batch
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_masks(eng_batch, kn_batch, self.NEG_INFTY)
            self.optim.zero_grad()
            kn_predictions = self.transformer(eng_batch,
                                        kn_batch,
                                        encoder_self_attention_mask.to(self.device),
                                        decoder_self_attention_mask.to(self.device),
                                        decoder_cross_attention_mask.to(self.device),
                                        enc_start_token=False,
                                        enc_end_token=False,
                                        dec_start_token=True,
                                        dec_end_token=True)
            labels = self.transformer.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
            loss = self.criterian(
                kn_predictions.view(-1, self.kn_vocab_size).to(self.device),
                labels.view(-1).to(self.device)
            ).to(self.device)
            valid_indicies = torch.where(labels.view(-1) == self.result_to_index[self.PADDING_TOKEN], False, True)
            loss = loss.sum() / valid_indicies.sum()
            loss.backward()
            self.optim.step()
            
            meanloss.append(loss.item())
            #train_losses.append(loss.item())
            if batch_num % 100 == 0:
                print(f"Iteration {batch_num} : {loss.item()}")
                print(f"Source Lang: {eng_batch[0]}")
                print(f"Result Translation: {kn_batch[0]}")
                result_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
                predicted_sentence = ""
                for idx in result_sentence_predicted:
                  if idx == self.result_to_index[self.END_TOKEN]:
                    break
                  predicted_sentence += self.index_to_result[idx.item()]
                print(f"Result Prediction: {predicted_sentence}")


                self.transformer.eval()
        if folder_cehckpoint != '':
            try:
                os.mkdir(folder_cehckpoint)
            except:
               pass

            dataHist['epoch'].append(epoch)
            dataHist['loss'].append(np.mean(meanloss))
            print("=======================")

            pd.DataFrame.from_dict(dataHist).to_csv(folder_cehckpoint+'/Loss_Record.csv',index=False)

            with open(folder_cehckpoint+'/Transformer.pickle', 'wb') as handle:
                pickle.dump(self.transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def translate(self,sentence):
    source_sentence = (sentence,)
    result_sentence = ("",)
    for word_counter in range(self.max_sequence_length):
      encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= self.create_masks(source_sentence, result_sentence, self.NEG_INFTY)
      predictions = self.transformer(source_sentence,
                                  result_sentence,
                                  encoder_self_attention_mask.to(self.device),
                                  decoder_self_attention_mask.to(self.device),
                                  decoder_cross_attention_mask.to(self.device),
                                  enc_start_token=False,
                                  enc_end_token=False,
                                  dec_start_token=True,
                                  dec_end_token=False)
      next_token_prob_distribution = predictions[0][word_counter]
      next_token_index = torch.argmax(next_token_prob_distribution).item()
      next_token = self.index_to_result[next_token_index]
      result_sentence = (result_sentence[0] + next_token, )
      if next_token == self.END_TOKEN:
        break
    return result_sentence[0]