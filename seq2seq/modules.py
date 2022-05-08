import random
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Any, Optional


class Encoder(nn.Module):

    def __init__(self, 
                input_dim: int, 
                emb_dim: int, 
                hid_dim: int, 
                n_layers: int, 
                dropout: int, 
                bidirectional: bool) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, 
                           hid_dim, 
                           num_layers=n_layers, 
                           dropout=dropout, 
                           bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.h_linear = nn.Linear(2 * self.hid_dim, hid_dim)
        self.c_linear = nn.Linear(2 * self.hid_dim, hid_dim)


    def forward(self, 
                src: torch.Tensor, 
                src_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

                
        #src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src sent len, batch size, emb dim]
        packed = nn.utils.rnn.pack_padded_sequence(embedded, 
                                      src_len.to('cpu'))
        # Compute the RNN output values of the encoder RNN. 
        packed_outputs, (hidden, cell) = self.rnn(packed)
        # outputs, hidden and cell should be initialized here. 
        
        encoder_states, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 

        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        if self.bidirectional:
            hidden = self.h_linear(torch.cat((hidden[0:1], 
                                            hidden[1:2]), dim=2)
            )
            cell = self.c_linear(torch.cat((cell[0:1], 
                                          cell[1:2]), dim=2)
            )

        return encoder_states, (hidden, cell)

   
# you can paste code of decoder from modules.py

class Attention(nn.Module):
    def __init__(self, 
                enc_hid_dim: int, 
                dec_hid_dim: int) -> None:
        super().__init__()
       
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn = nn.Linear((self.enc_hid_dim * 2) + self.dec_hid_dim, 
                              self.dec_hid_dim)
        self.v = nn.Linear(self.dec_hid_dim, 1, bias=False)
        
    def forward(self, 
                encoder_states: torch.Tensor, 
                hidden: torch.Tensor) -> torch.Tensor:
        
        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # hidden = [1, batch size, dec_hid_dim]
        src_len, batch, _ = encoder_states.shape
        # repeat hidden and concatenate it with encoder_outputs
        hidden = hidden.repeat(src_len, 1, 1)
        # calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_states), dim=2))) 
        # get attention, use softmax function which is defined, can change temperature
        attention = self.v(energy)
        return F.softmax(attention, dim=0)
        

class DecoderWithAttention(nn.Module):
    def __init__(self, 
                output_dim: int, 
                emb_dim: int, 
                enc_hid_dim: int, 
                dec_hid_dim: int, 
                N_LAYERS: int,
                attention: torch.Tensor) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim) 
        # linear layer to get next word
        self.out =  nn.Linear(enc_hid_dim, output_dim)

        
    def forward(self, 
                input: torch.Tensor, 
                encoder_states: torch.Tensor, 
                hidden: torch.Tensor,
                cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]       
        #input = [1, batch size]
        
        input = input.unsqueeze(0) # because only one word, no words sequence 
        #embedded = [1, batch size, emb dim]
        embedded = self.embedding(input)
        attention = self.attention(encoder_states, hidden)
        attention = attention.permute(1, 2, 0)
        encoder_states = encoder_states.permute(1, 0, 2)
        # get weighted sum of encoder_outputs
        context = torch.bmm(attention, encoder_states)
        context = context.permute(1, 0, 2)
        # concatenate weighted sum and embedded
        rnn_input = torch.cat((embedded, context), dim=2)
        # get predictions
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        prediction = self.out(output)
        prediction = prediction.squeeze(0)
        return prediction, (hidden, cell)
		
		
class Seq2Seq(nn.Module):
    def __init__(self, 
                encoder: nn.Module, 
                decoder: nn.Module, 
                device: str,
                src_pad: int) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.src_pad = src_pad
        assert encoder.hid_dim == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, 
                src: torch.Tensor, 
                src_len: torch.Tensor, 
                trg: torch.Tensor, 
                teacher_forcing_ratio = 0.5) -> torch.Tensor:
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, (hidden, cell) = self.encoder(src, src_len)
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):

            output, (hidden, cell) = self.decoder(input,
                                                  enc_states, 
                                                  hidden,
                                                  cell)

            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(1)
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
  