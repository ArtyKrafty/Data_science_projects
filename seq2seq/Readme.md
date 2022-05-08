<h1 align='center'>Seq2Seq. Neural Machine Translation</h1>
<p align="center"><img src="https://drive.google.com/uc?id=1j6gMHuo0gR4fMKJu3vFnJbWgbh6G7MWT"  border="0"></a></p>



##  Attention

Attention layer can take in the previous hidden state of the decoder , and all of the stacked forward and backward hidden states $H$ from the encoder. The layer will output an attention vector , that is the length of the source sentence, each element is between 0 and 1 and the entire vector sums to 1.

Intuitively, this layer takes what we have decoded so far , and all of what we have encoded $H$, to produce a vector $a_t$, that represents which words in the source sentence we should pay the most attention to in order to correctly predict the next word to decode $\hat{y}_{t+1}$. The decoder input word that has been embedded  $y_t$.

You can use any type of the attention scores between previous hidden state of the encoder $s_{t-1}$ and hidden state of the decoder $h \in H$, you prefer. We have met at least three of them:<br><br>
