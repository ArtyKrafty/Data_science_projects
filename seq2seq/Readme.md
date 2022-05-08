<h1 align='center'>Seq2Seq. Neural Machine Translation</h1>
<p align="center"><img src="https://drive.google.com/uc?id=1j6gMHuo0gR4fMKJu3vFnJbWgbh6G7MWT"  border="0"></a></p>



##  Attention

Attention layer can take in the previous hidden state of the decoder $s_{t-1}$, and all of the stacked forward and backward hidden states $H$ from the encoder. The layer will output an attention vector $a_t$, that is the length of the source sentence, each element is between 0 and 1 and the entire vector sums to 1.

Intuitively, this layer takes what we have decoded so far $s_{t-1}$, and all of what we have encoded $H$, to produce a vector $a_t$, that represents which words in the source sentence we should pay the most attention to in order to correctly predict the next word to decode $\hat{y}_{t+1}$. The decoder input word that has been embedded  $y_t$.

You can use any type of the attention scores between previous hidden state of the encoder $s_{t-1}$ and hidden state of the decoder $h \in H$, you prefer. We have met at least three of them:<br><br>

---
<font size="2"> `Attention` может принимать предыдущее скрытое состояние декодера $s_{t-1}$ и все конкатенированные `forward` и `backward` скрытые состояния $H$ от кодировщика. Слой выдает вектор `attention` $a_t$, то есть длину исходного предложения, каждый элемент которого находится в диапазоне от 0 до 1, а сумма всего вектора равна 1.
Интуитивно этот слой берет то, что мы уже раскодировали $s_{t-1}$, и все то, что мы закодировали $H$, чтобы создать вектор $a_t$, представляющий, какие слова в исходном предложении мы должны обратить наибольшее внимание, чтобы правильно предсказать следующее слово для декодирования $\hat{y}_{t+1}$. Входное слово декодера, которое было встроено в $y_t$.
Мы можем использовать любой тип оценки внимания между предыдущим скрытым состоянием кодировщика $s_{t-1}$ и скрытым состоянием декодера $h \in H$. Мы встречали как минимум три из них: </font>


$$\operatorname{score}\left(\boldsymbol{h}, \boldsymbol{s}_{t-1}\right)=\left\{\begin{array}{ll}
\boldsymbol{h}^{\top} \boldsymbol{s}_{t-1} & \text { dot } \\
\boldsymbol{h}^{\top} \boldsymbol{W}_{\boldsymbol{a}} \boldsymbol{s}_{t-1} & \text { general } \\
\boldsymbol{v}_{a}^{\top} \tanh \left(\boldsymbol{W}_{\boldsymbol{a}}\left[\boldsymbol{h} ; \boldsymbol{s}_{t-1}\right]\right) & \text { concat }
\end{array}\right.$$
--------