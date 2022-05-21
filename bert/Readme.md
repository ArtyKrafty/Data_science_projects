<h1 align='center'>Sentence Sentiment Classification. BERT</h1>
<p align="center"><img src="https://drive.google.com/uc?id=1yPEFyJ6POckTW7XgLgr1YBb7MCWL1ytd"  border="0" width="800"></a></p>



<b> **Baseline** подготовлен [Deep learning school](https://www.dlschool.org/pro-track) // [семинар](https://drive.google.com/file/d/1w_rTEWXQ_SA4YPXFjpkM0aU51bDgWLyI/view?usp=sharing).



_____

## Motivation

Our goal is to create a model that takes a sentence (just like the ones in our dataset) and produces either 1 (indicating the sentence carries a positive sentiment) or a 0 (indicating the sentence carries a negative sentiment). We can think of it as looking like this:


____

<font size="2">

Наша цель — создать модель, которая берет предложение (точно такое же, как в нашем наборе данных) и выдает либо 1 (указывает на то, что предложение несет в себе положительное настроение), либо 0 (указывает на то, что предложение несет в себе отрицательное настроение). Мы можем думать об этом так:</font>

<img src="https://jalammar.github.io/images/distilBERT/sentiment-classifier-1.png" />

Under the hood, the model is actually made up of two model.

* DistilBERT processes the sentence and passes along some information it extracted from it on to the next model. DistilBERT is a smaller version of BERT developed and open sourced by the team at HuggingFace. It’s a lighter and faster version of BERT that roughly matches its performance.
* The next model, a basic Logistic Regression model from scikit learn will take in the result of DistilBERT’s processing, and classify the sentence as either positive or negative (1 or 0, respectively).

The data we pass between the two models is a vector of size 768. We can think of this of vector as an embedding for the sentence that we can use for classification.
___
<font size="2">Под капотом модель фактически состоит из двух моделей.
* DistilBERT обрабатывает предложение и передает некоторую информацию, извлеченную из него, следующей модели. DistilBERT — это уменьшенная версия BERT, разработанная командой HuggingFace с открытым исходным кодом. Это более легкая и быстрая версия BERT, которая примерно соответствует его производительности.
* Следующая модель, базовая модель логистической регрессии от scikit Learn, будет принимать результат обработки DistilBERT и классифицировать предложение как положительное или отрицательное (1 или 0 соответственно).
Данные, которые мы передаем между двумя моделями, представляют собой вектор размером 768. Мы можем рассматривать этот вектор как вложение предложения, которое мы можем использовать для классификации.</font>

<img src="https://jalammar.github.io/images/distilBERT/distilbert-bert-sentiment-classifier.png" />

## Dataset
The dataset we will use in this example is [SST2](https://nlp.stanford.edu/sentiment/index.html), which contains sentences from movie reviews, each labeled as either positive (has the value 1) or negative (has the value 0):

___

<font size="2">В этом примере мы будем использовать набор данных [SST2](https://nlp.stanford.edu/sentiment/index.html), который содержит предложения из обзоров фильмов, каждое из которых помечено как положительное (имеет значение 1) или отрицательное. (имеет значение 0):</font>

<table class="features-table">
  <tr>
    <th class="mdc-text-light-green-600">
    sentence
    </th>
    <th class="mdc-text-purple-600">
    label
    </th>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      a stirring , funny and finally transporting re imagining of beauty and the beast and 1930s horror films
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      apparently reassembled from the cutting room floor of any given daytime soap
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      they presume their audience won't sit still for a sociology lesson
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      this is a visually stunning rumination on love , memory , history and the war between art and commerce
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      jonathan parker 's bartleby should have been the be all end all of the modern office anomie films
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
</table>







