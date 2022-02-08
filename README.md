# Word prediction IA
This model was made with the help of [Just1Truc](https://github.com/Just1truc) and [Izimio](https://github.com/izimio)
This script perfectly work on both Nvidia Cuda and casual CPU's according to your configuration. 

## Train
You can train the AI with
```
python3 create_model.py dataset.txt model_name
```
Dataset is a set of sentences in a text file with no line breaks.

A dataset should be at least 1 Mo of text.

Our NLP process on French language, to change it change the SpaCy model at the line 25 : 
```python
15. nlp = spacy.blank("fr")
```

## Generation
With the model previously made you can generate new text based on your dataset.
```
python3 generate.py model_name.pt "some first words" <new words count>
```
Your first words should be in dataset and written in lowercase.

## Functional Scheme
![Functional Scheme](https://raw.githubusercontent.com/MathiDEV/word_prediction_ia/main/IA%20PoC.png)
