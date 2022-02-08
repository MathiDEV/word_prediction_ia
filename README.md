# Word prediction IA
This model was made with the help of [Just1Truc](https://github.com/Just1truc)
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

## Pretrained models
If you don't have enough calculation process, we have a pretrained model for you in models folder.

It was based on Macron's speechs thanks to the work of [Izimio](https://github.com/izimio)

If you have a great trained model don't hesitate to PR !

## Functional Scheme
