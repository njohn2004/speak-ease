# Speak Ease

Real-time web application for accurate speech recognition, tailored for patients suffering from dysarthria (speech impediment)
 

# Problem Statement
Dysarthria traps thoughts in a body struggling to speak, isolating individuals from society and straining relationships. Every conversation becomes a battle, eroding confidence and independence.

Dysarthria severely impacts communication, affecting quality of life and independence.
Affects millions globally, including stroke victims(41-53%), Parkinson's patients(44-88%), and those suffering from other neurological conditions
Growing issue due to aging population and increasing neurological disorders

This solution can significantly improve communication and quality of life for a large, under-served population, making it a crucial healthcare innovation.

## High level Solution Overview
We will start with the state of the art end to end speech Recognition model with high accuracy. This high quality ASR model will be trained on hundreds of hours of typical or standard speech with no impairements. After we achieve high accuracy for the end to end model, then we will start fine-tuning parts of the model to an individual with speech impairement.<br>
So our main aproach is training a base model on a large dataset of normal speech and then training a personalised model using a much smaller slurred speech dataset. We can use tranfer learning for fine tuning parts of our base model.
# Model Architecture
![Screenshot from 2021-10-01 20-22-07](https://user-images.githubusercontent.com/42781233/135641230-4775970a-479f-4d40-9707-6c50c9b0bb5b.png)

# Base Model Performance
The base ASR model was trained on 100 hours of Librispeech Dataset.
- Final Epoch Average Loss: 0.46
- Final Epoch Average CER: 0.10
- Final Epoch Average WER: 0.11

# Dataset preparation
After we train our ASR model on hundreds of hours of typical speech, we are good to go for fine-tuning our model on impaired speech. We need to collect impaired speech dataset. We build web app using django framework to do the same.

## Link to web APP:
1. http://speech-collection.herokuapp.com/index/
2. https://mmig.github.io/speech-to-flac/
