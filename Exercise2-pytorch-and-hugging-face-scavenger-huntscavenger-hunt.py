# PyTorch tensors
# Scan through the PyTorch tensors documentation here. Be sure to look at the examples.

# In the following cell, create a tensor named my_tensor of size 3x3 with values of your choice. The tensor should be created on the GPU if available. Print the tensor.

# Fill in the missing parts labelled <MASK> with the appropriate code to complete the exercise.
​
# Hint: Use torch.cuda.is_available() to check if GPU is available
​
import torch
​
# Set the device to be used for the tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
​
# Create a tensor on the appropriate device
my_tensor = torch.tensor = ([[1, 2, 3], [4, 5, 6], [7,8,9]])
​
# Print the tensor
print(my_tensor)
​
print(torch.cuda.is_available())
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
False
# Check the previous cell
​
assert my_tensor.device.type in {"cuda", "cpu"}
assert my_tensor.shape == (3, 3)
​
print("Success!")


# Neural Net Constructor Kit torch.nn
# You can think of the torch.nn (documentation) module as a constructor kit for neural networks. It provides the building blocks for creating neural networks, including layers, activation functions, loss functions, and more.

# Instructions:

# Create a three layer Multi-Layer Perceptron (MLP) neural network with the following specifications:

# Input layer: 784 neurons
# Hidden layer: 128 neurons
# Output layer: 10 neurons
# Use the ReLU activation function for the hidden layer and the softmax activation function for the output layer. Print the neural network.

# Hint: MLP's use "fully-connected" or "dense" layers. In PyTorch's nn module, this type of layer has a different name. See the examples in this tutorial to find out more.


# Replace <MASK> with the appropriate code to complete the exercise.
​
import torch.nn as nn
​
​
class MyMLP(nn.Module):
    """My Multilayer Perceptron (MLP)
​
    Specifications:
​
        - Input layer: 784 neurons
        - Hidden layer: 128 neurons with ReLU activation
        - Output layer: 10 neurons with softmax activation
​
    """
​
    def __init__(self):
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
​
    def forward(self, x):
        # Pass the input to the second layer
        x = self.fc1(x)
​
        # Apply ReLU activation
        x = self.relu(x)
​
        # Pass the result to the final layer
        x = self.fc2(x)
​
        # Apply softmax activation
        x = self.softmax(x)
        
        return x
​
​
my_mlp = MyMLP()
print(my_mlp)
MyMLP(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (relu): ReLU()
  (softmax): Softmax(dim=1)
)
# Check your work here:
​
​
# Check the number of inputs
assert my_mlp.fc1.in_features == 784
​
# Check the number of outputs
assert my_mlp.fc2.out_features == 10
​
# Check the number of nodes in the hidden layer
assert my_mlp.fc1.out_features == 128
​
# Check that my_mlp.fc1 is a fully connected layer
assert isinstance(my_mlp.fc1, nn.Linear)
​
# Check that my_mlp.fc2 is a fully connected layer
assert isinstance(my_mlp.fc2, nn.Linear)


"""
PyTorch Training Loops
PyTorch makes writing a training loop easy!

Instructions:

Fill in the blanks!
y
# Replace <MASK> with the appropriate code to complete the exercise.
​
"""

def fake_training_loaders():
    for _ in range(30):
        yield torch.randn(64, 784), torch.randint(0, 10, (64,))
​
​
for epoch in range(3):
    # Create a training loop
    for i, data in enumerate(fake_training_loaders()):
        # Every data instance is an input + label pair
        x, y = data
​
        # Zero your gradients for every batch!
        optimizer.zero_grad()
​
        # Forward pass (predictions)
        y_pred = model(x)
​
        # Compute the loss and its gradients
        loss = loss_fn(y_pred, y)
        loss.backward()
​
        # Adjust learning weights
        optimizer.step()
​
        if i % 10 == 0:
            print(f"Epoch {epoch}, batch {i}: {loss.item():.5f}")


assert abs(loss.item() - 2.3) < 0.1, "the loss should be around 2.3 with random data"
# Great job! Now you know the basics of PyTorch! Let's turn to HuggingFace 🤗.

"""
Get to know HuggingFace
HuggingFace is a popular destination for pre-trained models and datasets that can be applied to a variety of tasks quickly and easily. In this section, we will explore the capabilities of HuggingFace and learn how to use it to build and train neural networks.

Download a model from HuggingFace and use it for sentiment analysis
HuggingFace provides a number of pre-trained models that can be used for a variety of tasks. In this exercise, we will use the distilbert-base-uncased-finetuned-sst-2-english model to perform sentiment analysis on a movie review.

Instructions:

Review the AutoModel tutorial on the HuggingFace website.
Instantiate an AutoModelForSequenceClassification model using the distilbert-base-uncased-finetuned-sst-2-english model.
Instantiate an AutoTokenizer using the distilbert-base-uncased-finetuned-sst-2-english model.
Define a function that will get a prediction
# Replace <MASK> with the appropriate code to complete the exercise.
"""
​
# Get the model and tokenizer
​
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
​
pt_model =  AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english") 
​
​
def get_prediction(review):
    """Given a review, return the predicted sentiment"""
​
     # Ensure the review is a string
    if not isinstance(review, str):
        raise ValueError("The review must be a string.")
​
    # Tokenize the review
    # (Get the response as tensors and not as a list)
    inputs = tokenizer(review, return_tensors='pt',padding=True, truncation=True)
​
    # Perform the prediction (get the logits)
    outputs = pt_model(**inputs)
​
    # Get the predicted class (corresponding to the highest logit)
    predictions = torch.argmax(outputs.logits, dim=-1)
​
    return "positive" if predictions.item() == 1 else "negative"
# Check

review = "This movie is not so great :("

print(f"Review: {review}")
print(f"Sentiment: {get_prediction(review)}")

assert get_prediction(review) == "negative", "The prediction should be negative"


review = "This movie rocks!"

print(f"Review: {review}")
print(f"Sentiment: {get_prediction(review)}")

assert get_prediction(review) == "positive", "The prediction should be positive"
# Check
​
review = "This movie is not so great :("
​
print(f"Review: {review}")
print(f"Sentiment: {get_prediction(review)}")
​
assert get_prediction(review) == "negative", "The prediction should be negative"
​
​
review = "This movie rocks!"
​
print(f"Review: {review}")
print(f"Sentiment: {get_prediction(review)}")
​
assert get_prediction(review) == "positive", "The prediction should be positive"

"""
Download a dataset from HuggingFace
HuggingFace provides a number of datasets that can be used for a variety of tasks. In this exercise, we will use the imdb dataset and pass it to the model we instantiated in the previous exercise.

Instructions:

Review the loading a dataset documentation
Fill in the blanks
"""

# Replace <MASK> with the appropriate code
​
from datasets import load_dataset
​
# Load the test split of the imdb dataset
dataset = load_dataset("imdb", split="test")
​
dataset

# Check

from pprint import pprint

from datasets import Dataset

assert isinstance(dataset, Dataset), "The dataset should be a Dataset object"
assert set(dataset.features.keys()) == {
    "label",
    "text",
}, "The dataset should have a label and a text feature"

# Show the first example
pprint(dataset[0])

"""
Now let's use the pre-trained model!
Let's make some predictions.

Instructions:

Fill in the blanks
# Replace <MASK> with the appropriate code
"""

# Get the last 3 reviews
reviews = dataset["text"][-3:]

# Get the last 3 labels
labels = dataset["label"][-3:]

# Check
for review, label in zip(reviews, labels):
    # Let's use your get_prediction function to get the sentiment
    # of the review!
    prediction = get_prediction(review)

    print(f"Review: {review[:80]} \n... {review[-80:]}")
    print(f'Label: {"positive" if label else "negative"}')
    print(f"Prediction: {prediction}\n")
# Replace <MASK> with the appropriate code
​
# Get the last 3 reviews
reviews = dataset["text"][-3:]
​
# Get the last 3 labels
labels = dataset["label"][-3:]
​
# Check
for review, label in zip(reviews, labels):
    # Let's use your get_prediction function to get the sentiment
    # of the review!
    prediction = get_prediction(review)
​
    print(f"Review: {review[:80]} \n... {review[-80:]}")
    print(f'Label: {"positive" if label else "negative"}')
    print(f"Prediction: {prediction}\n")
