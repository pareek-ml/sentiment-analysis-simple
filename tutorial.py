from pathlib import Path
import uuid
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline
import uvicorn

"""
.. _sentiment_analysis_tutorial:

Sentiment Analysis with FastAPI and Hugging Face Transformers
=============================================================

This tutorial demonstrates how to create a sentiment analysis service using FastAPI and Hugging Face Transformers.
We will use the `distilbert-base-uncased-finetuned-sst-2-english` model for sentiment analysis.

Requirements
------------
- `torch`
- `transformers`
- `fastapi`
- `pydantic`
- `uvicorn`

You can install the required packages using pip:

.. code-block:: bash

    pip install torch transformers fastapi pydantic uvicorn

"""

# %%
# Import necessary libraries
# ---------------------------


# %%
# Define the base path for the model
# ----------------------------------
BASE_PATH = Path().cwd()

# %%
# Function to select the appropriate device for inference
# -------------------------------------------------------
def get_device():
    """Select the appropriate device for inference."""
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

# %%
# Function to get the model directory
# -----------------------------------
def get_model_dir():
    """Get the model directory."""
    return (
        BASE_PATH
        / "models"
        / "distilbert-base-uncased-finetuned-sst-2-english"
    )

# %%
# Function to initialize the sentiment analysis pipeline
# ------------------------------------------------------
def initialize_classifier():
    """Initialize the sentiment analysis pipeline."""
    return pipeline(
        "sentiment-analysis",
        model=get_model_dir(),
        device=get_device(),
    )

# %%
# Initialize the classifier globally
# ----------------------------------
classifier = initialize_classifier()

# %%
# Function to perform sentiment analysis on the given text
# --------------------------------------------------------
def classify_text(text: str):
    """
    Perform sentiment analysis on the given text.
    Args:
        text (str): Input text to analyze.

    Returns:
        list: Sentiment analysis result.
    """
    return classifier(text)

# %%
# Create the FastAPI app
# ----------------------
app = FastAPI()

# %%
# Define the input schema for the sentiment analysis API
# ------------------------------------------------------
class SentimentRequest(BaseModel):
    """
    Schema for the input data to the sentiment analysis API.
    """
    sentence: str
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# %%
# Define the endpoint for sentiment prediction
# --------------------------------------------
@app.post("/predict")
async def predict_sentiment(data: SentimentRequest):
    """
    Predict sentiment for the input text.
    Args:
        data (SentimentRequest): Input data containing the sentence.

    Returns:
        dict: Sentiment analysis result.
    """
    sentiment = classify_text(data.sentence)
    return {"request_id": data.request_id, "sentiment": sentiment}

# %%
# Run the FastAPI app
# -------------------
if __name__ == "__main__":
    pass
# %%
# Run the FastAPI app
# docker build -t sentiment-api .
# docker run -d -p 8000:8000 sentiment-api