"""
This module contains the FastAPI application for sentiment analysis. 
It uses the Hugging Face Transformers library to perform sentiment analysis on the input text.
The application is designed to be run as a standalone service and can be deployed using Docker.
"""
from pathlib import Path
import uuid

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

BASE_PATH = Path().cwd()


# Sentiment Analysis Module
def get_device():
    """Select the appropriate device for inference."""
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


def get_model_dir():
    """Get the model directory."""
    return (
        BASE_PATH
        / "models"
        / "distilbert-base-uncased-finetuned-sst-2-english"
    )


def initialize_classifier():
    """Initialize the sentiment analysis pipeline."""
    return pipeline(
        "sentiment-analysis",
        model=get_model_dir(),
        device=get_device(),
    )


# Initialize the classifier globally
classifier = initialize_classifier()


def classify_text(text: str):
    """
    Perform sentiment analysis on the given text.
    Args:
        text (str): Input text to analyze.

    Returns:
        list: Sentiment analysis result.
    """
    return classifier(text)


# FastAPI App
app = FastAPI()


# Input Schema
class SentimentRequest(BaseModel):
    """
    Schema for the input data to the sentiment analysis API.
    """
    sentence: str
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


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

# Run the App
if __name__ == "__main__":
    # SentimentRequest(sentence="This is a test sentence")
    # print(BASE_PATH)
    pass
