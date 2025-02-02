# Sentiment Analysis API

This repository contains a FastAPI application for sentiment analysis using the Hugging Face Transformers library. The application is designed to be run as a standalone service and can be deployed using Docker.

## Prerequisites

- Docker installed on your machine
- Basic knowledge of Python and Docker

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/sentiment-analysis-simple.git
cd sentiment-analysis-simple
```
Note: Add model in the model directory

### 2. Build the Docker Image

Build the Docker image using the following command:

```sh
docker build -t sentiment-analysis-simple .
```

### 3. Run the Docker Container

Run the Docker container using the following command:

```sh
docker run -d -p 8000:8000 sentiment-analysis-simple
```

### 4. Access the API

The API will be accessible at `http://localhost:8000`. You can use tools like `curl` or Postman to interact with the API.

### 5. Example Request

To predict the sentiment of a sentence, send a POST request to the `/predict` endpoint with a JSON payload containing the sentence.

```sh
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"sentence": "I love this product!"}'
```

### 6. Response

The API will return a JSON response with the sentiment analysis result.

```json
{
    "request_id": "some-unique-id",
    "sentiment": [
        {
            "label": "POSITIVE",
            "score": 0.9998
        }
    ]
}
```

## Project Structure

- `app.py`: Contains the FastAPI application and sentiment analysis logic.
- `Dockerfile`: Dockerfile to build the Docker image.
- `models/`: Directory containing the pre-trained model weights.
- `requirements.txt`: List of Python dependencies.

## Conclusion

You now have a fully functional sentiment analysis API running in a Docker container. You can extend this application by adding more features or improving the model.

For any questions or issues, please open an issue in the repository.
