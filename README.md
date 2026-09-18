# Shimla Crop Predictor

## Project Overview

Shimla Crop Predictor is a machine learning based agricultural advisory system designed for the Shimla region.

The system combines soil information, historical weather data, crop recommendation, weather prediction, and frost risk analysis to provide useful agricultural recommendations.

## Main Features

- Crop recommendation using soil and weather parameters
- Weather prediction
- Frost risk detection
- Frost advisory
- Historical weather visualization
- FastAPI REST API
- Streamlit dashboard
- Automated testing
- Code coverage analysis
- Pylint code quality analysis
- Radon complexity analysis
- GitHub Actions continuous integration

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Streamlit
- Pytest
- Pylint
- Radon
- Git
- GitHub
- GitHub Actions

## Project Structure

```text
shimla-crop-predictor
│
├── .github
│   └── workflows
│
├── data
│   ├── raw
│   └── processed
│
├── models
├── notebooks
├── src
├── tests
├── metrics
│
├── .gitignore
├── project_code.zip
├── requirements.txt
└── README.md
## Docker Deployment

The Shimla Crop Predictor application is containerized using Docker and the image is available on Docker Hub.

### Docker Hub Image

`23mis0557/shimla-crop-predictor:latest`

### Pull the Docker Image

```bash
docker pull 23mis0557/shimla-crop-predictor:latest