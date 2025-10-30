# HMM-Based Activity Recognition System

This project implements a Hidden Markov Model (HMM) based activity recognition system using accelerometer and gyroscope data from wearable sensors. The system classifies human activities such as walking, standing, jumping, and still using time-domain and frequency-domain features extracted from sensor data.

## Project Overview

The notebook demonstrates a complete machine learning pipeline for activity recognition:

1. **Data Loading and Exploration**: Loading sensor data from multiple participants and activities
2. **Data Preprocessing**: Harmonizing sampling rates and merging accelerometer/gyroscope data
3. **Feature Extraction**: Computing comprehensive time-frequency features from sensor signals
4. **Model Training**: Training separate Gaussian HMMs for each activity class
5. **Evaluation**: Assessing model performance using sequence-level and window-level classification

## Dataset

The dataset consists of accelerometer and gyroscope readings from two participants performing four activities:
- Walking
- Standing
- Jumping
- Still

Each activity has multiple recordings captured using different mobile devices. The raw data includes:
- Accelerometer data (x, y, z axes)
- Gyroscope data (x, y, z axes)
- Timestamp information

## Methodology

### Data Preprocessing
- **Merging**: Combined accelerometer and gyroscope data using time-based alignment
- **Resampling**: Harmonized all recordings to a uniform 100 Hz sampling rate
- **Normalization**: Z-score normalization to remove device and participant biases

### Feature Extraction
Features are extracted from 3-second overlapping windows (75% overlap) and include:

**Time-Domain Features:**
- Mean, standard deviation, energy
- Zero-crossing rate
- Signal magnitude area (SMA)
- Cross-correlations between axes

**Frequency-Domain Features:**
- Dominant frequency
- Spectral entropy
- Band power in low (0.5-3 Hz), mid (3-8 Hz), and high (>8 Hz) frequency ranges

**Combined Features:**
- Total energy ratios
- Energy derivatives
- Magnitude-based features

### Model Architecture
- **Separate HMMs**: One Gaussian HMM trained per activity class
- **Hidden States**: 6 states for dynamic activities (walking, jumping), 3 states for static activities (standing, still)
- **Covariance Types**: Full covariance for dynamic activities, diagonal for static ones
- **Training**: Baum-Welch algorithm (EM optimization) with 600 iterations

### Evaluation Metrics
- **Sequence-level accuracy**: Classifying entire recording sequences
- **Window-level accuracy**: Per-window classification with majority voting and median smoothing
- **Per-class metrics**: Sensitivity and specificity for each activity
- **Confusion matrix**: Detailed classification performance visualization

## Results

The system achieves robust activity recognition performance with:
- Comprehensive feature engineering capturing both temporal and spectral characteristics
- Dimensionality reduction using PCA (95% variance retention)
- Sequence modeling with HMMs to capture temporal dependencies
- Post-processing techniques (median smoothing) for improved classification stability

## Files Description

- `HMM_group1.ipynb`: Main Jupyter notebook containing the complete implementation
- `Complete_Raw_Dataset.csv`: Merged raw sensor data from all participants and activities
- `Harmonized_Dataset.csv`: Preprocessed data with uniform sampling rates
- `Extracted_Features.csv`: Feature vectors extracted from sliding windows
- `HMM_TestPredictions.csv`: Model predictions on test data
- Various visualization files (PNG): Plots of sensor data, confusion matrices, transition matrices, etc.

## Dependencies

- Python 3.x
- pandas, numpy
- scipy, scikit-learn
- hmmlearn
- matplotlib, seaborn

## Usage

1. Mount Google Drive (for Colab environment)
2. Run the notebook cells sequentially
3. Generated datasets and visualizations will be saved automatically

## Key Insights

- HMMs effectively model the temporal dynamics of human activities
- Feature engineering combining time and frequency domains improves classification
- Separate models per activity allow for activity-specific state representations
- Post-processing techniques enhance prediction stability

## Authors

Group 1 - HMM Project