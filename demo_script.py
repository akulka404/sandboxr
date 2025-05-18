# demo_script.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import nltk
import ssl
import warnings
warnings.filterwarnings('ignore')

# Handle NLTK downloads with SSL verification disabled
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
print("\nDownloading NLTK data...")
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)  # Open Multilingual Wordnet

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter

# Create sample data
np.random.seed(42)
dates = pd.date_range('20230101', periods=100)
data = {
    'A': np.random.randn(100).cumsum(),
    'B': np.random.randn(100).cumsum(),
    'C': np.random.randn(100).cumsum()
}

# Create DataFrame
df = pd.DataFrame(data, index=dates)

# Create a figure with multiple subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# 1. Time Series Plot
for column in df.columns:
    ax1.plot(df.index, df[column], label=column)
ax1.set_title('Time Series of Cumulative Returns')
ax1.legend()
ax1.grid(True)

# 2. Rolling Statistics
rolling_mean = df.rolling(window=7).mean()
rolling_std = df.rolling(window=7).std()
ax2.plot(rolling_mean['A'], label='7-day Rolling Mean', color='blue')
ax2.fill_between(rolling_mean.index, 
                 rolling_mean['A'] - rolling_std['A'],
                 rolling_mean['A'] + rolling_std['A'],
                 alpha=0.2, color='blue')
ax2.set_title('Rolling Statistics (Column A)')
ax2.legend()
ax2.grid(True)

# 3. Correlation Plot
corr = df.corr()
im = ax3.imshow(corr, cmap='coolwarm')
ax3.set_title('Correlation Matrix')
plt.colorbar(im, ax=ax3)
ax3.set_xticks(range(len(df.columns)))
ax3.set_yticks(range(len(df.columns)))
ax3.set_xticklabels(df.columns)
ax3.set_yticklabels(df.columns)

# 4. Scatter Plot
ax4.scatter(df['A'], df['B'], alpha=0.5)
ax4.set_title('Scatter Plot: A vs B')
ax4.grid(True)

# Adjust layout and save
plt.tight_layout()
output_dir = 'output'
output_path = os.path.join(output_dir, 'analysis_plots.png')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

try:
    plt.savefig(output_path)
    print(f"\nPlot saved as '{output_path}'")
except Exception as e:
    print(f"\nWarning: Could not save plot: {str(e)}")

# Print statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Calculate and print financial metrics
# Handle log calculation more safely
log_returns = pd.DataFrame()
for col in df.columns:
    # Calculate returns first, then take log
    returns = df[col].pct_change()
    # Replace inf and -inf with NaN
    returns = returns.replace([np.inf, -np.inf], np.nan)
    # Fill NaN with 0
    returns = returns.fillna(0)
    # Add small epsilon to avoid log1p(0)
    returns = returns + 1e-10
    # Take log of (1 + returns)
    log_returns[col] = np.log1p(returns)

volatility = log_returns.std() * np.sqrt(252)
risk_free_rate = 0.02
sharpe_ratio = (log_returns.mean() * 252 - risk_free_rate) / volatility

print("\nAnnualized Volatility:")
print(volatility)
print("\nSharpe Ratio:")
print(sharpe_ratio)

# Print matrix operations results
print("\nMatrix Operations:")
matrix = np.random.randn(5, 5)
print("Matrix Rank:", np.linalg.matrix_rank(matrix))
print("Determinant:", np.linalg.det(matrix))

print("\n=== Machine Learning Analysis ===")
# Generate synthetic classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': [f'Feature_{i}' for i in range(X.shape[1])],
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

print("\n=== Natural Language Processing Analysis ===")
# Sample text data
texts = [
    "Machine learning is transforming the way we analyze data.",
    "Natural language processing helps computers understand human language.",
    "Deep learning models can process complex patterns in data.",
    "Data science combines statistics, programming, and domain knowledge.",
    "Artificial intelligence is revolutionizing various industries."
]

try:
    # Text preprocessing
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    processed_texts = []
    for text in texts:
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words
        ]
        
        processed_texts.append(processed_tokens)

    # Word frequency analysis
    all_words = [word for text in processed_texts for word in text]
    word_freq = Counter(all_words)

    print("\nMost Common Words:")
    for word, freq in word_freq.most_common(5):
        print(f"{word}: {freq}")

    # Calculate text statistics
    text_stats = {
        'avg_words_per_text': np.mean([len(text) for text in processed_texts]),
        'total_unique_words': len(set(all_words)),
        'most_common_word': word_freq.most_common(1)[0][0],
        'vocabulary_size': len(word_freq)
    }

    print("\nText Statistics:")
    for stat, value in text_stats.items():
        print(f"{stat}: {value}")

except Exception as e:
    print(f"\nError in NLP processing: {str(e)}")
    print("Continuing with other analyses...")

print("\n=== Advanced Numerical Analysis ===")
# Generate complex time series data
np.random.seed(42)
t = np.linspace(0, 10, 1000)
signal = (
    np.sin(t) + 
    0.5 * np.sin(2*t) + 
    0.25 * np.sin(3*t) + 
    0.1 * np.random.randn(1000)
)

# Perform FFT analysis
fft_result = np.fft.fft(signal)
freq = np.fft.fftfreq(len(signal))

# Find dominant frequencies
magnitude = np.abs(fft_result)
dominant_freq_idx = np.argsort(magnitude)[-5:][::-1]
dominant_freqs = freq[dominant_freq_idx]

print("\nDominant Frequencies in Signal:")
for i, freq in enumerate(dominant_freqs):
    print(f"Frequency {i+1}: {freq:.3f} Hz")

# Calculate signal statistics
signal_stats = {
    'mean': np.mean(signal),
    'std': np.std(signal),
    'skewness': pd.Series(signal).skew(),
    'kurtosis': pd.Series(signal).kurtosis()
}

print("\nSignal Statistics:")
for stat, value in signal_stats.items():
    print(f"{stat}: {value:.4f}")

# Matrix operations
print("\n=== Complex Matrix Operations ===")
# Create a complex matrix
complex_matrix = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

# Perform various matrix operations
matrix_operations = {
    'determinant': np.linalg.det(complex_matrix),
    'rank': np.linalg.matrix_rank(complex_matrix),
    'eigenvalues': np.linalg.eigvals(complex_matrix),
    'condition_number': np.linalg.cond(complex_matrix)
}

print("\nMatrix Analysis Results:")
for operation, result in matrix_operations.items():
    if isinstance(result, np.ndarray):
        print(f"{operation} (first 3 values): {result[:3]}")
    else:
        print(f"{operation}: {result:.4f}")

