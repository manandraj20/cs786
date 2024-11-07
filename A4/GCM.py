import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

# Load training data
X_train = pd.read_csv('./X.csv', header=None)
weights_train = X_train.iloc[:, 0].values
heights_train = X_train.iloc[:, 1].values
categories_train = X_train.iloc[:, 2].values

# Load test data
X_test = pd.read_csv('./y.csv', header=None)
weights_test = X_test.iloc[:, 0].values
heights_test = X_test.iloc[:, 1].values

# Define similarity function with weight preference
def similarity(x1, x2, weight_pref=0.7):
    """
    Calculate weighted Euclidean distance for similarity.
    A higher weight_pref favors weight over height.
    """
    weight_distance = (x1[0] - x2[0]) ** 2
    height_distance = (x1[1] - x2[1]) ** 2
    return np.exp(-weight_pref * weight_distance - (1 - weight_pref) * height_distance)

# Politeness factor: threshold for assigning categories
# This is a hypothetical adjustment value to favor 'average' over 'large'
politeness_threshold = 0.7

# Prediction function
def predict_category(test_point, X_train, y_train, weight_pref=0.7):
    """
    Predicts category based on training data and weighted similarity.
    """
    similarities = np.array([similarity(test_point, [weights_train[i], heights_train[i]], weight_pref)
                             for i in range(len(weights_train))])
    
    # Find the most similar training points
    best_matches = np.argsort(similarities)[::-1][:3]  # Taking top 3 matches
    avg_similarity = similarities[best_matches].mean()
    
    # Weighted voting with politeness adjustment
    category_votes = categories_train[best_matches]
    if avg_similarity < politeness_threshold and 3 in category_votes:
        return 2  # Assign 'average' if similarity is low and 'large' is among top matches
    else:
        return np.bincount(category_votes).argmax()  # Return most common category among best matches

# Apply predictions to test data
predicted_categories = [predict_category([weights_test[i], heights_test[i]], X_train, categories_train)
                        for i in range(len(weights_test))]

# Output results
for i, category in enumerate(predicted_categories):
    print(f"Test data point {i+1} (weight: {weights_test[i]}, height: {heights_test[i]}): Predicted category = {category}")
