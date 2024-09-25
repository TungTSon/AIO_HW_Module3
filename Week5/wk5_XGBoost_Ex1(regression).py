import numpy as np

# Step 1: Data initialization
X = np.array([23, 24, 26, 27])
Y = np.array([50, 70, 80, 85])

# Hyperparameters
lr = 0.3  # Learning rate
lambda_ = 0.0  # Regularization parameter
depth = 1  # Tree depth

# Step 1: Initial prediction as the mean of Y
initial_prediction = np.mean(Y)
print(f'f0 = {initial_prediction}')
residuals = Y - initial_prediction

# Step 2: Similarity Score calculation for root


def similarity_score(residuals, lambda_):
    sum_residuals = np.sum(residuals)
    score = (sum_residuals ** 2) / (len(residuals) + lambda_)
    return score


root_similarity_score = similarity_score(residuals, lambda_)
print(f"Root Similarity Score: {root_similarity_score}")

# Step 3: Calculate Similarity Score for potential splits


def calculate_gain(x, residuals, threshold, lambda_):
    left_mask = x < threshold
    right_mask = x >= threshold

    left_residuals = residuals[left_mask]
    right_residuals = residuals[right_mask]

    left_similarity = similarity_score(left_residuals, lambda_)
    right_similarity = similarity_score(right_residuals, lambda_)

    return left_similarity, right_similarity


# Step 4: Evaluate all possible splits
splits = [23.5, 25, 26.5]
best_gain = -np.inf
best_split = None

for split in splits:
    left_similarity, right_similarity = calculate_gain(
        X, residuals, split, lambda_)
    gain = left_similarity + right_similarity - root_similarity_score
    print(
        f"Left_similarity = {left_similarity}, Right_similarity = {right_similarity}")
    print(f"Split: {split}, Gain: {gain}")

    if gain > best_gain:
        best_gain = gain
        best_split = split

print(f"Best Split: {best_split}, Best Gain: {best_gain}")

# Step 5: Calculate Output values for each leaf


def calculate_output(residuals, lambda_):
    sum_residuals = np.sum(residuals)
    output = sum_residuals / (len(residuals) + lambda_)
    return output


# Apply best split to the data
left_mask = X < best_split
right_mask = X >= best_split

left_residuals = residuals[left_mask]
right_residuals = residuals[right_mask]

left_output = calculate_output(left_residuals, lambda_)
right_output = calculate_output(right_residuals, lambda_)

print(f"Left Output: {left_output}, Right Output: {right_output}")

# Step 6: Update the predictions
new_predictions = np.where(X < best_split,
                           initial_prediction + lr * left_output,
                           initial_prediction + lr * right_output)

print(f"Updated Predictions: {new_predictions}")
