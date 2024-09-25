import numpy as np
# Step 1: Data initialization
X = np.array([23, 24, 26, 27])
Y = np.array([0, 0, 1, 1])  # False is 0, True is 1

# Hyperparameters
lr = 0.3  # Learning rate
lambda_ = 0.0  # Regularization parameter
depth = 1  # Tree depth is 1

# Step 1: Initial probability prediction (starting with 0.5 for classification)
initial_probability = 0.5
previous_probabilities = np.full(Y.shape, initial_probability)

# Residuals (classification case)
residuals = Y - previous_probabilities

# Step 2: Similarity Score calculation for root


def similarity_score_classification(residuals, previous_probabilities, lambda_):
    sum_residuals = np.sum(residuals)
    sum_prob_adjustment = np.sum(
        previous_probabilities * (1 - previous_probabilities))
    score = (sum_residuals ** 2) / (sum_prob_adjustment + lambda_)
    return score


root_similarity_score = similarity_score_classification(
    residuals, previous_probabilities, lambda_)
print(f"Root Similarity Score: {root_similarity_score}")

# Step 3: Calculate Similarity Score for potential splits


def calculate_gain_classification(x, residuals, previous_probabilities, threshold, lambda_):
    left_mask = x < threshold
    right_mask = x >= threshold

    left_residuals = residuals[left_mask]
    right_residuals = residuals[right_mask]

    left_probabilities = previous_probabilities[left_mask]
    right_probabilities = previous_probabilities[right_mask]

    left_similarity = similarity_score_classification(
        left_residuals, left_probabilities, lambda_)
    right_similarity = similarity_score_classification(
        right_residuals, right_probabilities, lambda_)

    return left_similarity, right_similarity


# Step 4: Evaluate all possible splits
splits = [23.5, 25, 26.5]
best_gain = -np.inf
best_split = None

for split in splits:
    left_similarity, right_similarity = calculate_gain_classification(
        X, residuals, previous_probabilities, split, lambda_)
    gain = left_similarity + right_similarity - root_similarity_score
    print(f"Split: {split}, Gain: {gain}")
    print(
        f"Left Similarity: {left_similarity}, Right Similarity: {right_similarity}")

    if gain > best_gain:
        best_gain = gain
        best_split = split

print(f"Best Split: {best_split}, Best Gain: {best_gain}")

# Step 5: Calculate Output values for each leaf


def calculate_output_classification(residuals, previous_probabilities, lambda_):
    sum_residuals = np.sum(residuals)
    sum_prob_adjustment = np.sum(
        previous_probabilities * (1 - previous_probabilities))
    output = sum_residuals / (sum_prob_adjustment + lambda_)
    return output


# Apply best split to the data
left_mask = X < best_split
right_mask = X >= best_split

left_residuals = residuals[left_mask]
right_residuals = residuals[right_mask]

left_probabilities = previous_probabilities[left_mask]
right_probabilities = previous_probabilities[right_mask]

left_output = calculate_output_classification(
    left_residuals, left_probabilities, lambda_)
right_output = calculate_output_classification(
    right_residuals, right_probabilities, lambda_)

print(f"Left Output: {left_output}, Right Output: {right_output}")

# Step 6: Update Log Odds and Predictions


def update_log_odds(previous_probabilities, lr, output):
    log_prediction = np.log(previous_probabilities /
                            (1 - previous_probabilities))
    new_log_prediction = log_prediction + lr * output
    new_probabilities = 1 / (1 + np.exp(-new_log_prediction))
    return new_probabilities


# Update the probabilities for each region based on the output
new_probabilities = np.where(X < best_split,
                             update_log_odds(
                                 previous_probabilities, lr, left_output),
                             update_log_odds(previous_probabilities, lr, right_output))

print(f"Updated Probabilities: {new_probabilities}")
