import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def validate_lengths(predictions_list, labels_list):
    if len(predictions_list) != len(labels_list):
        raise ValueError("Predictions and labels lists must have the same length."
                         f" Got {len(predictions_list)} and {len(labels_list)}.")


def get_species_predictions(predictions_path):
    modelResults_df = pd.read_csv(predictions_path)
    predictions_list = modelResults_df["7"].tolist()
    probs_list = modelResults_df["final_prob"].tolist()

    return predictions_list, probs_list


def read_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels_list = [int(line.strip()) for line in f]

    return labels_list


def plot_cumulative(list_of_results, accuracies, model_id):

    plt.figure(figsize=(8, 6))
    plt.plot([0, 100], [0, 100],  label="Ideal Calibration", color='gray')

    for results in list_of_results:
      # Sort by predicted probabilities (ascending order)
      sorted_indices = np.argsort(results['probs'])
      sorted_probs = results['probs'][sorted_indices]
      sorted_correct = results['correctness'][sorted_indices]

      # Compute cumulative probability sum
      cumulative_probs = np.cumsum(sorted_probs)  # Raw cumulative probability sums
      cumulative_correct = np.cumsum(sorted_correct)  # Raw cumulative correct sums

      n = len(sorted_correct)
      cumulative_probs =  cumulative_probs / n * 100  # Normalize to percentage
      cumulative_correct = cumulative_correct / n * 100  # Normalize to percentage

      # Plot cumulative probability vs. cumulative correct
      plt.plot(cumulative_probs, cumulative_correct, color='black')

      # Highlight the last point with a marker
      plt.plot(cumulative_probs[-1], cumulative_correct[-1], marker='o', markersize=8, color='black')

    plt.xlabel("Cumulative Probability")
    plt.ylabel("Cumulative Correct")
    plt.title(f"PROTAX-GPU {model_id}\n Accuracy={np.mean(accuracies)}%")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_id}_Calibration.png")
    plt.close()

    return


def calculate_correctness(predictions_list, probs_list, labels_list):

  correctness_labels = []

  for i in range(len(predictions_list)):
    if predictions_list[i] == labels_list[i]:
      correctness_labels.append(1)

    else:
      correctness_labels.append(0)

  results = {}
  results["probs"] = np.array(probs_list)
  results["correctness"] = np.array(correctness_labels)

  # Calculate final accuracy
  accuracy = (correctness_labels.count(1) / len(predictions_list)) * 100
  print(f"Accuracy: {accuracy:.2f}%")

  return results, accuracy


def evaluate(predictions_path, labels_path, model_id):

    predictions_list, probs_list = get_species_predictions(predictions_path)  
    labels_list = read_labels(labels_path)
    validate_lengths(predictions_list, labels_list)

    results, accuracy = calculate_correctness(predictions_list, probs_list, labels_list)

    plot_cumulative([results], [accuracy], model_id)

    return

