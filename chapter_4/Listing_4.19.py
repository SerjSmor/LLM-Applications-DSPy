from chapter_4.Listing_4_13 import evaluate_baseline

overall_scores = []
for _ in range(5):
    overall_scores.append(evaluate_baseline(num_threads=10))
