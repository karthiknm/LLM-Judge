Evaluation Summary
==================

Main output files
-----------------
- combined_summeval_eval_all_available.csv
- combined_summeval_eval_common_ids.csv
- combined_model_comparison_common_ids.csv
- combined_model_coverage.csv

Final model-vs-SummEval results
-------------------------------
Macro average over the four dimensions:

1.5B
- Spearman: 0.3878
- Pearson: 0.4339
- Kendall: 0.3468
- MAE: 0.6786
- RMSE: 1.0133
- Exact Match: 0.4088
- Off-by-1 Accuracy: 0.7814

3B
- Spearman: 0.4322
- Pearson: 0.4743
- Kendall: 0.3868
- MAE: 0.7783
- RMSE: 1.0217
- Exact Match: 0.2656
- Off-by-1 Accuracy: 0.7552

Interpretation summary
----------------------
- The 3B model is stronger at preserving the relative ordering of summary
  quality, which is why it wins the correlation metrics.
- The 1.5B model is stronger at matching the exact human-assigned score on
  each example, which is why it wins MAE, RMSE, exact match, and off-by-1.
- This is not a contradiction: the two models are better at different parts
  of the evaluation problem.

Dimension-level interpretation
------------------------------
- Coherence: 3B is the stronger judge. It tracks human coherence judgments
  better and is also much closer on absolute error.
- Consistency: 3B is the strongest dimension overall and is the clearest win
  for the 3B model.
- Fluency: 1.5B is much better on absolute score matching. The 3B model tends
  to score fluency too low.
- Relevance: mixed result. 3B is slightly better at ranking summaries by
  relevance, but 1.5B is closer to the gold score numerically.

Calibration notes
-----------------
- 1.5B tends to over-score coherence.
- 3B tends to under-score fluency.
- 3B also under-scores relevance.
- Both models are relatively well calibrated on consistency.

Recommended takeaway
--------------------
- If the goal is ranking or selecting summaries in a way that best follows
  human preferences, prefer the 3B model.
- If the goal is reproducing SummEval-style numeric labels as closely as
  possible, prefer the 1.5B model.
- In a report, the safest conclusion is: 3B is better for ranking fidelity,
  while 1.5B is better for absolute score calibration.

Anything else to do
-------------------
Nothing is required for the evaluation itself. The final metrics already use
all 1600 examples for both models.

Optional cleanup items:
- Repair the 8 raw 3B rows so `parse_ok` also becomes 1600/1600.
- Use `combined_summeval_eval_common_ids.csv` as the main table in a report.
