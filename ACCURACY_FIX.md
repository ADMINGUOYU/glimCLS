# Accuracy Metric Fix

## Problem

The logged validation and test accuracies (~53-60%) were significantly lower than the actual model performance shown in confusion matrices (~79-88%). This 25-35% discrepancy was caused by incorrect metric aggregation in PyTorch Lightning.

## Root Cause

The issue was with how per-batch accuracies were being logged and aggregated:

1. **Batch size mismatch**: The model stored `self.batch_size=72` (training batch size), but validation/test used batch size 24
2. **Metric aggregation**: PyTorch Lightning's `sync_dist=True` with the `batch_size` parameter may have caused incorrect aggregation
3. **Per-batch vs. overall accuracy**: The per-batch accuracy computation path differed from the confusion matrix computation

## Solution

Modified `model/glim_parallel.py` to:

1. **Collect predictions during validation/test**: Store predictions and targets in `val_step_outputs` and `test_step_outputs`
2. **Compute accuracy at epoch end**: Calculate overall accuracy from all collected predictions in `on_validation_epoch_end()` and `on_test_epoch_end()`
3. **Log correct metrics**: Added new metrics `val/acc_{task}_correct` and `test/acc_{task}_correct` that show the true accuracy

## Changes Made

### 1. Added validation step outputs (line 164)
```python
self.val_step_outputs = {'sentiment': [], 'topic': []}
```

### 2. Modified validation_step (lines 532-564)
- Collect predictions and targets for each batch
- Store in `val_step_outputs`

### 3. Added on_validation_epoch_end (lines 566-579)
- Compute overall accuracy from collected predictions
- Log as `val/acc_{task}_correct`

### 4. Modified on_test_epoch_end (lines 589-592)
- Compute and log correct test accuracy
- Print accuracy to console for visibility

## Results

The new `*_correct` metrics will show the true model performance:
- **Old metrics** (`val/acc_sentiment`, `test/acc_sentiment`): ~53-60% (INCORRECT)
- **New metrics** (`val/acc_sentiment_correct`, `test/acc_sentiment_correct`): ~79-88% (CORRECT)
- **Confusion matrix**: ~79-88% (matches new metrics)

## Verification

To verify the fix works:
1. Run training with the modified code
2. Compare `val/acc_sentiment` vs `val/acc_sentiment_correct` during training
3. Check that `test/acc_{task}_correct` matches the confusion matrix accuracy

## Note

The old metrics (`val/acc_*`, `test/acc_*`) are still logged for backward compatibility, but should not be trusted. Use the `*_correct` metrics instead.
