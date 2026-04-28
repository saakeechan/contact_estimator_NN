# Solutions for Promoting Contact Continuity

## Problem
Contact predictions can be noisy with rapid switching between contact/no-contact states. We want smoother, more stable predictions.

## Solutions (Ordered by Complexity)

### 1. **Label Smoothing** ⭐ RECOMMENDED - Easiest
Prevents overconfident 0/1 predictions, creating smoother transitions.

**Implementation**: Modify BCEWithLogitsLoss to use smoothed labels
- Instead of [0, 1], use [ε, 1-ε] where ε = 0.05-0.1
- Benefits: No architecture change, just loss modification
- Add `label_smoothing` parameter to config

**Effect**: Model becomes less confident → smoother probability transitions → fewer spurious flips

---

### 2. **Temporal Consistency Loss** ⭐ RECOMMENDED - Most Direct
Explicitly penalize differences between consecutive predictions.

**Implementation**: Add loss term measuring prediction changes over time
- Fetch consecutive windows in training
- Compute: `temporal_loss = |prediction[t] - prediction[t-1]|`
- Total loss: `BCE_loss + temporal_lambda * temporal_loss`
- Add `temporal_lambda` parameter to config (try 0.01-0.1)

**Effect**: Directly trains model to produce smooth predictions over time

---

### 3. **Weighted BCE Loss**
If contact is rare, upweight contact class to reduce false negatives.

**Implementation**: Use pos_weight parameter in BCEWithLogitsLoss
- `pos_weight = (num_no_contact / num_contact)` per leg
- Helps if model tends to predict "no contact" too often
- Add `pos_weight` parameter to config

---

### 4. **Post-Processing Smoothing** (Last Resort)
Apply temporal filter to predictions after inference.

**Options**:
- Median filter: Replace each prediction with median of surrounding window
- Moving average: Average predictions over sliding window
- Hysteresis: Require N consecutive predictions before switching state

**Downside**: Doesn't improve model training, just post-processes

---

## Recommended Approach

**Start with #1 (Label Smoothing) + #2 (Temporal Consistency)**:
1. Easy to implement together
2. Complementary effects
3. No architecture changes needed

Config additions:
```yaml
# Continuity promotion
label_smoothing: 0.05      # Smooth labels: [0,1] → [0.05, 0.95]
temporal_lambda: 0.05      # Weight for temporal consistency loss
```

## Expected Improvements
- Fewer rapid state changes
- More stable contact predictions
- Better generalization to noisy sensor data
- Slight reduction in peak accuracy (worth the tradeoff for stability)
