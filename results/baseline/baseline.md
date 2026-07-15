### Evaluation & Comparison Results

| Data Directory | Model Version | AUC | Interrupted Turns | Mean Response Delay | Operating Point |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | Silence Baseline | 0.514 | 0.0% | **1600 ms** | threshold=1.00, delay=1600ms |
| **English** | Starter `train.py` | 0.599 | 5.0% | **1190 ms** | threshold=0.55, delay=600ms |
| **English** | **New Calibrated Model** | **0.923** | **5.0%** | **638 ms** | threshold=0.45, delay=350ms |
| **Hindi** | Silence Baseline | 0.501 | 5.0% | **850 ms** | threshold=0.05, delay=850ms |
| **Hindi** | Starter `train.py` | 0.634 | 5.0% | **850 ms** | threshold=0.05, delay=850ms |
| **Hindi** | **New Calibrated Model** | **0.934** | **3.0%** | **355 ms** | threshold=0.45, delay=100ms |

### Observations:
* **English Performance**:
  - The **Silence Baseline** requires a full 1600 ms timeout to ensure no interruptions.
  - The **Starter `train.py`** improves the response delay to 1190 ms (AUC: 0.599) at the cost of 5% interruptions.
  - The **New Calibrated Model** significantly outperforms both, cutting the response delay to **638 ms** with a high AUC of **0.923**.
* **Hindi Performance**:
  - The **Silence Baseline** and **Starter `train.py`** are both stuck at a 850 ms response delay.
  - The **New Calibrated Model** achieves a breakthrough response delay of **355 ms** (AUC: **0.934**) with only **3%** interrupted turns, well within the 5% budget. This confirms that speaker-relative z-scoring and pitch contour slope features generalize exceptionally well to Hindi.