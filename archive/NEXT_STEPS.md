# 🏎️ F1 Prediction Project: Next Steps (Phase 2)

**Phase 1 (Model Refinement)** is complete. 🏆
**Winner**: Tuned Causal XGBoost (MAE 0.4119s).

## ⚠️ Critical Insight (Monaco Update)
We discovered that the model under-performs on narrow tracks like **Monaco** (MAE ~0.9s).
*   **Reason**: The current model treats `FieldDelta` (traffic) the same way for every track.
*   **Reality**: 0.5s gap in Monaco = Stuck. 0.5s gap in Monza = Overtaking opportunity.

## 1. Immediate Action: Feature Engineering 🛠️
To fix the Monaco issue, we need to add **Track Topology Features** in Phase 2:
- **`OvertakingDifficultyIndex`**: A static score for each track (e.g., Monaco=10, Monza=2).
- **`TrackWidth`**: Average circuit width.
- **`DirtyAirFactor`**: How much does being behind a car hurt lap time on *this* specific track?

## 2. Application Development 📱
Build a user interface to visualize predictions.
- **Web App**: A Next.js/React dashboard showing real-time lap predictions.
- **Features**:
    - "Race Strategy" simulator.
    - "Driver vs Driver" pace comparison.

## 3. Real-Time Integration ⏱️
- **Pipeline**: Ingest live timing data (OpenF1 API / FastF1).
- **Inference**: Run `final_f1_model.pkl`.

---
*Status: Model Verified (Weakness identified in Monaco).*
