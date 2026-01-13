# 🧪 Model Experimentation Log

This document summarizes the rigorous testing performed to optimize the F1 Lap Time Prediction Model.

## 🏆 The Champion
**Model v1 (Single XGBoost)**
- **MAE**: `0.4119s`
- **Configuration**: 6 Features, Optimized Depth (9), 108 Trees.
- **Why it wins**: Best generalization. It learns shared physics (Tyre/Fuel) from the entire dataset without fragmentation.
- **Status**: **DEPLOYED** ✅

## 🥈 The Runner-Up
**Model v2 (Condition Difficulty)**
- **MAE**: `0.4208s`
- **Configuration**: 7 Features (Includes `DifficultyInteraction`), Single Model.
- **Strength**: Handles outliers (e.g., Azerbaijan Street Circuit logic) better.
- **Weakness**: Slightly less accurate on "Normal" laps due to added feature noise.
- **Verdict**: Excellent "V2" prototype for Phase 2.

## 🥉 The Experiments (Dual Models)

### Experiment A: Split Models (Clear vs Traffic)
- **Concept**: Train one model for Clear Air, one for Traffic.
- **MAE**: `0.428s`
- **Result**: **Failed**. Splitting the data reduced the training samples for each model, hurting their ability to learn fundamental physics.

### Experiment B: Dual Model + Pace Delta
- **Concept**: Add `DriverAheadDelta` (Pace difference) to the Traffic Model.
- **MAE**: `0.421s` (Improved).
- **Result**: **Parity with v2**. The feature works, but the Dual Architecture bottleneck (data fragmentation) prevents it from beating the Champion.

## 🔑 Key Takeaways for Presentation
1.  **Simplicity Won**: A single, well-tuned model beat complex multi-model architectures.
2.  **Data Volume Matters**: Splitting data for "Specialized Experts" hurt performance more than the specialization helped.
3.  **Future Feature**: The "Pace Delta" feature is validated as useful and should be integrated into the Single Model in Phase 2.
