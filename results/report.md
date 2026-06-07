# Technical IQA Combo Benchmark: FLIVE+TOPIQ vs KonIQ+FLIVE

## 1. Executive Summary
After a rigorous benchmark across three public datasets (FLIVE, KonIQ, SPAQ) and a confirmed Hard-FP v2 dataset, **Combination B (KonIQ + FLIVE)** is identified as the safer and more reliable path for A-cut technical quality assessment.

While **Combination A (FLIVE + TOPIQ mixed112)** achieves higher SRCC on SPAQ and FLIVE, it fails significantly on the Hard-FP safety check, over-scoring technical failures by an average of **8.7 to 12.7 points** compared to Combination B. Given that the primary goal of the technical IQA module in the A-cut pipeline is to "guard" against bad photos, the superior safety of the KonIQ-based combination outweighs the marginal general quality gains of TOPIQ.

## 2. Protocol
- **Datasets**:
    - FLIVE test (N=3981)
    - KonIQ-10k test (N=1008)
    - SPAQ test (N=1125)
    - Hard-FP v2 (Confirmed technical failures, N=44)
- **Metrics**: SRCC, PLCC, MAE, RMSE, Bias, std_ratio.
- **Safety Metrics**: Mean score on Hard-FP (lower is better), violation counts > 65.
- **Score Scale**: All predictions and targets are normalized to 0..100 scale.
- **Bootstrap**: 1000 iterations, paired resampling, seed 42.

## 3. Available Prediction Sources
Predictions were extracted from the following verified directories:
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/` (General IQA)
- `outputs/eval_techiqa_guard_v1_hard_fp_confirmed_v2_20260522/` (Safety)

## 4. Score Normalization Checks
- All models (koniq_mobile, flive_mobile, topiq_mixed112) produce scores in the 0..100 range.
- Standard deviations for TOPIQ and FLIVE are generally lower (more compressed) than KonIQ on benchmark datasets.

## 5. Benchmark Results

### 5.1 FLIVE Test (N=3981)
| Model / Fusion | SRCC | PLCC | MAE | Bias | std_ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| koniq_mobile | 0.447 | 0.529 | 9.07 | -7.58 | 1.84 |
| flive_mobile | 0.630 | 0.757 | 3.12 | 1.08 | 0.75 |
| topiq_mixed112 | 0.469 | 0.563 | 5.02 | -2.36 | 1.22 |
| **A1: F+T Mean** | **0.574** | 0.697 | 3.47 | -0.64 | 0.90 |
| **B1: K+F Mean** | 0.534 | 0.646 | 4.98 | -3.25 | 1.19 |

### 5.2 KonIQ Test (N=1008)
| Model / Fusion | SRCC | PLCC | MAE | Bias | std_ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| koniq_mobile | 0.866 | 0.891 | 5.59 | 1.18 | 0.92 |
| flive_mobile | 0.688 | 0.735 | 14.26 | 13.71 | 0.36 |
| topiq_mixed112 | 0.863 | 0.884 | 5.80 | 2.46 | 0.85 |
| **A1: F+T Mean** | 0.856 | 0.875 | 9.10 | 8.08 | 0.58 |
| **B1: K+F Mean** | **0.864** | 0.886 | 8.52 | 7.44 | 0.61 |

### 5.3 SPAQ Test (N=1125)
| Model / Fusion | SRCC | PLCC | MAE | Bias | std_ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| koniq_mobile | 0.856 | 0.845 | 9.58 | -0.32 | 0.91 |
| flive_mobile | 0.827 | 0.681 | 20.88 | 18.42 | 0.44 |
| topiq_mixed112 | 0.899 | 0.893 | 7.91 | 1.30 | 0.83 |
| **A1: F+T Mean** | **0.902** | 0.867 | 13.32 | 9.86 | 0.60 |
| **B1: K+F Mean** | 0.887 | 0.819 | 13.78 | 9.05 | 0.61 |

## 6. Hard-FP v2 Safety Results (N=44, Lower is Better)
| Model / Fusion | Mean Score | Median | Count > 65 | Status |
| :--- | :---: | :---: | :---: | :--- |
| koniq_mobile | 47.8 | 47.9 | 0 | Strongest Guard |
| flive_mobile | 74.2 | 74.4 | 43 | No Guard |
| topiq_mixed112 | 65.4 | 68.4 | 28 | Weak Guard |
| **A1: F+T Mean** | **69.8** | 70.6 | 37 | **FAIL (Dangerous)** |
| **B1: K+F Mean** | **61.0** | 61.6 | 7 | **PASS (Safe)** |
| A4: F+T Min-Cap | 68.6 | 70.6 | 36 | Weak improvement |
| B4: K+F KonIQ-Guard | 55.8 | 55.9 | 1 | Very Safe |
| B5: K+F Disagree-Guard| 52.8 | 52.9 | 0 | Safest |

### Top 15 Safety Failures (Sorted by Combination A Error)
| image_path | koniq | flive | topiq | A4 (F+T) | B4 (K+F) | Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| EMOTIC..._440339.jpg | 52.4 | 77.1 | 76.6 | 76.9 | 60.4 | low_res |
| EMOTIC..._gapet.jpg | 47.8 | 75.7 | 77.5 | 76.6 | 55.8 | technical_bad |
| 1675342165226-3.jpg | 36.5 | 71.9 | 78.8 | 75.4 | 44.5 | manual_flagged |
| EMOTIC..._051324.jpg | 55.2 | 75.5 | 71.9 | 73.7 | 63.2 | technical_bad |

## 7. Bootstrap Confidence Intervals (A1 vs B1 Delta)
- **Delta SRCC (A1 - B1)**: 0.057 [0.047, 0.067] (A1 is significantly better at ranking)
- **Delta Hard-FP Mean (A1 - B1)**: **8.77 [6.90, 10.64]** (A1 is significantly more dangerous)
- **Delta Hard-FP Mean (A4 - B4)**: **12.76 [9.71, 15.45]** (Guard versions confirm the gap)

## 8. FLIVE+TOPIQ vs KonIQ+FLIVE Direct Comparison
- **Ranking**: Combination A wins on general IQA benchmarks (Average SRCC ~0.77 vs ~0.76).
- **Safety**: Combination B wins decisively on technical failure detection. Combination A over-scores confirmed "bad" images by nearly 9 points on average.
- **Accuracy (KonIQ)**: Combination B is more accurate on the high-quality KonIQ dataset.

## 9. Interpretation
TOPIQ mixed112 is a powerful general IQA model, but it inherits the "semantic bias" common in recent high-capacity models: it tends to give high scores to technically poor images if the content (semantics) is attractive. FLIVE-Mobile shares this bias. KonIQ-Mobile, being trained with synthetic distortions, remains the only model that consistently penalizes low-level technical artifacts (noise, blur, low resolution).

Using FLIVE + TOPIQ creates a "Safety Blindspot" where technical failures are consistently over-scored as "Good" (avg 69.8). KonIQ + FLIVE maintains a safe guard (avg 61.0), which can be further improved with KonIQ-specific capping (B4, avg 55.8).

## 10. Recommendation
**Choice B: Keep current KonIQ + FLIVE (or optimize within this combination).**

Replacing KonIQ with TOPIQ in a two-model fusion is rejected due to safety risks. If higher performance is required, a three-model path (KonIQ+FLIVE+TOPIQ) should be investigated, but KonIQ MUST remain as a primary weight or min-cap guard to prevent technical false positives.

## 11. No-Go Conditions
- **DO NOT** deploy FLIVE + TOPIQ mean fusion (A1) or simple variants.
- **DO NOT** use FLIVE or TOPIQ as the sole "guard" for low-resolution or technical bad images.
- **NO-GO** on any fusion that results in a Hard-FP v2 mean score above 60.0.
