# Single-Sentence Diagnosis Report

- Case ID: `case_short`
- Model: `Helsinki-NLP/opus-mt-de-en`
- Auto label: `omission`
- Reason: Many source tokens have low attention coverage.

## Text

- Source: Wir danken dem Ausschuss für seine Arbeit.
- Prediction: We thank the committee for its work.
- Reference: We thank the committee for its work.

## Diagnostics

- Low-coverage fraction: 0.875
- Length ratio (pred/ref): 1.0
- Bigram repetition: 0.0
- Highlighted source: [Wir] [danken] [dem] Ausschuss [für] [seine] [Arbeit] [.]
- Heatmap: `assets/diagnosis\heatmap_case_short.png`
