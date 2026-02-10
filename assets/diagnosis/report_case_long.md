# Single-Sentence Diagnosis Report

- Case ID: `case_long`
- Model: `Helsinki-NLP/opus-mt-de-en`
- Auto label: `omission`
- Reason: Many source tokens have low attention coverage.

## Text

- Source: Trotz der schwierigen Verhandlungen stimmte das Parlament dem Abkommen mit großer Mehrheit zu.
- Prediction: Despite the difficult negotiations, Parliament approved the agreement by a large majority.
- Reference: Despite the difficult negotiations, Parliament approved the agreement by a large majority.

## Diagnostics

- Low-coverage fraction: 0.8667
- Length ratio (pred/ref): 1.0
- Bigram repetition: 0.0
- Highlighted source: [Trotz] [der] [schwierigen] [Verhandlungen] [stimmt] [e] [das] [Parlament] [dem] Abkommen [mit] großer [Mehrheit] [zu] [.]
- Heatmap: `assets/diagnosis\heatmap_case_long.png`
