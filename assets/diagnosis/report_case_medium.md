# Single-Sentence Diagnosis Report

- Case ID: `case_medium`
- Model: `Helsinki-NLP/opus-mt-de-en`
- Auto label: `omission`
- Reason: Many source tokens have low attention coverage.

## Text

- Source: Die Präsidentin erklärte, dass die Debatte morgen fortgesetzt wird.
- Prediction: The President said that the debate will continue tomorrow.
- Reference: The President stated that the debate will continue tomorrow.

## Diagnostics

- Low-coverage fraction: 0.7273
- Length ratio (pred/ref): 1.0
- Bigram repetition: 0.0
- Highlighted source: [Die] [Präsidentin] [erklärte] [,] [dass] [die] Debatte morgen fortgesetzt [wird] [.]
- Heatmap: `assets/diagnosis\heatmap_case_medium.png`
