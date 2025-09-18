# Arrhythmia Dataset

- Download link: https://archive.ics.uci.edu/dataset/5/arrhythmia

## Objective:
- **Goal**: Based on 12-lead ECG and demographic features, determine the presence of arrhythmia and further classify each sample into one of 16 classes (01 = “normal”, 02–15 = various arrhythmia types, 16 = “others/unclassified”).

## Data:
- **Instances**: 452
- **Features**: 279 (206 continuous, others nominal/categorical)
- **Missing values**: Yes (explicitly noted by UCI)
- Features include demographics (age, sex, height, weight) and ECG-derived attributes such as QRS, PR, QT, T, P intervals, waveform widths/amplitudes/areas across multiple leads
- Feature indices are grouped by lead: DI, DII, DIII, aVR, aVL, aVF, V1–V6, each covering width, amplitude, and area measurements
- 16 class codes: 01=Normal; 02–15 correspond to specific arrhythmia types (ischemic changes, old anterior/inferior MI, sinus tachycardia/bradycardia, ventricular and supraventricular premature contraction, left/right bundle branch block, AV blocks, left ventricular hypertrophy, atrial fibrillation/flutter, etc.); 16=Others

## Preprocessing Method:
- Load arrhythmia.data with pandas.read_csv, treating ? as missing values (NaN)
- Verify column count is 280 (279 features + 1 class label) and assign headers:
- Attempt to parse column names from arrhythmia.names
- If parsing is incomplete, generate f001–f279 as placeholder feature names, and name the last column class
- Handle missing values as follows:
- Numeric features: impute with column median
- Categorical features: impute with column mode
- Keep all original 16 class labels (no collapsing to binary)
- Save the cleaned dataset as a standard CSV file for further modeling and visualization
