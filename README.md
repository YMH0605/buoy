# LH Farm Buoy LSTM — Quick README

Short pipeline to fetch buoy data from Firebase, convert DO %→mg/L, build sliding‑window tensors, and run a PyTorch LSTM for short‑term forecasts.

## What it does

* Pulls latest data from `/LH_Farm/pond_<ID>` (type=`a_buoy`).
* Converts DO % to mg/L using temp & pressure corrections (Henry‑law based).
* Creates 4‑step sequences → predict next 1 step (10‑min resolution), rolled out for `n_ahead`.
* Model inputs: `[DO_mgL, Temp_C, hour_minute]`; outputs: `[DO_mgL, Temp_C]`.

## Requirements

```
numpy pandas matplotlib scipy scikit-learn firebase-admin pytz torch
pytorch-lightning meteostat pvlib astral
```

## Setup

1. Put Firebase service account JSON at project root as `fb_key.json` (do **not** commit).
2. Ensure Realtime DB has `/LH_Farm/pond_1` … with keys: `do`, `temp`, `pressure`, `init_do`, `type`.

## Quickstart

```python
from data_processing import *

# 1. Load the model
filepath = 'lstm_on_bouy.pth'
lstm = LSTM(num_classes=2, input_size=3, hidden_size=64, num_layers=3).to(device)
lstm.load_state_dict(torch.load(filepath, map_location=device))

# 2. Prediction Process
dataX, do, future_predicts, train_size, val_size, mean, std = Predict(
    1,     # First parameter：pond_id (1, 2, 5, 18, 19, 21, 22, 30, 52)
    6,     # Second parameter：how many time steps ahead (10mins each step，6 steps = 60mins)
    lstm   # Third parameters：Loaded model
)

# 3. See Prediction Result
print(future_predicts)
```

## Defaults

* `seq_length=4`, `n_future=1` (iterative rollout)
* timestep = 10 minutes
* timezone: UTC → America/Chicago
* date filter: index[:8] > `20250701`

## Notes

* Only last 20 results are loaded to faster the process.
* Standardization is fit on the train slice inside `Load_Data`.
* If you split utilities to `data_processing.py`, remove duplicates in `main.py`.

## Troubleshooting

* **`fb_key.json` not found** → place file in working dir or use absolute path.
## License


