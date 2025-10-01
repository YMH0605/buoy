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
from main import LSTM, Predict
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(num_classes=2, input_size=3, hidden_size=64, num_layers=3).to(device)
model.load_state_dict(torch.load('lstm_on_bouy.pth', map_location=device))

# 6 steps (~60 min) for pond 1
dataX, df_now, yhat, *_ = Predict(1, 6, model)
print(yhat)  # (n_ahead, 1, 3): [DO_mgL, Temp_C, hour_minute]
```

## Defaults

* `seq_length=4`, `n_future=1` (iterative rollout)
* timestep = 10 minutes
* timezone: UTC → America/Chicago
* date filter: index[:8] > `20250701`

## Notes

* Last 10 records are dropped before modeling to avoid partial packets.
* Standardization is fit on the train slice inside `Load_Data`.
* If you split utilities to `data_processing.py`, remove duplicates in `main.py`.

## Troubleshooting

* **`fb_key.json` not found** → place file in working dir or use absolute path.
* **Shape mismatch** → make sure model hyperparams at inference match training.
* **Empty data** → check date filter and `type=='a_buoy'`.

## License

Add your license here (e.g., MIT).
