# Car Price Predictor — Streamlit

## Files
- `app.py` — Streamlit interface and feature engineering
- `catboost_car_price_model.pkl` — your trained CatBoost model
- `requirements.txt` — Python dependencies

## Run locally
1. Keep `app.py`, `requirements.txt`, and `catboost_car_price_model.pkl` in the same folder.
2. Open a terminal in that folder and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

## Important model assumptions
- The app uses reference year `2024` for `car_age`, as specified in the supplied training notes.
- The prediction is converted back with `numpy.expm1`, assuming the training target was `log1p(price)`. If training used `log(price)`, change this to `numpy.exp(prediction_log)`.
- Engine fields that cannot be parsed use the supplied fallback values (3.0 liters, 300 HP, 6 transmission speeds). Replace these with training-set medians if those were used during training.
- Make sure the `LUXURY_BRANDS` set and categorical spellings match the original training pipeline for best results.
