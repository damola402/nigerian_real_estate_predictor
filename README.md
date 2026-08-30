# Nigerian Real Estate Price Prediction Using Machine Learning

A Python/Streamlit app that cleans a real Abuja property-listing dataset,
analyzes it, engineers a new feature, trains and compares two ML models,
and predicts sale prices through a simple GUI.

## Project structure

```
project/
├── data/
│   ├── raw_data.csv          # original scraped PropertyPro Abuja listings
│   └── cleaned_data.csv      # produced by 1_data_cleaning.py
├── charts/                   # PNG charts produced by 2_eda.py
├── model/
│   ├── price_model.pkl       # trained model + encoders, produced by 3_train_model.py
│   └── model_comparison.csv  # MAE/RMSE/R2 comparison table
├── 1_data_cleaning.py        # Step 1: cleaning + feature engineering
├── 2_eda.py                  # Step 2: 5 analytical questions + charts
├── 3_train_model.py          # Step 3: train & compare Linear Regression vs Decision Tree
├── app.py                    # Step 4: Streamlit GUI (prediction + analysis tabs)
└── requirements.txt
```

## How to run it

```bash
pip install -r requirements.txt

# Run these once, in order, to regenerate the cleaned data / charts / model:
python 1_data_cleaning.py
python 2_eda.py
python 3_train_model.py

# Then launch the app:
streamlit run app.py
```

The repo already includes the generated `cleaned_data.csv`, charts, and
`price_model.pkl`, so you can skip straight to `streamlit run app.py` if you
just want to see the GUI — but re-run the pipeline yourself before your
demo so you're comfortable explaining each step.

## Dataset

481 raw property listings scraped from PropertyPro for Abuja, covering both
sale and rent listings across 16 areas (Maitama, Asokoro, Wuse-2, Gwarinpa,
etc.) and 7 property types. Columns: title, price, listing type, area,
neighbourhood, address, bedrooms, bathrooms, toilets, property type, plus
metadata (id, dates, source URL).

**Important scope decision:** this project keeps SALE listings only (231 of
481 rows). Sale prices (hundreds of millions of naira) and rent prices
(annual rent, much lower) aren't comparable, so mixing them would make
"price" meaningless as a single target variable.

## Cleaning decisions (for your defense)

Each decision is also commented directly above the relevant code in
`1_data_cleaning.py`:

1. **Dropped columns** not useful for prediction: `description` (boilerplate
   safety text, not real listing text), `pid`, `source_url`, `date_added`,
   `last_updated`, `address`.
2. **Filtered to sale-only listings** (see above) — 231 rows.
3. **Removed 2 duplicate rows** (same title/area/price/bedrooms).
4. **Imputed missing bathrooms/toilets** using the median for that specific
   bedroom count, rather than one global median, so imputed values stay
   realistic for the size of property.
5. **Fixed data types**: bathrooms/toilets converted to whole numbers.
6. **Standardised text categories**: trimmed spaces and consistent
   capitalisation on `area`, `neighbourhood`, `property_type` so the same
   place/type isn't split into multiple categories.
7. **Merged the single `Mansion` row into `Detached Duplex`** — one
   observation can't be learned from and risks landing entirely in only the
   train or test split.
8. **Removed 11 bulk/multi-unit listings** (e.g. "33 Units Of Spacious
   Apartment" priced at ₦3.5B for the whole building) — these price an
   entire block, not one property, and would badly distort a single-property
   predictor if left in. This was found by inspecting outlier rows, not
   assumed in advance — a good talking point for your defense.
9. **Outlier trimming on price**: dropped listings below ₦1,000,000 (likely
   data entry errors) and above the 99th percentile (extreme luxury
   mansions that would distort a simple model).

Final cleaned dataset: **215 rows**.

## Engineered feature

No square-metre / property-size field exists in the source data, so instead
of price-per-square-metre we engineered:

- `total_rooms` = bedrooms + bathrooms + toilets
- `price_per_room` = price ÷ total_rooms

This gives a normalised "cost per room" figure for comparing affordability
across property types and locations regardless of exact size.

## Analytical questions answered (`2_eda.py`)

1. Which property type has the highest average price? → **Detached Duplex**
2. Which area has the highest average price? → **Wuse-2**
3. Does bedroom count affect price? → Yes, moderate positive correlation (~0.68)
4. Does total room count affect price? → Yes, moderate positive correlation (~0.65)
5. Which property type is most affordable per room? → **Bungalow**

## Model comparison

| Model             | MAE     | RMSE    | R²   |
|-------------------|---------|---------|------|
| Linear Regression | ~₦195M  | ~₦356M  | 0.57 |
| Decision Tree      | ~₦210M  | ~₦384M  | 0.49 |

**Linear Regression was selected** — better R² (explains more price
variance) and lower error on this dataset size. Features used: area,
property type, bedrooms, bathrooms, toilets (all encoded/numeric).

Talking point for your defense: with only 215 rows and no size/amenity data,
R² of ~0.57 is a reasonable, honest result — not inflated, not hidden.
Location and property type carry real predictive signal, but a lot of price
variation (finishing quality, exact plot, negotiation) simply isn't captured
by this dataset.

