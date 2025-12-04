# =========================
# Quantile Regression for Swiss Tax App (with Canton handling + manual input)
# - Baseline (50%) + 10% & 90% band
# - 25% & 75% quantile lines
# - Variables handled separately: income, canton (with canton-specific tax),
#   debt, marital_status
# - Target can be: 'federal_tax', 'cantonal_tax', or 'total_tax'
# - Optional ML: LightGBM quantile models
# - Manual user input supported
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Core quantile regression (linear, interpretable)
import statsmodels.api as sm
import patsy  # for formula w/ splines & categorical coding

# Optional ML quantile models (comment out if you don't use)
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# ---------------------------
# Canton multipliers (toy demo)
# These scale cantonal tax; replace with real multipliers/tariffs later.
# Lower value ~ lower cantonal burden (e.g., ZG), higher ~ higher burden (e.g., GE).
# ---------------------------
CANTON_MULTIPLIER = {
    "ZH": 1.00,  # Zürich
    "GE": 1.25,  # Genève
    "VD": 1.15,  # Vaud
    "BE": 1.05,  # Bern
    "SG": 0.95,  # St. Gallen
    "ZG": 0.80,  # Zug
}

# -------------------------------------------------------------------
# 0) LOAD YOUR DATA (replace this with your path)
#    Required columns (when using real data):
#    ['federal_tax','cantonal_tax','income','debt','marital_status','canton']
# -------------------------------------------------------------------

USE_SYNTHETIC = True  # set False when you have real data

if not USE_SYNTHETIC:
    df = pd.read_csv("your_tax_data.csv")
else:
    # ---- Synthetic data generator (so you can run end-to-end now) ----
    rng = np.random.default_rng(42)
    n = 1200

    # Separate variables
    marital = rng.choice(["single", "married"], size=n, p=[0.55, 0.45])
    canton = rng.choice(list(CANTON_MULTIPLIER.keys()), size=n, p=[0.28, 0.12, 0.14, 0.18, 0.18, 0.10])
    income = rng.normal(100_000, 35_000, size=n).clip(30_000, 350_000)
    debt = np.maximum(0, rng.normal(40_000, 30_000, size=n))

    # Canton-specific multiplier
    mult = np.array([CANTON_MULTIPLIER[c] for c in canton])

    # True (hidden) functions to simulate taxes
    # Federal progressive + mild debt effect + marital relief
    federal_tax = 0.028 * income + 5e-7 * income**2 + 0.00008 * debt
    federal_tax -= np.where(marital == "married", 2500, 0)

    # Cantonal progressive + debt effect, scaled by canton multiplier + marital relief
    cantonal_base = 0.055 * income + 8e-7 * income**2 + 0.00025 * debt
    cantonal_tax = cantonal_base * mult
    cantonal_tax -= np.where(marital == "married", 1500 * mult, 0)

    # Noise + a few outliers
    eps_f = rng.normal(0, 1600, size=n)
    eps_c = rng.normal(0, 2200, size=n)
    federal_tax = (federal_tax + eps_f).clip(0)
    cantonal_tax = (cantonal_tax + eps_c).clip(0)

    df = pd.DataFrame({
        "federal_tax": federal_tax,
        "cantonal_tax": cantonal_tax,
        "income": income,
        "debt": debt,
        "marital_status": marital,
        "canton": canton
    })

# Add total tax (handy default target)
df["total_tax"] = df["federal_tax"] + df["cantonal_tax"]

# -------------------------------------------------------------------
# 1) CONFIG
# -------------------------------------------------------------------
TARGET = "total_tax"       # 'federal_tax' | 'cantonal_tax' | 'total_tax'
MARITAL_TO_PLOT = "single" # which group to visualize in the curve plot
CANTON_TO_PLOT = "ZH"      # which canton to visualize in the curve plot
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# Filter to a group for clean curves (plot only; models train on full data)
df_plot = df.copy()
if MARITAL_TO_PLOT in df["marital_status"].unique():
    df_plot = df_plot[df_plot["marital_status"] == MARITAL_TO_PLOT]
if CANTON_TO_PLOT in df["canton"].unique():
    df_plot = df_plot[df_plot["canton"] == CANTON_TO_PLOT]

# -------------------------------------------------------------------
# 2) DESIGN MATRICES (statsmodels QuantReg)
#    Allow mild nonlinearity in income via B-splines.
#    Include BOTH categorical effects: marital_status AND canton.
# -------------------------------------------------------------------
# Formula: target ~ spline(income) + debt + marital_status + canton
formula = f"{TARGET} ~ bs(income, df=4) + debt + C(marital_status) + C(canton)"

# Build design matrices on the full dataset for estimation
y, X = patsy.dmatrices(formula, df, return_type='dataframe')

# -------------------------------------------------------------------
# 3) Fit Quantile Regressions (10, 25, 50, 75, 90)
# -------------------------------------------------------------------
models = {}
pred_curves = {}

# Make an income grid for smooth curves within the plotting group
x_grid = np.linspace(df_plot["income"].min(), df_plot["income"].max(), 200)
grid = pd.DataFrame({
    "income": x_grid,
    "debt": df_plot["debt"].median(),         # fix debt at median for the visual
    "marital_status": MARITAL_TO_PLOT,
    "canton": CANTON_TO_PLOT
})

# Design matrix for the grid (same structure as training)
_, X_grid = patsy.dmatrices(formula, grid, return_type='dataframe')

for q in QUANTILES:
    mod = sm.QuantReg(y, X)
    res = mod.fit(q=q)
    models[q] = res
    pred_curves[q] = res.predict(X_grid).values

# -------------------------------------------------------------------
# 4) Plot: baseline (50%), 10–90 band, plus 25% and 75% lines
# -------------------------------------------------------------------
plt.figure(figsize=(9, 5))

# Shaded band between 10% and 90%:
plt.fill_between(
    x_grid,
    pred_curves[0.10],
    pred_curves[0.90],
    alpha=0.2,
    label="10–90% band"
)

# 25% and 75% quantile lines:
plt.plot(x_grid, pred_curves[0.25], linestyle="--", label="25%")
plt.plot(x_grid, pred_curves[0.75], linestyle="--", label="75%")

# Baseline (median):
plt.plot(x_grid, pred_curves[0.50], linewidth=2, label="50% (median)")

# Scatter actual observations for the plotted marital+canton group:
plt.scatter(df_plot["income"], df_plot[TARGET], s=12, alpha=0.35, label="observed")

plt.title(
    f"Quantile regression of {TARGET.replace('_',' ')} vs income  "
    f"(marital={MARITAL_TO_PLOT}, canton={CANTON_TO_PLOT})"
)
plt.xlabel("Income")
plt.ylabel(TARGET.replace("_", " ").title())
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 5) OPTIONAL: Nonlinear ML quantile models (LightGBM)
#    Trains separate models for 10/25/50/75/90 quantiles.
#    Use when relationships are strongly non-linear or interactions are complex.
# -------------------------------------------------------------------
if HAS_LGBM:
    print("\nTraining LightGBM quantile models (optional ML layer)…")
    # Prepare features for LightGBM
    X_ml = df[["income", "debt"]].copy()
    X_ml = pd.concat([
        X_ml,
        pd.get_dummies(df["marital_status"], prefix="marital", drop_first=False),
        pd.get_dummies(df["canton"], prefix="canton", drop_first=False)
    ], axis=1)
    y_ml = df[TARGET].values

    # Build grid for predictions (consistent with earlier plot)
    Xg_ml = grid[["income", "debt"]].copy()
    Xg_ml = pd.concat([
        Xg_ml,
        pd.get_dummies(grid["marital_status"], prefix="marital", drop_first=False),
...         pd.get_dummies(grid["canton"], prefix="canton", drop_first=False)
...     ], axis=1)
...     # Ensure same columns order
...     Xg_ml = Xg_ml.reindex(columns=X_ml.columns, fill_value=0)
... 
...     preds_ml = {}
...     for q in QUANTILES:
...         m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=700, learning_rate=0.05)
...         m.fit(X_ml, y_ml)
...         preds_ml[q] = m.predict(Xg_ml)
... 
...     # Plot ML curves (overlay or as separate figure)
...     plt.figure(figsize=(9, 5))
...     plt.fill_between(x_grid, preds_ml[0.10], preds_ml[0.90], alpha=0.2, label="10–90% band (ML)")
...     plt.plot(x_grid, preds_ml[0.25], linestyle="--", label="25% (ML)")
...     plt.plot(x_grid, preds_ml[0.75], linestyle="--", label="75% (ML)")
...     plt.plot(x_grid, preds_ml[0.50], linewidth=2, label="50% (median, ML)")
...     plt.scatter(df_plot["income"], df_plot[TARGET], s=12, alpha=0.35, label="observed")
...     plt.title(
...         f"LightGBM Quantile Regression of {TARGET.replace('_',' ')} vs income  "
...         f"(marital={MARITAL_TO_PLOT}, canton={CANTON_TO_PLOT})"
...     )
...     plt.xlabel("Income")
...     plt.ylabel(TARGET.replace("_", " ").title())
...     plt.legend()
...     plt.tight_layout()
...     plt.show()
... 
... # -------------------------------------------------------------------
... # 6) Predict for a manual user input (income, canton, debt, marital_status)
... #    Returns a dict of quantiles and shows a small horizontal interval plot.
... # -------------------------------------------------------------------
... def predict_user_quantiles(user_row: dict, models_dict, formula_str):
...     """
...     user_row example:
...       {"income": 120000, "debt": 30000, "marital_status": "single", "canton": "ZH"}
    """
    # Build a one-row DF and design matrix with the SAME formula
    ux = pd.DataFrame([user_row])
    # Make sure categories exist (statsmodels handles unseen levels by dropping, so prefer valid values)
    _, UX = patsy.dmatrices(formula_str, ux, return_type='dataframe')
    out = {}
    for q, res in models_dict.items():
        out[q] = float(res.predict(UX)[0])
    return out

def plot_user_interval(user_row, preds):
    # Simple one-person interval visual (10–90 shaded; 25/50/75 lines)
    plt.figure(figsize=(6,4))
    plt.fill_between([0,1], preds[0.10], preds[0.90], alpha=0.2, label="10–90% band")
    plt.plot([0,1],[preds[0.25]]*2, linestyle="--", label="25%")
    plt.plot([0,1],[preds[0.75]]*2, linestyle="--", label="75%")
    plt.plot([0,1],[preds[0.50]]*2, linewidth=2, label="50% (median)")
    plt.scatter([0.5],[preds[0.50]], s=40, label="Your estimate")
    plt.xticks([])
    plt.ylabel(f"Predicted {TARGET.replace('_',' ')} (CHF)")
    title = f"Your personalized interval (marital={user_row['marital_status']}, canton={user_row['canton']})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------- Manual input section (uncomment to use as CLI) ---------
if __name__ == "__main__":
    # Ask user for each variable separately
    try:
        income_in = float(input("Income (CHF): ").strip())
        debt_in = float(input("Debt (CHF): ").strip())
        marital_in = input("Marital status [single/married]: ").strip().lower()
        canton_in = input(f"Canton code {list(CANTON_MULTIPLIER.keys())}: ").strip().upper()
        if canton_in not in CANTON_MULTIPLIER:
            raise ValueError("Unknown canton code. Please use one of: " + ", ".join(CANTON_MULTIPLIER.keys()))
        if marital_in not in ["single", "married"]:
            raise ValueError("Marital status must be 'single' or 'married'.")
    except Exception as e:
        print("Input error:", e)
        raise SystemExit(1)

    user_dict = {
        "income": income_in,
        "debt": debt_in,
        "marital_status": marital_in,
        "canton": canton_in
    }

    # Predict quantiles from the trained statsmodels QuantReg
    user_preds = predict_user_quantiles(user_dict, models, formula)
    print("\nYour quantile predictions (CHF):")
    for q in sorted(user_preds):
        print(f" q={q:.2f}: {user_preds[q]:,.0f}")

    # Show small interval visualization
    plot_user_interval(user_dict, user_preds)

# -------------------------------------------------------------------
# 7) Print a compact model summary (for the 50% quantile)
# -------------------------------------------------------------------
print("\nMedian (50%) quantile regression summary (statsmodels):")
print(models[0.50].summary())
