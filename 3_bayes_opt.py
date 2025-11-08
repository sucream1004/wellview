import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from scipy.stats import norm


# ----------------------------------------------------------------------
# 1. Utility: model evaluation with cross-validation
# ----------------------------------------------------------------------
def evaluate_model(model, X, y, cv):
    y_pred = cross_val_predict(model, X, y, cv=cv)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return mae, rmse, r2


# ----------------------------------------------------------------------
# 2. Step 1 – Compare surrogate models (GPR / RF / XGBoost)
#    using grid_result.csv
# ----------------------------------------------------------------------
def compare_surrogates(grid_csv_path: str):
    # Load data
    data = pd.read_csv(grid_csv_path)
    X = data[['x', 'y', 'w', 'h']].values
    y = data['vq'].values

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # (1) Gaussian Process Regression (GPR)
    kernel = (C(1.0, (1e-3, 1e3)) *
              Matern(length_scale=1.0, nu=2.5) +
              WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=10,
                                   normalize_y=True)
    model_gpr = TransformedTargetRegressor(regressor=gpr,
                                           transformer=StandardScaler())

    # (2) Random Forest Regression (RF)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf = TransformedTargetRegressor(regressor=rf,
                                          transformer=StandardScaler())

    # (3) XGBoost Regression
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        random_state=42,
        objective='reg:squarederror'
    )
    model_xgb = TransformedTargetRegressor(regressor=xgb_model,
                                           transformer=StandardScaler())

    # Evaluate
    mae_gpr, rmse_gpr, r2_gpr = evaluate_model(model_gpr, X, y, kf)
    mae_rf, rmse_rf, r2_rf = evaluate_model(model_rf, X, y, kf)
    mae_xgb, rmse_xgb, r2_xgb = evaluate_model(model_xgb, X, y, kf)

    print("\n=== Surrogate Model Comparison (grid_result.csv) ===")
    print("Gaussian Process Regression:")
    print(f"  MAE:  {mae_gpr:.4f}")
    print(f"  RMSE: {rmse_gpr:.4f}")
    print(f"  R²:   {r2_gpr:.4f}")

    print("\nRandom Forest Regression:")
    print(f"  MAE:  {mae_rf:.4f}")
    print(f"  RMSE: {rmse_rf:.4f}")
    print(f"  R²:   {r2_rf:.4f}")

    print("\nXGBoost Regression:")
    print(f"  MAE:  {mae_xgb:.4f}")
    print(f"  RMSE: {rmse_xgb:.4f}")
    print(f"  R²:   {r2_xgb:.4f}")



# ----------------------------------------------------------------------
# 3. Expected Improvement (EI) acquisition function
# ----------------------------------------------------------------------
def expected_improvement(X, model, Y_sample, xi=0.01):
    mu, sigma = model.predict(X, return_std=True)
    sigma = sigma.reshape(-1)

    Y_best = np.max(Y_sample)  # Maximization problem
    improvement = mu - Y_best - xi

    with np.errstate(divide='warn'):
        Z = np.where(sigma > 0, improvement / sigma, 0.0)
        ei = np.where(
            sigma > 0,
            improvement * norm.cdf(Z) + sigma * norm.pdf(Z),
            0.0
        )
    return ei, mu, sigma


# ----------------------------------------------------------------------
# 4. Step 2 – Fit GP on aggregated vq and select top-100 EI candidates
#    using results_200k.csv
# ----------------------------------------------------------------------
def select_top_ei_candidates(bo_csv_path: str,
                             num_candidates: int = 5000,
                             top_k: int = 100,
                             output_csv: str = "top100_candidates.csv"):

    df = pd.read_csv(bo_csv_path)

    if "grid_type" in df.columns and "vq" in df.columns and \
       not {"x", "y", "w", "h"}.issubset(df.columns):

        df = df.groupby("grid_type")["vq"].sum().reset_index()

        xy_str = (df["grid_type"].str.split(")").str[0] + ")")
        wh_str = (df["grid_type"].str.split(")").str[1] + ")")

        df["x"] = xy_str.apply(lambda s: eval(s))[0]
        df["y"] = xy_str.apply(lambda s: eval(s))[2]
        df["w"] = wh_str.apply(lambda s: eval(s))[0]
        df["h"] = wh_str.apply(lambda s: eval(s))[2]

    X_data = df[['x', 'y', 'w', 'h']].values
    Y_data = df['vq'].values

    # Define GP surrogate with Matern kernel
    kernel = (C(1.0, (1e-3, 1e3)) *
              Matern(length_scale=1.0, nu=2.5) +
              WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=10,
                                   normalize_y=True)
    gpr.fit(X_data, Y_data)

    # Parameter bounds from data range
    pbounds = {
        'x': (np.min(X_data[:, 0]), np.max(X_data[:, 0])),
        'y': (np.min(X_data[:, 1]), np.max(X_data[:, 1])),
        'w': (np.min(X_data[:, 2]), np.max(X_data[:, 2])),
        'h': (np.min(X_data[:, 3]), np.max(X_data[:, 3]))
    }

    # Sample random candidates within the bounds
    x_cand = np.random.uniform(*pbounds['x'], num_candidates)
    y_cand = np.random.uniform(*pbounds['y'], num_candidates)
    w_cand = np.random.uniform(*pbounds['w'], num_candidates)
    h_cand = np.random.uniform(*pbounds['h'], num_candidates)
    X_candidates = np.column_stack((x_cand, y_cand, w_cand, h_cand))

    # Compute EI for each candidate
    ei_values, mu_candidates, sigma_candidates = expected_improvement(
        X_candidates, gpr, Y_data, xi=0.01
    )

    # Select top-K candidates by EI
    top_idx = np.argsort(ei_values)[-top_k:][::-1]
    top_points = X_candidates[top_idx]
    top_ei = ei_values[top_idx]
    top_mu = mu_candidates[top_idx]

    # Create DataFrame for export
    df_topk = pd.DataFrame(top_points, columns=['x', 'y', 'w', 'h'])
    df_topk["ei"] = top_ei
    df_topk["mu_pred"] = top_mu

    df_topk.to_csv(output_csv, index=False)
    print(f"Top {top_k} EI candidates saved to '{output_csv}'.")

    # Optional: print top-5
    print("\nTop 5 candidate parameter sets based on Expected Improvement:")
    for i in range(min(5, top_k)):
        print(
            f"Rank {i+1}: x={top_points[i,0]:.4f}, "
            f"y={top_points[i,1]:.4f}, w={top_points[i,2]:.4f}, "
            f"h={top_points[i,3]:.4f}, EI={top_ei[i]:.4f}, "
            f"Predicted vq (mu)={top_mu[i]:.4f}"
        )


# ----------------------------------------------------------------------
# 5. Main script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    compare_surrogates("grid_result.csv")

    select_top_ei_candidates(
        bo_csv_path="result_200k.csv",
        num_candidates=5000,
        top_k=100,
        output_csv="top100_candidates.csv"
    )
