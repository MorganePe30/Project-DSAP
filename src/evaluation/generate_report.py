from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_CSV = RESULTS_DIR / "model_comparison.csv"
PLOT1 = RESULTS_DIR / "true_vs_predicted.png"
PLOT2 = RESULTS_DIR / "time_series_predictions.png"
OUT_HTML = RESULTS_DIR / "report.html"


def main():
    if not MODEL_CSV.exists():
        raise FileNotFoundError(f"Missing: {MODEL_CSV}. Run main.py first.")

    df = pd.read_csv(MODEL_CSV)
    df_sorted = df.sort_values("rmse_test").reset_index(drop=True)
    best = df_sorted.iloc[0]

    table_html = df_sorted.to_html(index=False, float_format=lambda x: f"{x:.3f}")

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Fed Policy ML — Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ margin-bottom: 0; }}
            .sub {{ color: #555; margin-top: 4px; }}
            .box {{ background: #f6f6f6; padding: 12px 16px; border-radius: 8px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background: #f0f0f0; }}
            img {{ max-width: 900px; width: 100%; border: 1px solid #ddd; border-radius: 8px; margin-top: 12px; }}
        </style>
    </head>
    <body>
        <h1>Fed Policy ML Project — Results</h1>
        <div class="sub">Automatically generated from results/ (run python main.py first)</div>

        <h2>Best model (lowest test RMSE)</h2>
        <div class="box">
            <b>{best["model"]}</b><br>
            Test RMSE: {best["rmse_test"]:.3f} | Test MAE: {best["mae_test"]:.3f} | Test R²: {best["r2_test"]:.3f}
        </div>

        <h2>Model comparison</h2>
        {table_html}

        <h2>Prediction plots</h2>
        <h3>True vs Predicted</h3>
        <img src="true_vs_predicted.png" alt="True vs Predicted">

        <h3>Time series predictions</h3>
        <img src="time_series_predictions.png" alt="Time series predictions">

    </body>
    </html>
    """

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[OK] Report generated: {OUT_HTML}")


if __name__ == "__main__":
    main()