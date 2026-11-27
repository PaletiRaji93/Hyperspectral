
import pandas as pd
from pathlib import Path

output_dir = Path("outputs/3dcnn/group_cv")

files = [
    "oof_pot_predictions.csv",
    "oof_leaf_predictions.csv",
    "fold_metrics.csv",
    "training_history.csv",
]

for filename in files:
    file_path = output_dir / filename

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    df = pd.read_csv(file_path)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    if filename == "oof_pot_predictions.csv":
        print("\nIncorrectly classified pots:")
        if "True_Label" in df.columns and "Predicted_Label" in df.columns:
            incorrect = df[df["True_Label"] != df["Predicted_Label"]]
            print(incorrect.to_string(index=False))
        else:
            print("Check the column names above for actual and predicted labels.")

print("\nInspection completed.")
