from PIL import Image

files = [
    "models/LinearRegression_feature_importance.png",
    "models/LinearRegression_actual_vs_pred.png",
    "models/DecisionTree_actual_vs_pred.png",
    "models/DecisionTree_feature_importance.png",
    "models/RandomForest_actual_vs_pred.png",
    "models/RandomForest_feature_importance.png",
]

for f in files:
    try:
        im = Image.open(f)
        print(f"OK\t{f}\t{im.size}\t{im.mode}\t{im.format}")
    except Exception as e:
        print(f"ERR\t{f}\t{e}")
