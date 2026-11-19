# Trains Greenway Home Energy's forest on 560 homes, measures it on the other 140, and explains it with your three
# functions from explain.py, printing what each step gave, so you can see them at work. Ready to use.
# Run it with: python report.py (it takes a little while: 200 trees, and each column shuffled 10 times)
import pandas as pd
from sklearn.metrics import mean_absolute_error

from bill_model import build_bill_model, split_homes
from explain import importance_table, plain_summary, top_features

homes = pd.read_csv("home_energy.csv")
X_train, X_test, y_train, y_test = split_homes(homes, 42)
print("1) homes:", len(homes))
print("2) training homes:", len(X_train))
print("3) test homes:", len(X_test))

model = build_bill_model(0).fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))
print("4) average miss on the test homes:", round(mae, 2))

table = importance_table(model, X_test, y_test, 0)
for number, row in enumerate(table.itertuples(), start=5):
    print(f"{number}) {row.feature}: {row.importance:.2f} (spread {row.spread:.2f})")

top = top_features(table, 3)
print("11) the top three:", top)
for number, sentence in enumerate(plain_summary(top, mae), start=12):
    print(f"{number}) {sentence}")
