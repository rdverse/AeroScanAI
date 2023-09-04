import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_synthetic_data(num_samples, defect_probabilities):
  """
  Generate a dataframe with synthetically generated data.

  Args:
    num_rows: The number of rows in the dataframe.
    defect_probabilities: A list of three probabilities, one for each defect class.

  Returns:
    A Pandas dataframe with the synthetic data.
  """

  # Create a list of defect labels.
  defect_labels = [
      np.random.choice([0, 1, 2], p=defect_probabilities) for _ in range(num_samples)
  ]

  # Generate the features.
  backwall = np.random.randint(2, size=num_samples)
  frontwall = np.random.randint(2, size=num_samples)
  ramp = np.random.randint(2, size=num_samples)
  geometry = np.random.randint(2, size=num_samples)
  no_peaks = np.random.randint(1, 8, size=num_samples)
  noise = np.random.uniform(0, 0.7, size=num_samples)
  max = np.random.uniform(10, 17, size=num_samples)
  min = np.random.uniform(0, 3, size=num_samples)
  signal_noise_ratio = np.random.uniform(0, 3, size=num_samples)

  # Create the dataframe.
  df = pd.DataFrame({
      "backwall": backwall,
      "frontwall": frontwall,
      "ramp": ramp,
      "geometry": geometry,
      "no_peaks": no_peaks,
      "noise": noise,
      "max": max,
      "min": min,
      "signal_noise_ratio": signal_noise_ratio,
      "defect": defect_labels,
  })

  return df

import xgboost as xgb
from sklearn.metrics import accuracy_score
import daal4py as d4p
df = generate_synthetic_data(num_samples=1000, defect_probabilities=[0.6,0.25,0.15])
X_train, X_test, y_train, y_test = train_test_split(df.drop('defect', axis=1), df.defect, test_size=0.25)

xgb_train = xgb.DMatrix(X_train, label=np.array(y_train))

xgb_model = xgb.train(xgb_train, num_boost_round=20)
d4p_model = d4p.get_gbt_model_from_xgboost(xgb_model)

dt = d4p_model(X_train, label=y_train)
dt_test = d4p_model(X_test, label=y_test)
print("Scoring on test data")
print("Accuracy score: ", accuracy_score(y_test, np.argmax(d4p_model.predict(data=X_test), axis=1)))
print("Scoring on train data")
print("Accuracy score: ", accuracy_score(y_train, np.argmax(d4p_model.predict(data=X_train), axis=1)))