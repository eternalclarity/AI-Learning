import pandas as pd
import numpy as np

x = {
    "m": ["male", "female", "male", "male"],
    "x": [1, 2, 3, 4],
    "y": [1, 2, np.nan, np.nan]
}

x = pd.DataFrame(x)

x_df = x.isnull().sum()
x_df = x_df.loc[x_df > 0]
print(x_df)

