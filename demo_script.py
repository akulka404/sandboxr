# demo_script.py
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(100, 3), columns=list("ABC"))
print(df.describe().to_string())

