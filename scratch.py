#%%

import dataget
import plotly.express as px
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np

# %%

df = dataget.image.udacity_simulator().get()

# %%
px.histogram(df, x="steering").show()

