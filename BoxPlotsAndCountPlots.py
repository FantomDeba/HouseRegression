import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r'E:\placement\HouseRegression\processed_houses.csv')

# Select numerical columns
num_cols = df.select_dtypes(include='number').columns.tolist()

# Melt into long-form
df_long = df[num_cols].melt(var_name='feature', value_name='value')

# Use catplot to generate a grid of boxplots
g = sns.catplot(
    data=df_long,
    x='feature', y='value',
    kind='box',
    col='feature',
    col_wrap=8,        # 8 plots per row
    sharex=False,
    sharey=False,
    height=2,          # height of each subplot
    aspect=1
)

g.set_xticklabels(rotation=90, fontsize=6)
g.set_titles('{col_name}')
plt.tight_layout()
plt.show()
