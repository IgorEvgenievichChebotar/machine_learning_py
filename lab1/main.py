import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dt = pd.read_csv('../real_estate.csv').dropna()
print(dt)

dt['Transact Date'].head(100).plot()
dt['Age'].head(100).plot()
dt['Distance To Transport'].head(100).plot()
dt['Shops'].head(100).plot()
dt['Latitude'].head(100).plot()
dt['Longitude'].head(100).plot()
dt['Price'].head(100).plot()

sns.lineplot(data=dt["Transact Date"].head(100))
sns.lineplot(data=dt["Age"].head(100))
sns.lineplot(data=dt["Distance To Transport"].head(100))
sns.lineplot(data=dt["Shops"].head(100))
sns.lineplot(data=dt["Latitude"].head(100))
sns.lineplot(data=dt["Longitude"].head(100))
sns.lineplot(data=dt["Price"].head(100))

plt.show()
