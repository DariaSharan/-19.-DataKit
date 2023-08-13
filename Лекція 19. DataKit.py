# Task 1
# Numpy
#a. Create an array with shape (4, 3) of: a. all zeros b. ones c. numbers from 0 to 11
#b. Tabulate the following function: F(x)=2x^2+5, x∈[1,100] with step 1.
#c. Tabulate the following function: F(x)=e^−x, x∈[−10,10] with step 1.

import numpy as np

# Task a
array_zeros = np.zeros((4, 3))
array_ones = np.ones((4, 3))
array_numbers = np.arange(12).reshape(4, 3)

# Task b
def func_b(x):
    return 2 * x**2 + 5

x_values_b = np.arange(1, 101, step=1)
tabulated_values_b = func_b(x_values_b)

# Task c
def func_c(x):
    return np.exp(-x)

x_values_c = np.arange(-10, 11, step=1)
tabulated_values_c = func_c(x_values_c)

print("Task a:")
print("Array of zeros:\n", array_zeros)
print("Array of ones:\n", array_ones)
print("Array of numbers from 0 to 11:\n", array_numbers)

print("\nTask b:")
for x, y in zip(x_values_b, tabulated_values_b):
    print(f"F({x}) = {y}")

print("\nTask c:")
for x, y in zip(x_values_c, tabulated_values_c):
    print(f"F({x}) = {y}")



#Task 2
#Pandas
#a. Import the dataset from this [address] and assign it to df variable.
#b. Select only the Team, Yellow Cards and Red Cards columns.
#c. How many teams participated in the Euro2012?
#d. Filter teams that scored more than 6 goals

import pandas as pd

# Task a
url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv"
df = pd.read_csv(url)

# Task b
selected_columns = df[["Team", "Yellow Cards", "Red Cards"]]

# Task c
num_teams = df["Team"].nunique()

# Task d
filtered_teams = df[df["Goals"] > 6]

print("Task b:")
print(selected_columns)

print("\nTask c:")
print(f"Number of teams participated: {num_teams}")

print("\nTask d:")
print("Teams that scored more than 6 goals:")
print(filtered_teams[["Team", "Goals"]])

#Task 3. DataViz

#a. Choose a dataset, you can use Seaborn [example datasets]. Create a cheat sheet for yourself containing all plot types discussed in the lecture. Provide the following info:

#       - Plot type
#       - Use cases (categorical data, distribution, etc.)
#       - Example on the dataset

import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic = sns.load_dataset("titanic")

# Plot type: Bar Plot
sns.barplot(x="class", y="fare", data=titanic)

# Plot type: Count Plot
sns.countplot(x="sex", data=titanic)

# Plot type: Box Plot
sns.boxplot(x="class", y="fare", data=titanic)

# Plot type: Violin Plot
sns.violinplot(x="class", y="age", data=titanic)

# Plot type: Histogram
sns.histplot(data=titanic, x="age", kde=True)

# Plot type: Scatter Plot
sns.scatterplot(x="age", y="fare", data=titanic)

# Plot type: Pair Plot
sns.pairplot(data=titanic, hue="sex")

# Plot type: Heatmap (exclude non-numeric columns)
correlation_matrix = titanic.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(correlation_matrix, annot=True)

# Plot type: Line Plot
sns.lineplot(x="age", y="fare", data=titanic)

# Plot type: Strip Plot
sns.stripplot(x="class", y="age", data=titanic, jitter=True)

# Plot type: Swarm Plot
sns.swarmplot(x="class", y="age", data=titanic)

# Plot type: PairGrid
g = sns.PairGrid(titanic, vars=["age", "fare"], hue="sex")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()

# Show the plots
plt.show()