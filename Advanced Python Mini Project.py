import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_csv("student_habits_performance.csv")
print(df.head())
print("Shape of data:", df.shape)
print("Columns:", df.columns)
print("Missing values:\n", df.isnull().sum())
df = df.dropna()
print("Missing values after dropna:\n", df.isnull().sum())
duplicates = df.duplicated()
print("Number of duplicate rows:", duplicates.sum())
z_scores = stats.zscore(df[['study_hours_per_day', 'sleep_hours', 'exam_score']])
outliers = (abs(z_scores) > 3).any(axis=1)
print("Number of outlier rows:", outliers.sum())
df = df[~outliers]
print("Shape after removing outliers:", df.shape)
z_scores_after = stats.zscore(df[['study_hours_per_day', 'sleep_hours', 'exam_score']])
outliers_after = (abs(z_scores_after) > 3).any(axis=1)
print("Number of outlier rows after removal:", outliers_after.sum())
print("Shape after removing outliers:", df.shape)
top_features = df.corr(numeric_only=True)['exam_score'].abs().nlargest(4).index
top_features = top_features.drop('exam_score')
print("Top 3 most correlated features with exam_score:", top_features.tolist())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
X = df[['study_hours_per_day', 'mental_health_rating', 'exercise_frequency']]
y = df['exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')

X_single = df[['study_hours_per_day']]
y = df['exam_score']

model = LinearRegression()
model.fit(X_single, y)

x_range = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
x_range = pd.DataFrame(x_range, columns=['study_hours_per_day'])
y_pred_line = model.predict(x_range)

plt.figure(figsize=(8, 5))
plt.scatter(X_single, y, color='blue', label='Actual data')
plt.plot(x_range, y_pred_line, color='red', label='Prediction line')
plt.title("Linear Regression: Study Hours vs Exam Score")
plt.xlabel("Study Hours per Day")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')

X_single = df[['mental_health_rating']]
y = df['exam_score']

model = LinearRegression()
model.fit(X_single, y)

x_range = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
x_range = pd.DataFrame(x_range, columns=['mental_health_rating'])
y_pred_line = model.predict(x_range)

plt.figure(figsize=(8, 5))
plt.scatter(X_single, y, color='blue', label='Actual data')
plt.plot(x_range, y_pred_line, color='red', label='Prediction line')
plt.title("Linear Regression: Mental Health Rating vs Exam Score")
plt.xlabel("Mental Health Rating")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')

X_single = df[['social_media_hours']]
y = df['exam_score']

model = LinearRegression()
model.fit(X_single, y)

x_range = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
x_range = pd.DataFrame(x_range, columns=['social_media_hours'])
y_pred_line = model.predict(x_range)

plt.figure(figsize=(8, 5))
plt.scatter(X_single, y, color='blue', label='Actual data')
plt.plot(x_range, y_pred_line, color='red', label='Prediction line')
plt.title("Linear Regression: Social Media Hours vs Exam Score")
plt.xlabel("Social Media Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

features = ['study_hours_per_day', 'mental_health_rating', 'social_media_hours']
results = []

for feature in features:
    X = df[[feature]]
    y = df['exam_score']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    results.append([feature, round(mae, 2), round(r2, 2)])

results_df = pd.DataFrame(results, columns=['Feature', 'MAE', 'R²'])
print(results_df)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

X = df[['study_hours_per_day', 'mental_health_rating', 'social_media_hours']]
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("R²:", round(r2, 2))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = df['study_hours_per_day']
y = df['social_media_hours']
z = df['exam_score']
color = df['mental_health_rating']

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=color, cmap='viridis', s=20, alpha=0.7)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
cbar.set_label('Mental Health Rating')

ax.set_xlabel('Study Hours per Day')
ax.set_ylabel('Social Media Hours')
ax.set_zlabel('Exam Score')
ax.set_title('3D Scatter: Study, Social Media, Exam Score (Color = Mental Health)')

plt.tight_layout()
plt.show()
