import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#DATA LOAD
df = pd.read_csv("C:\\Users\\shubh\\Downloads\\dataset.csv")
df.columns = df.columns.str.strip()
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
df.head()
df.info()
print("Unique Countries:", df['COUNTRY'].nunique())
cols = ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION']
for col in cols:
    print(f"\n{col}:\n", df[col].value_counts())
    
#DATA CLEANING
    
month_cols = ['2025-M04','2025-M05','2025-M06','2025-M07','2025-M08',
              '2025-M09','2025-M10','2025-M11','2025-M12',
              '2026-M01','2026-M02','2026-M03']
# Filter CPI YOY data
yoy_df = df[
    (df['INDEX_TYPE'] == 'Consumer price index (CPI)') &
    (df['TYPE_OF_TRANSFORMATION'].str.contains('Year-over-year', na=False))
].copy()
print("YOY % Change rows:", len(yoy_df))
# Convert to numeric (important)
yoy_df[month_cols] = yoy_df[month_cols].apply(pd.to_numeric, errors='coerce')
# Create target
yoy_df['avg_yoy_inflation'] = yoy_df[month_cols].mean(axis=1)
# Remove empty rows
yoy_df = yoy_df.dropna(subset=['avg_yoy_inflation'])
print("After removing empty rows:", len(yoy_df))
yoy_df[['COUNTRY','COICOP_1999','avg_yoy_inflation'] + month_cols[:4]].head()

#Feature Scaling (MinMax Scaling)
scaler = MinMaxScaler()
num_cols = yoy_df.select_dtypes(include='number').columns
yoy_scaled = yoy_df.copy()
yoy_scaled[num_cols] = scaler.fit_transform(yoy_df[num_cols].fillna(0))
print("Before Scaling:")
print(yoy_df[num_cols].head())
print("\nAfter MinMax Scaling:")
print(yoy_scaled[num_cols].head())

#Scaling Visualization
plt.figure(figsize=(12,5))
for i, data in enumerate([yoy_df, yoy_scaled], 1):
    plt.subplot(1, 2, i)
    sns.histplot(data['avg_yoy_inflation'], kde=True)
    plt.title("Before Scaling" if i == 1 else "After MinMax Scaling")
plt.show()

plt.figure(figsize=(12,5))
for i, data in enumerate([yoy_df, yoy_scaled], 1):
    plt.subplot(1, 2, i)
    sns.boxplot(data=data[['avg_yoy_inflation']])
    plt.title("Before Scaling" if i == 1 else "After Scaling")
plt.show()

#Distribution of YOY Inflation Rates
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Histogram
axes[0].hist(yoy_df['avg_yoy_inflation'].dropna(), bins=40)
axes[0].set(title='Distribution of Average YOY CPI Inflation',
            xlabel='YOY % Change', ylabel='Frequency')

# Boxplot by COICOP
cat_data = yoy_df[yoy_df['COICOP_1999'] != 'All Items']
order = cat_data.groupby('COICOP_1999')['avg_yoy_inflation'].median().sort_values().index

sns.boxplot(data=cat_data, y='COICOP_1999', x='avg_yoy_inflation',
            order=order, ax=axes[1])

axes[1].set(title='Inflation Distribution by COICOP Category',
            xlabel='YOY % Change', ylabel='')

plt.tight_layout()
plt.show()

# Country-level inflation (All Items)
all_items = yoy_df[yoy_df['COICOP_1999'] == 'All Items']
country_avg = all_items.groupby('COUNTRY')['avg_yoy_inflation'].mean().sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, data in enumerate([country_avg.head(15), country_avg.tail(15)]):
    data.plot(kind='barh', ax=axes[i])
    axes[i].set(
        title='Top 15 Countries — Highest Avg CPI Inflation' if i == 0 
              else 'Bottom 15 Countries — Lowest Avg CPI Inflation',
        xlabel='Avg YOY % Change'
    )
    axes[i].invert_yaxis()

plt.tight_layout()
plt.show()
print("Highest inflation country:", country_avg.idxmax())
print("Lowest inflation country:", country_avg.idxmin())

#4Category-wise Inflation — Global Average
cat_avg = yoy_df[yoy_df['COICOP_1999'] != 'All Items'] \
    .groupby('COICOP_1999')['avg_yoy_inflation'] \
    .mean().sort_values()

plt.figure(figsize=(12, 6))
cat_avg.plot(kind='barh',
             color=['tomato' if v > 0 else 'steelblue' for v in cat_avg])

plt.axvline(0, linestyle='--')
plt.title('Global Average YOY Inflation by COICOP Category')
plt.xlabel('Avg YOY % Change')
plt.tight_layout()
plt.show()
print("Most inflationary category:", cat_avg.idxmax())
print("Least inflationary category:", cat_avg.idxmin())

#Monthly Trend — Top 5 High-Inflation Countries
top5 = country_avg.head(5).index

trend = all_items[all_items['COUNTRY'].isin(top5)][['COUNTRY'] + month_cols] \
    .melt(id_vars='COUNTRY', var_name='Month', value_name='YOY_Inflation')

plt.figure(figsize=(14, 6))
sns.lineplot(data=trend, x='Month', y='YOY_Inflation', hue='COUNTRY', marker='o')

plt.title('Monthly CPI Trend — Top 5 High Inflation Countries')
plt.xlabel('Month')
plt.ylabel('YOY % Change')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.show()

#Correlation Heatmap — Monthly CPI Values
plt.figure(figsize=(12, 6))

sns.heatmap(yoy_df[month_cols].corr(),
            annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

plt.title('Correlation Between Monthly CPI Readings')
plt.tight_layout()
plt.show()

#Pairplot — Selected Months
sample_months = ['2025-M04','2025-M07','2025-M10','2026-M01']

sns.pairplot(yoy_df[sample_months].dropna(),
             diag_kind='kde', plot_kws={'alpha':0.4})

plt.suptitle('Pairplot — Quarterly CPI Snapshots', y=1.02)
plt.show()

#Outlier Detection — Boxplot of Monthly Readings
plt.figure(figsize=(14, 5))

sns.boxplot(data=yoy_df[month_cols])
plt.xticks(rotation=45)

plt.title('Outlier Detection — Monthly CPI YOY % Values')
plt.ylabel('YOY % Change')

plt.tight_layout()
plt.show()

#Feature Engineering for ML
ml_df = yoy_df.dropna(subset=month_cols, thresh=6).copy()

# Features and target
feature_months = ['2025-M04','2025-M05','2025-M06','2025-M07','2025-M08','2025-M09']
target = 'avg_yoy_inflation'

# One-Hot Encoding (no LabelEncoder needed)
ml_df = pd.get_dummies(ml_df, columns=['COICOP_1999', 'COUNTRY'], drop_first=True)

X = ml_df[feature_months + [col for col in ml_df.columns if 'COICOP_1999_' in col or 'COUNTRY_' in col]]
y = ml_df[target]

# Create X and y
X = ml_df[feature_months + [col for col in ml_df.columns if 'COICOP_1999_' in col or 'COUNTRY_' in col]]
y = ml_df[target]

print("Final ML dataset shape:", X.shape)
print("Features:", list(X.columns))

#Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Model trained!")

#Model Evaluation
def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("MAE :", round(mean_absolute_error(y_true, y_pred), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 4))
    print("R²  :", round(r2_score(y_true, y_pred), 4))

# Evaluate Linear Regression
lr_pred = evaluate("Linear Regression", y_test, lr.predict(X_test))

#Actual vs Predicted 
y_test_arr = np.array(y_test)
lr_pred_arr = np.array(lr.predict(X_test))

plt.figure(figsize=(7, 6))

sns.scatterplot(x=y_test_arr, y=lr_pred_arr)

# Perfect prediction line
plt.plot([y_test_arr.min(), y_test_arr.max()],
         [y_test_arr.min(), y_test_arr.max()],
         'r--')

plt.xlabel('Actual CPI YOY %')
plt.ylabel('Predicted CPI YOY %')
plt.title('Actual vs Predicted — Linear Regression')
plt.tight_layout()
plt.show()

#Feature Importance
importance = pd.Series(lr.coef_, index=X.columns).sort_values()

plt.figure(figsize=(10, 5))
importance.plot(kind='barh')

plt.title('Feature Importance (Linear Regression Coefficients)')
plt.xlabel('Coefficient Value')

plt.tight_layout()
plt.show()

#Model Comparison — R² Score
lr_pred = lr.predict(X_test)

# Calculate R² score
r2 = r2_score(y_test, lr_pred)

# Plot
plt.figure(figsize=(6, 4))
bar = plt.bar(['Linear Regression'], [r2])

# Add value label
plt.text(bar[0].get_x() + bar[0].get_width()/2, r2,
         f'{r2:.3f}', ha='center')

plt.ylim(0, 1.1)
plt.title('Model Performance (R² Score)')
plt.ylabel('R² Score')
plt.tight_layout()
plt.show()

#Key Insights & Conclusions
lr_pred = lr.predict(X_test)

print("=" * 55)
print("     CONSUMER PRICE INDEX — KEY FINDINGS")
print("=" * 55)

print("\nTotal countries analyzed :", yoy_df["COUNTRY"].nunique())
print("Total COICOP categories  :", yoy_df["COICOP_1999"].nunique())

print("\n Highest average inflation:",
      country_avg.idxmax(), f"({country_avg.max():.2f}%)")

print(" Lowest average inflation:",
      country_avg.idxmin(), f"({country_avg.min():.2f}%)")

print("\nMost inflationary category:",
      cat_avg.idxmax(), f"({cat_avg.max():.2f}%)")

print("Least inflationary category:",
      cat_avg.idxmin(), f"({cat_avg.min():.2f}%)")

#FIXED LINE
print(f"\nBest Model R² Score: Linear Regression ({r2_score(y_test, lr_pred):.4f})")

print("\nLinear Regression used for prediction.")
print("=" * 55)
