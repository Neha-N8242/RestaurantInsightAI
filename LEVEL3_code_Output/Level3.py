import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

sys.stdout.flush()

print("Starting Level 3 script at", pd.Timestamp.now())
print("Python Version:", sys.version)
print("Pandas Version:", pd.__version__)
print("Matplotlib Version:", matplotlib.__version__)
print("Seaborn Version:", sns.__version__)
print("Matplotlib Backend:", matplotlib.get_backend())
print("Current Working Directory:", os.getcwd())

# Dataset path
dataset_path = r"C:\Users\nehan\Downloads\Dataset.csv"
print(f"Checking for dataset at: {dataset_path}")

if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found. Please verify the file exists.")
    sys.exit(1)
print(f"Found {dataset_path}")

try:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully!")
    print("First few rows:")
    print(df.head(3))
    print("Columns in dataset:", df.columns.tolist())  
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    sys.exit(1)

df['Aggregate rating'] = df['Aggregate rating'].fillna(df['Aggregate rating'].median())

print("\n=== Task 1: Predictive Modeling ===")

# Feature selection and encoding
features = ['Price range', 'Votes', 'Has Table booking', 'Has Online delivery']
if 'Cuisines' in df.columns:
    features.append('Cuisines')
df = df[features + ['Aggregate rating']].dropna()
le = LabelEncoder()
for col in ['Has Table booking', 'Has Online delivery', 'Cuisines']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

X = df[features]
y = df['Aggregate rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    print(f"\n{name}:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

best_model_name = min(results, key=lambda x: results[x]['MSE'])
print(f"\nBest Model: {best_model_name} with MSE {results[best_model_name]['MSE']:.2f}")

best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
fig1 = plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_best, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Actual vs Predicted Ratings ({best_model_name})')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.savefig('actual_vs_predicted.png')
plots = [fig1]

print("\n=== Task 2: Customer Preference Analysis ===")

if 'Cuisines' in df.columns:
    cuisine_ratings = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
    print("\nAverage Rating by Cuisine (Top 10):")
    print(cuisine_ratings.head(10))
else:
    print("\nWarning: 'Cuisines' column not found. Skipping cuisine rating analysis.")

if 'Cuisines' in df.columns and 'Votes' in df.columns:
    cuisines_by_votes = df.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False)
    print("\nMost Popular Cuisines by Total Votes (Top 5):")
    print(cuisines_by_votes.head())
    
    fig2 = plt.figure(figsize=(8, 5))
    cuisines_by_votes.head(5).plot(kind='bar')
    plt.title('Top 5 Cuisines by Total Votes')
    plt.xlabel('Cuisine')
    plt.ylabel('Total Votes')
    plt.xticks(rotation=45)
    plt.savefig('top_cuisines_by_votes.png')
    plots.append(fig2)
else:
    print("Warning: 'Cuisines' or 'Votes' column not found. Skipping cuisine popularity plot.")

if 'Cuisines' in df.columns:
    top_cuisines = cuisines_by_votes.head(5).index
    fig3 = plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cuisines', y='Aggregate rating', data=df[df['Cuisines'].isin(top_cuisines)])
    plt.title('Rating Distribution for Top 5 Cuisines')
    plt.xlabel('Cuisine')
    plt.ylabel('Aggregate Rating')
    plt.xticks(rotation=45)
    plt.savefig('rating_distribution_by_cuisine.png')
    plots.append(fig3)
else:
    print("Warning: 'Cuisines' column not found. Skipping cuisine rating comparison.")

current_plot = 0
while True:
    if current_plot < len(plots):
        plt.figure(plots[current_plot].number)
        plt.show()
        print(f"\nDisplaying Plot {current_plot + 1} of {len(plots)}")
        print("Press 'n' for Next, 'p' for Previous, 'q' to Quit")
        
        if plt.waitforbuttonpress(10):
            key = plt.gcf().canvas.manager.key_press_handler_id
            if key == ord('q'): 
                print("Exiting plot viewer.")
                break
            elif key == ord('n') and current_plot < len(plots) - 1: 
                current_plot += 1
            elif key == ord('p') and current_plot > 0: 
                current_plot -= 1
        
        plt.close() 
    else:
        print("No plots to display.")
        break

print("Script completed. Check saved plots in:", os.getcwd())