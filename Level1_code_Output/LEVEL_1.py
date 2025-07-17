import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.stdout.flush()

print("Starting Level 1 script at", pd.Timestamp.now())
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
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    sys.exit(1)

df['Cuisines'] = df['Cuisines'].fillna('Unknown')
df['Aggregate rating'] = df['Aggregate rating'].fillna(df['Aggregate rating'].median())

sns.set(style="whitegrid")

print("\n=== Task 1: Data Exploration and Preprocessing ===")
print("Rows and Columns:", df.shape)
print("\nMissing Values:")
print(df[['Restaurant Name', 'City', 'Cuisines', 'Aggregate rating', 'Votes']].isnull().sum())
print("\nAggregate Rating Stats:")
print(df['Aggregate rating'].describe())


plots = []

fig1 = plt.figure(figsize=(8, 5))
sns.histplot(df['Aggregate rating'], bins=20)
plt.title('Distribution of Aggregate Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('rating_histogram.png')
plots.append(fig1)

print("\n=== Task 2: Descriptive Analysis ===")
print("\nStats for Cost, Votes, and Rating:")
print(df[['Average Cost for two', 'Votes', 'Aggregate rating']].describe())
print("\nTop 5 Cities:")
print(df['City'].value_counts().head(5))

fig2 = plt.figure(figsize=(8, 5))
df['City'].value_counts().head(5).plot(kind='bar')
plt.title('Top 5 Cities by Restaurant Count')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.savefig('top_cities.png')
plots.append(fig2)

cuisines = df['Cuisines'].str.split(', ', expand=True).stack().value_counts()
print("\nTop 5 Cuisines:")
print(cuisines.head(5))
fig3 = plt.figure(figsize=(8, 5))
cuisines.head(5).plot(kind='bar')
plt.title('Top 5 Cuisines by Restaurant Count')
plt.xlabel('Cuisine')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.savefig('top_cuisines.png')
plots.append(fig3)

print("\n=== Task 3: Geospatial Analysis ===")
print("\nTop 5 Cities by Restaurant Count:")
print(df['City'].value_counts().head(5))

fig4 = plt.figure(figsize=(10, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='Aggregate rating', data=df)
plt.title('Restaurant Locations by Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('location_scatter.png')
plots.append(fig4)

current_plot = 0
while True:
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

print("Script completed. Check saved plots in:", os.getcwd())