import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.stdout.flush()

print("Starting Level 2 script at", pd.Timestamp.now())
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

print("\n=== Task 1: Table Booking and Online Delivery ===")

table_booking_pct = (df['Has Table booking'].value_counts(normalize=True) * 100).round(2)
print("\nPercentage of Restaurants with Table Booking:")
print(table_booking_pct)

online_delivery_pct = (df['Has Online delivery'].value_counts(normalize=True) * 100).round(2)
print("\nPercentage of Restaurants with Online Delivery:")
print(online_delivery_pct)

table_booking_ratings = df.groupby('Has Table booking')['Aggregate rating'].mean().round(2)
print("\nAverage Rating by Table Booking:")
print(table_booking_ratings)

fig1 = plt.figure(figsize=(6, 4))
table_booking_ratings.plot(kind='bar')
plt.title('Average Rating by Table Booking')
plt.xlabel('Has Table Booking')
plt.ylabel('Average Rating')
plt.xticks(rotation=0)
plt.savefig('table_booking_ratings.png')
plots = [fig1]

delivery_by_price = df.groupby('Price range')['Has Online delivery'].value_counts(normalize=True).unstack() * 100
print("\nPercentage of Online Delivery by Price Range:")
print(delivery_by_price.round(2))

fig2 = plt.figure(figsize=(8, 5))
delivery_by_price.plot(kind='bar', stacked=True)
plt.title('Online Delivery Availability by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.legend(title='Has Online Delivery')
plt.savefig('delivery_by_price.png')
plots.append(fig2)


print("\n=== Task 2: Price Range Analysis ===")


print("\nPrice Range Distribution:")
print(df['Price range'].value_counts().sort_index())

fig3 = plt.figure(figsize=(6, 4))
df['Price range'].value_counts().sort_index().plot(kind='bar')
plt.title('Price Range Distribution')
plt.xlabel('Price Range')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=0)
plt.savefig('price_range_dist.png')
plots.append(fig3)

avg_rating_by_price = df.groupby('Price range')['Aggregate rating'].mean().round(2)
print("\nAverage Rating by Price Range:")
print(avg_rating_by_price)

fig4 = plt.figure(figsize=(6, 4))
avg_rating_by_price.plot(kind='bar')
plt.title('Average Rating by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Average Rating')
plt.xticks(rotation=0)
plt.savefig('avg_rating_by_price.png')
plots.append(fig4)

rating_color_by_price = df.groupby(['Price range', 'Rating color'])['Aggregate rating'].mean().unstack().round(2)
print("\nAverage Rating by Price Range and Rating Color:")
print(rating_color_by_price)

for pr in rating_color_by_price.index:
    max_color = rating_color_by_price.loc[pr].idxmax()
    max_rating = rating_color_by_price.loc[pr].max()
    print(f"Price Range {pr}: Highest rated color is {max_color} with average rating {max_rating}")

fig5 = plt.figure(figsize=(10, 6))
sns.countplot(x='Price range', hue='Rating color', data=df)
plt.title('Rating Color Distribution by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Number of Restaurants')
plt.legend(title='Rating Color')
plt.savefig('rating_color_by_price.png')
plots.append(fig5)

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