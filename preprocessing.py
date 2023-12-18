import pandas as pd
import re

def remove_english_and_emojis(text):
    # Remove English words and emojis
    text = re.sub('[^\u0600-\u06FF\s\d]+', '', text)

    return text.strip()

# Load the CSV file into a DataFrame
input_file = 'test _no_label.csv'
output_file = 'cleaned_data.csv'

# Specify encoding as utf-8-sig when reading the CSV file
df = pd.read_csv(input_file, encoding='utf-8-sig')

# Apply the cleaning function to the 'review_description' column (replace with the actual column name)
df['review_description'] = df['review_description'].apply(remove_english_and_emojis)

# Create a list 'data' to store the cleaned Arabic sentences
data = df['review_description'].tolist()

# Save the cleaned data to a new CSV file with utf-8-sig encoding
df.to_csv(output_file, index=False, encoding='utf-8-sig')
