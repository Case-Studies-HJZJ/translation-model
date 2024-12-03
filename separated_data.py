import pandas as pd

# Define a function to classify sentences as formal or informal
def classify_formal_informal(data):
    formal_keywords = ["Sie", "Ihr", "Ihnen", "Ihrer"]  # Formal German indicators
    informal_keywords = ["du", "dein", "dir", "dich", "deiner"]  # Informal German indicators

    # Initialize lists for formal and informal sentences
    formal_data = []
    informal_data = []

    # Iterate through the rows to classify sentences
    for _, row in data.iterrows():
        english_sentence = row['English']
        german_sentence = row['German']

        # Check for formal or informal indicators
        if any(keyword in german_sentence for keyword in formal_keywords):
            formal_data.append((english_sentence, german_sentence))
        elif any(keyword in german_sentence for keyword in informal_keywords):
            informal_data.append((english_sentence, german_sentence))
        else:
            formal_data.append((english_sentence, german_sentence))
            informal_data.append((english_sentence, german_sentence))

    return formal_data, informal_data


# Load the TSV file into a DataFrame
data = pd.read_csv("Sentences.tsv", sep='\t', encoding='utf-8')

# Apply the classification
formal, informal = classify_formal_informal(data)

# Convert to DataFrame and save to CSV
formal_df = pd.DataFrame(formal, columns=["English", "German"])
informal_df = pd.DataFrame(informal, columns=["English", "German"])

# Save the results as CSV files
formal_df.to_csv("data/Formal_Sentences.csv", index=False, encoding='utf-8')
informal_df.to_csv("data/Informal_Sentences.csv", index=False, encoding='utf-8')