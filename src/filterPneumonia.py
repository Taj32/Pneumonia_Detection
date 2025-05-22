import pandas as pd

df = pd.read_csv('data/Raw/Data_Entry_2017.csv')

# print out columns
print(df.columns)


# Keep rows wehre 'pneumonia is in the labels
pneumonia_df = df[df['Finding Labels'].str.contains('Pneumonia')]

# Add a binary label column (1 = pneumonia, 0 = normal)
normal_df = df[df['Finding Labels'] == 'No Finding']
normal_df['Pneumonia Label'] = 0
pneumonia_df['Pneumonia Label'] = 1

# Combine the pneumonia and normal data
final_df = pd.concat([pneumonia_df, normal_df])

# Shuffle the dataset
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save this to use later in your pipeline
final_df.to_csv('data/processed/pneumonia_vs_normal.csv', index=False)