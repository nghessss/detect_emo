import pandas as pd

# Function to read content from a file
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Function to concatenate content of two files into a new csv file
def concat_files(content1, content2, file_name):
    try:
        df = pd.DataFrame({'sentence': content1, 'sentiment': content2})
        df.to_csv(file_name, index=False, encoding='utf-8')
    except Exception as e:
        print(f"Error writing to file: {e}")

# Read the content from two files
content1 = read_file('./sentence.txt')
content2 = read_file('./sentiments.txt')
# change data of content2 to negative, neutral, positive
content2 = [x.replace('0', 'negative') for x in content2]
content2 = [x.replace('1', 'neutral') for x in content2]
content2 = [x.replace('2', 'positive') for x in content2]
content2 = [x.replace('\n', '') for x in content2]
content1 = [x.replace('\n', '') for x in content1]
# Concatenate content and write to new csv file
if content1 is not None and content2 is not None:
    concat_files(content1, content2, 'concatenated.csv')
