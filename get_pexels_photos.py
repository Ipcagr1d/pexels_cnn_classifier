import os
import csv
import requests

# set up search criteria
search_terms = ['lemon', 'lychee', 'grape', 'kiwi', 'apple'] # keywords for search

# loop through each search term
for term in search_terms:
    # read CSV file to get image URLs and image IDs
    csv_file = os.path.join(f'{term}_images', f'{term}_urls.csv')
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header row
        for row in reader:
            image_id = row[0]
            image_url = row[1]
            response = requests.get(image_url)
            file_name = os.path.basename(image_url)
            file_ext = os.path.splitext(file_name)[1]
            file_name = f'{image_id}{file_ext}'
            file_path = os.path.join(f'{term}_images', file_name)
            with open(file_path, 'wb') as f:
                f.write(response.content)
                print(f'Saved {file_name} to {term}_images folder.')
