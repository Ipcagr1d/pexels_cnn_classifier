import os
import csv


from pexels_api import API
from dotenv import load_dotenv

load_dotenv()

# set up API
api_key = os.getenv('PEXELS_API_KEY')
api = API('NML9LYjorpAVsQdHb6g8eRMzwNajuoTl0QxwdB85aWN2OkE6iBYMmCRu')

# set up search criteria
search_terms = ['lemon', 'lychee', 'grape'] # keywords for search
num_images = 150 # number of images to download for each keyword
page_size = 80 # maximum number of images per page

# loop through each search term
for term in search_terms:
    # create folder to save images and CSV file
    folder_name = f'{term}_images'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    csv_file = os.path.join(folder_name, f'{term}_urls.csv')
    
    # search for images based on term
    photos = []
    for page in range(1, (num_images // page_size) + 2):
        results = api.search(term, page=page, results_per_page=page_size)['photos']
        if not results:
            break
        photos.extend(results)
        if len(photos) >= num_images:
            break
    photos = photos[:num_images]

    # save URLs to CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'image_url'])
        for i, photo in enumerate(photos):
            image_id = f'{term}_{i}'
            image_url = photo['src']['original']
            writer.writerow([image_id, image_url])
            print(f'Saved URL for {image_id} to {csv_file} file.')