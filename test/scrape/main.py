import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

movies = []
movie_titles = []
movie_images = []
movie_contents = []
df = pd.DataFrame(columns = ['Title', 'Image', 'Content'])

url = "https://myvideogamelist.com/"
r = requests.get(url)

soup = bs(r.content, 'html5lib')


table = soup.findAll('div', attrs={'class': 'panel-body'})
# print(table.h2)
for row in table:
   movs = row.findAll('div')
   for mov in movs:
      titles = mov.find_all('h2')
      if titles:
          
        #   movie_titles.append()
         for title in titles:
                movie_titles.append(title.text)
                con =  title.find_next_sibling('p')
                con1 =  con.find_next_sibling('p')
                con2 =  con1.find_next_sibling('p')
                movie_images.append(con1.find('img')['src'])
                movie_contents.append(con2.text)

df['Title'] = movie_titles
df['Image'] = movie_images
df['Content'] = movie_contents

df.to_csv('movies.csv')