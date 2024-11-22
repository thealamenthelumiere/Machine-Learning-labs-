import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from dataclasses import dataclass

# Конфигурация
BASE_URL = 'http://books.toscrape.com/'

@dataclass
class Book:
    title: str
    price: float
    rating: str
    availability: str
    category: str
    upc: str
    product_type: str
    price_excl_tax: float
    price_incl_tax: float
    tax: float
    description: str

    @classmethod
    def from_soup(cls, soup, category):
        # Извлечение заголовка
        title = soup.find('h1').get_text()
        # Извлечение цены
        price = float(soup.find('p', class_='price_color').get_text()[1:])
        # Извлечение рейтинга
        rating = soup.find('p', class_='star-rating')['class'][1]
        # Извлечение доступности
        availability = soup.find('p', class_='instock availability').get_text(strip=True)
        # Извлечение информации о продукте
        product_info = soup.find('table', class_='table table-striped')
        product_info_dict = {}
        for row in product_info.find_all('tr'):
            key = row.find('th').get_text()
            value = row.find('td').get_text()
            product_info_dict[key] = value
        upc = product_info_dict.get('UPC')
        product_type = product_info_dict.get('Product Type')
        price_excl_tax = float(product_info_dict.get('Price (excl. tax)')[1:])
        price_incl_tax = float(product_info_dict.get('Price (incl. tax)')[1:])
        tax = float(product_info_dict.get('Tax')[1:])
        # Извлечение описания
        description_tag = soup.find('div', id='product_description')
        if description_tag:
            description = description_tag.find_next_sibling('p').get_text()
        else:
            description = ''
        return cls(title, price, rating, availability, category, upc, product_type, price_excl_tax, price_incl_tax, tax, description)

def get_category_urls():
    url = BASE_URL + 'index.html'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    category_links = soup.find('ul', class_='nav-list').find('ul').find_all('a')
    category_urls = [BASE_URL + category_link['href'] for category_link in category_links]
    return category_urls

def get_book_urls(category_url):
    book_urls = []
    while True:
        response = requests.get(category_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article', class_='product_pod')
        for article in articles:
            book_url = article.find('h3').find('a')['href']
            book_url = book_url.replace('../../../', BASE_URL + 'catalogue/')
            book_urls.append(book_url)
        # Проверка наличия следующей страницы
        next_button = soup.find('li', class_='next')
        if next_button:
            next_page_url = next_button.find('a')['href']
            category_url = category_url.rsplit('/', 1)[0] + '/' + next_page_url
        else:
            break
    return book_urls

def get_book_data(book_url, category):
    response = requests.get(book_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    book = Book.from_soup(soup, category)
    return book

def main():
    category_urls = get_category_urls()
    books_data = []
    for category_url in category_urls:
        category = category_url.split('/')[-2]
        print(f"Обработка категории: {category}")
        book_urls = get_book_urls(category_url)
        print(f"Найдено {len(book_urls)} книг в категории {category}")
        for book_url in book_urls:
            try:
                book = get_book_data(book_url, category)
                books_data.append(book)
                time.sleep(0.1)  # Вежливая пауза между запросами
            except Exception as e:
                print(f"Ошибка при обработке книги по адресу {book_url}: {e}")
    # Создание DataFrame
    books_df = pd.DataFrame([book.__dict__ for book in books_data])
    # Добавление дополнительных переменных
    books_df['title_length'] = books_df['title'].apply(len)
    books_df['description_length'] = books_df['description'].apply(len)
    books_df['num_words_description'] = books_df['description'].apply(lambda x: len(x.split()))
    # Сохранение в CSV
    books_df.to_csv('books.csv', index=False)
    print("Датасет сохранен в файл books.csv")

if __name__ == "__main__":
    main()
