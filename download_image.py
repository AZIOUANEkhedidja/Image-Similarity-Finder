from icrawler.builtin import GoogleImageCrawler

def download_images():
    keywords = ['texture', 'human', 'nature', 'cat', 'dog', 'horse', 'baby', 'pizza', 'cake', 'flower']
    
    for keyword in keywords:
        crawler = GoogleImageCrawler(storage={'root_dir': f'./images/{keyword}'})
        crawler.crawl(keyword=keyword, max_num=100)

download_images()
