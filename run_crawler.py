from project.crawler.crawler import Crawler
import requests
# # clear any existing log handlers
# import logging
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)

# # set up fresh logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# seeds = []

# start = ["https://www.bing.com/search?q=t%c3%bcbingen&lq=0&pq&setlang=en&cc=us", "https://www.bing.com/search?q=t%c3%bcbingen&lq=0&pq=&setlang=en&cc=us&FPIG=7336BEE677774276834C0FC4D98825F0&first=9"]


# test the crawler with a single seed URL
if __name__ == "__main__":
    crawler = Crawler(
        seed=["https://en.wikipedia.org/wiki/Baden-W%C3%BCrttemberg"],
        auto_resume=True,
        verbose=False
    )

    print("Starting crawler...")
    crawler.run(amount=30)

