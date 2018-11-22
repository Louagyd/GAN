import os
from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler
from icrawler.builtin import BaiduImageCrawler
from icrawler.builtin import FlickrImageCrawler
from icrawler.builtin import GreedyImageCrawler


if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-k", "--keyword", action="store", type="string", dest="keyword", default="", help="keyword to search, you should use _ instead of space, example: spiral_galaxy")
    parser.add_option("-d", "--dir", action="store", type="string", dest="dir", default="crawled", help="target directory")
    parser.add_option("-e", "--engine", action="store", type="string", dest="engine", default="google", help="which engine: google, bing, baidu, flickr, greedy")
    parser.add_option("-o", "--offset", action="store", type="string", dest="offset", default="0", help="offset")
    parser.add_option("-l", "--limit",  action="store", type="string", dest="limit", default="10000", help="limit")


    (options, args) = parser.parse_args()

    keyword = options.keyword
    engine = options.engine
    dir = options.dir
    offset = int(options.offset)
    limit = int(options.limit)

    if engine == "google":
        crawler = GoogleImageCrawler(storage={'root_dir': 'dir'})
    elif engine == "bing":
        crawler = BingImageCrawler(storage={'root_dir': 'dir'})
    elif engine == "baidu":
        crawler = BaiduImageCrawler(storage={'root_dir': 'dir'})
    elif engine == "flickr":
        crawler = FlickrImageCrawler(storage={'root_dir': 'dir'})
    elif engine == "greedy":
        crawler = GreedyImageCrawler(storage={'root_dir': 'dir'})
    else:
        crawler = GoogleImageCrawler(storage={'root_dir': 'dir'})

    crawler.crawl(keyword=keyword.replace("_", " "), offset=offset, max_num=limit)


# bing_crawler = BingImageCrawler('crawl_target/bing')
# bing_crawler.crawl(keyword='spiral galaxy', offset=0, max_num=1000,
#                    feeder_thr_num=1, parser_thr_num=1, downloader_thr_num=4,
#                    min_size=None, max_size=None)
# baidu_crawler = BaiduImageCrawler('crawl_target/baidu')
# baidu_crawler.crawl(keyword='spiral galaxy', offset=0, max_num=1000,
#                     feeder_thr_num=1, parser_thr_num=1, downloader_thr_num=4,
#                     min_size=None, max_size=None)