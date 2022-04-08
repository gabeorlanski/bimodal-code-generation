import logging
import re
from urllib.parse import urlparse, urljoin

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor, IGNORED_EXTENSIONS
from scrapy.http import Request, HtmlResponse
from urllib.parse import urlparse
import shutil
from ..items import Page

logger = logging.getLogger()


class TutorialSpider(CrawlSpider):
    name = 'docspider'

    def __init__(self, out_name, output_path, url, allowed_path, disallow=None, *a, **kw):
        super().__init__(*a, **kw)
        disallow = disallow or []

        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://%s/' % url
        self.url = url
        self.allowed_domains = [
            re.sub(r'^www\.', '', urlparse(url).hostname)
        ]
        self.allowed_path = allowed_path
        self.link_extractor = LinkExtractor(
            allow=urljoin(urlparse(url).hostname, allowed_path) + r'/.*',
            deny=disallow,
            deny_extensions=IGNORED_EXTENSIONS + ['php', 'rst', 'txt', 'tgz', 'asc', 'gz']
        )
        self.cookies_seen = set()

        if output_path.joinpath(out_name).exists():
            shutil.rmtree(output_path.joinpath(out_name))

        self.output_path = output_path.joinpath(out_name)
        self.output_path.mkdir(parents=True)

    def start_requests(self):
        return [Request(self.url, callback=self.parse, dont_filter=True)]

    def parse(self, response):
        """Parse a PageItem and all requests to follow
        @url http://www.scrapinghub.com/
        @returns items 1 1
        @returns requests 1
        @scrapes url title foo
        """
        cleaned_name = urlparse(response.url).path[1:]
        if self.allowed_path:
            assert self.allowed_path in cleaned_name
            cleaned_name = cleaned_name[len(self.allowed_path):]
        if cleaned_name.startswith('/'):
            cleaned_name = cleaned_name[1:]
        if not cleaned_name:
            cleaned_name = 'index.html'
        cleaned_name = cleaned_name.replace('/', '_')
        if not cleaned_name.endswith('.html'):
            cleaned_name = f'{cleaned_name}.html'
        out_file = self.output_path.joinpath(cleaned_name)
        if out_file.exists():
            logger.warning(f"Not saving {response.url} as it is a duplicate")
            return []
        logger.info(f'Saving {cleaned_name}')
        with out_file.open('w') as f:
            f.write(response.text.strip())
        page = self._get_item(response, cleaned_name)
        r = [page]
        r.extend(self._extract_requests(response))
        return r

    def _get_item(self, response, cleaned_name):
        referer = response.request.headers.get('Referer')
        if referer and not isinstance(referer, str):
            referer = referer.decode('utf-8')

        item = Page(
            cleaned_name=cleaned_name,
            url=response.url,
            size=str(len(response.body)),
            referer=referer,
        )
        self._set_title(item, response)
        return item

    def _extract_requests(self, response):
        r = []
        if isinstance(response, HtmlResponse):
            links = self.link_extractor.extract_links(response)
            r.extend(Request(x.url, callback=self.parse) for x in links)
        return r

    def _set_title(self, page, response):
        if isinstance(response, HtmlResponse):
            title = response.xpath("//title/text()").extract()
            if title:
                page['title'] = title[0]
