import os

import yaml
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from pathlib import Path
import click

if Path().absolute().stem == 'crawler':
    root = Path().absolute().parents[2]
else:
    root = Path().absolute()
save_path = root.joinpath('data/crawled_tutorials')
print(save_path)


@click.command()
@click.argument('name')
def crawl(name):
    os.environ.setdefault('SCRAPY_SETTINGS_MODULE', 'crawler.settings')
    settings = get_project_settings()
    if not root.joinpath('data/crawled_maps').exists():
        root.joinpath('data/crawled_maps').mkdir()
    cfg = yaml.load(root.joinpath(f"data/tutorials/cfg/{name}.yaml").open(), yaml.Loader)

    with root.joinpath(f'data/crawled_maps/{name}.json').open('w'):
        pass
    settings['FEEDS'] = {
        str(root.joinpath(f'data/crawled_maps/{name}.json')): {"format": "json"}
    }
    process = CrawlerProcess(settings)

    disallow = cfg.get('disallow', [])
    print(f"Starting from {cfg['url']}")

    # 'followall' is the name of one of the spiders of the project.
    process.crawl(
        'docspider',
        out_name=name,
        url=cfg['url'],
        output_path=save_path,
        allowed_path=cfg.get('allowed_path', None),
        disallow=disallow,
        disallow_file_types=cfg.get('disallow_file_types', [])
    )
    process.start()


if __name__ == '__main__':
    crawl()
