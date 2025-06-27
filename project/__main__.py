from .frontend import SearchEnginePage
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--host", help = "Address of the Website", default='localhost')
parser.add_argument("--port", help = "Port of the Website", default='8080')

args = parser.parse_args()

page = SearchEnginePage()
page.run(host=args.host, port=int(args.port))