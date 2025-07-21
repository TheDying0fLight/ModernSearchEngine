from .frontend import PageFactory
import argparse
import logging

parser = argparse.ArgumentParser()

parser.add_argument("--host", help = "Address of the Website", default='127.0.0.1')
parser.add_argument("--port", help = "Port of the Website", default='8080')
parser.add_argument("--no_logging", help = "Disable logging", action='store_true')
parser.add_argument("--path", help = "Path to index", default='data')

args = parser.parse_args()

factory = PageFactory()
if args.no_logging:
    logging.disable()
factory.run(host=args.host, port=int(args.port))