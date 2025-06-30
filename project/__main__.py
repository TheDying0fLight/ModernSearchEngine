from .frontend import PageFactory
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--host", help = "Address of the Website", default='localhost')
parser.add_argument("--port", help = "Port of the Website", default='8080')

args = parser.parse_args()

factory = PageFactory()
factory.run(host=args.host, port=int(args.port))