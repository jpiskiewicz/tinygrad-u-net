#!/usr/bin/env python3

from sys import argv
from collections import Counter
from json import dumps


if __name__ == "__main__":
  with open(argv[1]) as f:
    cache_misses = f.readlines()
  print(dumps(dict(Counter(cache_misses)), indent=2))
  