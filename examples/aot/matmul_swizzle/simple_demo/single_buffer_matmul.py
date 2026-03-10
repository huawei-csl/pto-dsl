import argparse

from step1_baseline import build


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    print(build())
