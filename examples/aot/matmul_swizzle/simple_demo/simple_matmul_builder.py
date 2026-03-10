import argparse

from step2_doublebuffer import build as build_step2
from step3_swizzle import build as build_step3
from step4_manual_pipelining import build as build_step4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Emit explicit record/wait events instead of relying on auto sync insertion.",
    )
    parser.add_argument(
        "--disable-swizzle",
        action="store_true",
        help="Emit step2 (double-buffer only) instead of swizzled versions.",
    )
    args = parser.parse_args()
    if args.manual_sync:
        print(build_step4())
    elif args.disable_swizzle:
        print(build_step2())
    else:
        print(build_step3())
