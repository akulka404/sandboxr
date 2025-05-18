# sandboxr/cli.py

import argparse
import sys
from .manager import SandboxManager

def main():
    p = argparse.ArgumentParser(prog="sandboxr")
    p.add_argument("--backend", choices=["virtualenv","docker"], required=True)
    p.add_argument("--packages", default="", help="comma-separated list")
    p.add_argument("--timeout", "-t", type=int, default=30, help="Max seconds to allow code to run (default: 30)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--code", help="Python code string")
    grp.add_argument("--file", help="Path to .py file")
    p.add_argument("--keep", action="store_true", help="Do not teardown")
    args = p.parse_args()

    pkgs = args.packages.split(",") if args.packages else []
    mgr = SandboxManager(args.backend, pkgs)
    sandbox = mgr.create()

    if args.code:
        out, err, code = sandbox.exec(args.code, timeout=args.timeout)
    else:
        out, err, code = sandbox.exec_file(args.file, timeout=args.timeout)

    if out:
        # Print to stdout
        sys.stdout.write(out)
    if err:
        # Print errors to stderr
        sys.stderr.write(err)

    if not args.keep:
        sandbox.teardown()

    sys.exit(code)

if __name__ == "__main__":
    main()
