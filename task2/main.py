"""Backward-compatible source entrypoint."""

# try:
#     from cogcappro.cli.train import main
#     pass
# except ModuleNotFoundError as exc:
#     if exc.name != "cogcappro":
#         raise
#     from src.cogcappro.cli.train import main

from src.cogcappro.cli.train import main
if __name__ == "__main__":
    main()
