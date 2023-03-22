
from fn.namer import (
    Namer,
    get_current_nft_id,
    get_current_plot_id,
    new_nft_id,
    new_plot_id,
)
from fn.utils import get_time, head_tail, overlay


def handle_path_args(args):
    path_style = "rel"  # relative path style: dir/file.ext
    if args["-a"]:  # file name only: file.ext
        path_style = "file"
    elif args["-A"]:  # absolute path style: /a/b/file.ext
        path_style = "abs"
    return overlay(args, path_style=path_style)


def handle_args(fn, args):
    args = handle_path_args(args)
    if args["-l"]:
        return fn.lst(d=args["DIR"], path_style=args["path_style"], ext=not args["-f"])
    if args["-L"]:
        return fn.lst_recent(d=args["DIR"], path_style=args["path_style"], ext=not args["-f"])
    if args["-r"]:
        return fn.recent(d=args["DIR"], path_style=args["path_style"], ext=not args["-f"])
    if args["-s"]:
        return fn.recent_prochash(d=args["DIR"])
    if args["-p"]:
        return [fn.get_pid_sha()]
    if args["-g"]:
        return [fn.get_sha()]
    return [fn.name(milli=args["-m"])]
