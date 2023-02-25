import os
from glob import glob
from os import getpid

from .files import (
    deduplicate_files,
    get_file_name_tokenizer,
    rel_abs_path,
    remove_extension,
)
from .git import _init_git_sha_cmd
from .utils import get_time, getsha, overlay, sortfx

SEP = "-"


class Fn:
    def __init__(self, prefix="", postfix="", git_sha_size=5, pid_sha_size=6, milli=True, utc=False):
        self.git_sha_size = git_sha_size
        self.pid_sha_size = pid_sha_size
        self.postfix = postfix
        self.prefix = prefix
        self.tokenizer = get_file_name_tokenizer(SEP, git_sha_size, pid_sha_size)
        self.gitsha = _init_git_sha_cmd(self.git_sha_size)
        self.pid_sha = self.__get_pid_time_sha()
        self.milli = milli
        self.utc = utc

    def __enter__(self):
        return self

    def __exit__(self, t, value, traceback):
        return False

    def __is_git(self):
        if not self.gitsha:
            raise ValueError("not in a git repo")
        return self.gitsha

    def __get_pid_time_sha(self):
        return getsha([get_time(), getpid()])[: self.pid_sha_size]

    @property
    def name(self):
        l = [self.prefix, get_time(milli=self.milli, utc=self.utc), SEP, self.gitsha, SEP, self.pid_sha, self.postfix]
        return "".join(l)

    def _get_current_files(self, d=".", path_style="rel", ext=True):
        if d is None:
            d = "."
        files = self.tokenizer(glob("{:s}/*".format(d)))
        if not ext:
            files = deduplicate_files([overlay(f, _raw=remove_extension(f["_raw"])) for f in files])

        return rel_abs_path(d, path_style, sorted(files, key=sortfx))

    def get_pid_sha(self):
        return self.pid_sha

    def get_sha(self):
        return self.__is_git()

    def recent(self, **args):
        current = list(self._get_current_files(**args))
        if not current:
            return []
        prochash = current[-1]["prochash"]
        return map(lambda f: f["_raw"], filter(lambda f: f["prochash"] == prochash, current))

    def lst(self, **args):
        sha = self.get_sha()
        return map(lambda x: x["_raw"], filter(lambda x: x["gitsha"] == sha, self._get_current_files(**args)))

    def lst_recent(self, **args):
        self.get_sha()
        current = list(self._get_current_files(**args))
        if not current:
            return []
        sha = current[-1]["gitsha"]
        return map(lambda x: x["_raw"], filter(lambda x: x["gitsha"] == sha, current))

    def recent_prochash(self, d):
        current = list(self._get_current_files(d))
        if current:
            yield current[-1]["prochash"]
