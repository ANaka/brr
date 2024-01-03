import tempfile
from glob import glob
from os import getpid
from pathlib import Path
from typing import Union

import boto3

from .files import (
    deduplicate_files,
    get_file_name_tokenizer,
    rel_abs_path,
    remove_extension,
)
from .git import _init_git_sha_cmd
from .utils import OUTPUTS_ROOT, get_time, getsha, overlay, sortfx

SEP = "-"
BUCKET = "algorithmic-ink-versioned"
KEY = "current_output_id"


class Fn:
    def __init__(self, prefix="", postfix="", git_sha_size=5, pid_sha_size=3, milli=False, utc=False):
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
        element_list = [
            self.prefix,
            get_time(milli=self.milli, utc=self.utc),
            self.gitsha,
            self.pid_sha,
            self.postfix,
        ]
        element_list = [e for e in element_list if len(e) > 0]
        return SEP.join(element_list)

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

    @staticmethod
    def save_output_id(
        output_id,
        Bucket: str = None,
        Key: str = None,
        boto3_session_kwargs=None,
    ):

        boto3_session_kwargs = {} if boto3_session_kwargs is None else boto3_session_kwargs
        Bucket = BUCKET if Bucket is None else Bucket
        Key = KEY if Key is None else Key
        session = boto3.Session(**boto3_session_kwargs).client("s3")
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(output_id.encode())
        tmp.close()
        session.upload_file(Filename=tmp.name, Bucket=Bucket, Key=Key)
        return f" s3://{Bucket}/{Key}"

    @staticmethod
    def get_current_output_id(
        Bucket: str = None,
        Key: str = None,
        boto3_session_kwargs=None,
    ):
        boto3_session_kwargs = {} if boto3_session_kwargs is None else boto3_session_kwargs
        Bucket = BUCKET if Bucket is None else Bucket
        Key = KEY if Key is None else Key
        session = boto3.Session(**boto3_session_kwargs).client("s3")
        return session.get_object(Bucket=Bucket, Key=Key)["Body"].read().decode()

    def new_savepath(
        self,
        ext: str = ".svg",
        outputs_root: Union[str, Path] = None,
        Bucket: str = None,
        Key: str = None,
        boto3_session_kwargs=None,
    ):
        outputs_root = Path(outputs_root) if outputs_root is not None else OUTPUTS_ROOT
        _ = self.save_output_id(self.name, Bucket=Bucket, Key=Key, boto3_session_kwargs=boto3_session_kwargs)
        return outputs_root / f"{self.name}{ext}"


def new_savepath(postfix: str = "", ext: str = ".svg"):
    """convenience function, use Fn().new_savepath() for more control"""
    return Fn(postfix=postfix).new_savepath(ext=ext)
