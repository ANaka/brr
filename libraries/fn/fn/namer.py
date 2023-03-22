import os
import tempfile

import boto3
from fn.fn import Fn


def get_plot_id():
    return os.environ["plot_id"]


class Namer(Fn):
    try:
        session = boto3.Session(profile_name="naka")
    except:
        session = boto3.Session()
    s3 = session.client("s3")

    def save_plot_id(self, Bucket="algorithmic-ink", Key="current_plot_id"):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        plot_id = self.name
        print(f"saved {plot_id} to s3://{Bucket}/{Key}")
        tmp.write(plot_id.encode())
        tmp.close()
        self.s3.upload_file(Filename=tmp.name, Bucket=Bucket, Key=Key)
        return plot_id

    def get_current_plot_id(self, Bucket="algorithmic-ink", Key="current_plot_id"):
        return self.s3.get_object(Bucket=Bucket, Key=Key)["Body"].read().decode()

    def save_nft_id(self, Bucket="algorithmic-ink", Key="current_nft_id"):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        plot_id = self.name
        print(f"saved {plot_id} to s3://{Bucket}/{Key}")
        tmp.write(plot_id.encode())
        tmp.close()
        self.s3.upload_file(Filename=tmp.name, Bucket=Bucket, Key=Key)
        return plot_id

    def get_current_nft_id(self, Bucket="algorithmic-ink", Key="current_nft_id"):
        return self.s3.get_object(Bucket=Bucket, Key=Key)["Body"].read().decode()


def new_plot_id(*args, **kwargs):
    namer = Namer(*args, **kwargs)
    plot_id = namer.save_plot_id()
    return plot_id


def get_current_plot_id(*args, **kwargs):
    namer = Namer(*args, **kwargs)
    return namer.get_current_plot_id()


def new_nft_id(*args, **kwargs):
    namer = Namer(*args, **kwargs)
    plot_id = namer.save_nft_id()
    return plot_id


def get_current_nft_id(*args, **kwargs):
    namer = Namer(*args, **kwargs)
    return namer.get_current_nft_id()
