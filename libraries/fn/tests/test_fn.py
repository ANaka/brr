from fn.fn import Fn


def test_Fn():
    assert isinstance(Fn().name, str)


def test_new_savepath():
    fp = Fn(postfix="test").new_savepath(ext=".txt")

    # write empty text to file
    with open(fp, "w") as f:
        f.write("")

    assert fp.exists()

    # remove file
    fp.unlink()
