from fn.fn import Fn, new_savepath


def test_Fn():
    assert isinstance(Fn().name, str)


def test_Fn_new_savepath():
    fp = Fn(postfix="test").new_savepath(ext=".txt")

    # write empty text to file
    with open(fp, "w") as f:
        f.write("")

    assert fp.exists()

    # remove file
    fp.unlink()


def test_new_savepath():
    fp = new_savepath(postfix="test2", ext=".txt")
    # write empty text to file
    with open(fp, "w") as f:
        f.write("")

    assert fp.exists()

    # remove file
    fp.unlink()
