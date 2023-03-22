from fn.namer import Namer

def test_plot_ids():
    namer = Namer()
    s = namer.save_plot_id()
    assert s == namer.get_current_plot_id()
    