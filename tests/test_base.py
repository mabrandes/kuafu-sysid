from kuafu_sysid.models.base import Forecaster


def test_forecaster_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        Forecaster()  # abstract, cannot instantiate
