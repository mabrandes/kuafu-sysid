def test_import():
    import kuafu_sysid
    from kuafu_sysid import fit_model, validate_model

    assert kuafu_sysid.__version__
    assert callable(fit_model)
    assert callable(validate_model)
