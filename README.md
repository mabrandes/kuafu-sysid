# kuafu-sysid

System identification (model fitting & validation) shared across `kuafu_*` projects.

## Install

Editable (during development):

```toml
[tool.uv.sources]
kuafu-sysid = { path = "../kuafu-sysid", editable = true }
```

Git (once pushed to GitHub):

```toml
kuafu-sysid = { git = "https://github.com/mabrandes/kuafu-sysid.git" }
```

## Layout

- `fit.py` — `fit_model(data, target, ...)` model identification.
- `validate.py` — `validate_model(model, data, target, ...)` metrics/diagnostics.
