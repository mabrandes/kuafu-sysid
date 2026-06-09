"""kuafu_sysid — system identification (model fitting/validation) for kuafu projects."""

__version__ = "0.1.0"

from kuafu_sysid.fit import fit_model
from kuafu_sysid.validate import validate_model

__all__ = ["fit_model", "validate_model"]
