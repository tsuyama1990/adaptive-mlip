import os

from pyacemaker.domain_models.constants import (
    ENV_VAR_NAME_UV_PROJECT_ENV,
    UV_PROJECT_ENVIRONMENT_DEFAULT,
)

os.environ[ENV_VAR_NAME_UV_PROJECT_ENV] = os.environ.get(
    ENV_VAR_NAME_UV_PROJECT_ENV, UV_PROJECT_ENVIRONMENT_DEFAULT
)
