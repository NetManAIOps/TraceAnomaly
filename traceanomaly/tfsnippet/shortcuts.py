"""
This package provides shortcuts to utilities from second-level packages.
"""

from .dataflows import *
from .utils.config_ import *
from .utils.config_utils import *
from .utils.model_vars import *
from .utils.reuse import *

__all__ = [
    # from tfsnippet.dataflows
    'DataFlow', 'DataMapper', 'SlidingWindow',

    # from tfsnippet.utils.model_vars
    'model_variable', 'get_model_variables',

    # from tfsnippet.utils.config_
    'settings',

    # from tfsnippet.utils.config_utils
    'Config', 'ConfigField',
    'get_config_defaults', 'register_config_arguments',

    # from tfsnippet.utils.reuse
    'get_reuse_stack_top', 'instance_reuse', 'global_reuse',
    'VarScopeObject',
]
