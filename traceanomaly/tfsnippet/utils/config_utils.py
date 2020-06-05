from argparse import ArgumentParser, Action
from collections import OrderedDict
from contextlib import contextmanager

import six
import numpy as np
import yaml

from .doc_utils import DocInherit
from .type_utils import is_integer, is_float

__all__ = [
    'ConfigValidator', 'IntConfigValidator', 'FloatConfigValidator',
    'BoolConfigValidator', 'StrConfigValidator',
    'register_config_validator', 'get_config_validator',
    'Config', 'ConfigField',
    'get_config_defaults', 'register_config_arguments', 'scoped_set_config',
]


@DocInherit
class ConfigValidator(object):
    """Base config value validator."""

    def validate(self, value, strict=False):
        """
        Validate the `value`.

        Args:
            value: The value to be validated.
            strict (bool): If :obj:`True`, disable type conversion.
                If :obj:`False`, the validator will try its best to convert the
                input `value` into desired type.

        Returns:
            The validated value.

        Raises:
            ValueError, TypeError: If the value cannot pass validation.
        """
        raise NotImplementedError()


class IntConfigValidator(ConfigValidator):
    """Config value validator for integer values."""

    def validate(self, value, strict=False):
        if not strict:
            int_value = int(value)
            float_value = float(value)
            if np.abs(int_value - float_value) > np.finfo(float_value).eps:
                raise TypeError('casting a float number into integer is '
                                'not allowed')
            value = int_value
        if not is_integer(value):
            raise TypeError('{!r} is not an integer'.format(value))
        return value


class FloatConfigValidator(ConfigValidator):
    """Config value validator for float values."""

    def validate(self, value, strict=False):
        if not strict:
            value = float(value)
        if not is_float(value):
            raise TypeError('{!r} is not a float number'.format(value))
        return value


class BoolConfigValidator(ConfigValidator):
    """Config value validator for boolean values."""

    def validate(self, value, strict=False):
        if not strict:
            if isinstance(value, six.string_types):
                value = str(value).lower()
                if value in ('1', 'on', 'yes', 'true'):
                    value = True
                elif value in ('0', 'off', 'no', 'false'):
                    value = False
            elif is_integer(value):
                if value == 1:
                    value = True
                elif value == 0:
                    value = False
            if not isinstance(value, bool):
                raise TypeError('{!r} cannot be casted into boolean'.
                                format(value))
        else:
            if not isinstance(value, bool):
                raise TypeError('{!r} is not a boolean'.format(value))
        return value


class StrConfigValidator(ConfigValidator):
    """Config value validator for string values."""

    def validate(self, value, strict=False):
        if not strict:
            value = str(value)
        if not isinstance(value, six.string_types):
            raise TypeError('{!r} is not a string'.format(value))
        return value


_config_validators_registry = OrderedDict([
    (int, IntConfigValidator),
    (float, FloatConfigValidator),
    (bool, BoolConfigValidator),
    (six.binary_type, StrConfigValidator),
    (six.text_type, StrConfigValidator),
])


def register_config_validator(type, validator_class):
    """
    Register a config value validator.

    Args:
        type: The value type.
        validator_class: The validator class type.
    """
    _config_validators_registry[type] = validator_class


def get_config_validator(type):
    """
    Get an instance of :class:`ConfigValidator` for specified `type`.

    Args:
        type: The value type.

    Returns:
        ConfigValidator: The config value validator.
    """
    if type in _config_validators_registry:
        return _config_validators_registry[type]()
    for type_, cls in six.iteritems(_config_validators_registry):
        if issubclass(type, type_):
            return cls()
    raise TypeError('No validator has been registered for `type` {!r}'.
                    format(type))


class ConfigField(object):
    """A config field."""

    def __init__(self, type, default=None, description=None, nullable=False,
                 choices=None):
        """
        Construct a new :class:`ConfigField`.

        Args:
            type: The value type.
            default: The default config value.
                :obj:`None` if not specified.
            description: The config description.
            nullable: Whether or not :obj:`None` is a valid value?
                Default is :obj:`False`.
            choices: Optional valid choices for the config value.
        """
        validator = get_config_validator(type)
        nullable = bool(nullable)
        if not nullable and default is None:
            raise ValueError('`nullable` is False, but `default` value is not '
                             'specified.')
        if choices is not None:
            choices = tuple(validator.validate(v, strict=True) for v in choices)
            if default is not None and default not in choices:
                raise ValueError('Invalid value for `default`: {!r} is not '
                                 'one of {}'.format(default, list(choices)))

        self._validator = validator
        self._type = type
        self._default_value = default
        self._description = description
        self._nullable = nullable
        self._choices = choices

    @property
    def type(self):
        """Get the value type."""
        return self._type

    @property
    def default_value(self):
        """Get the default config value."""
        return self._default_value

    @property
    def description(self):
        """Get the config description."""
        return self._description

    @property
    def nullable(self):
        """Whether or not :obj:`None` is a valid value?"""
        return self._nullable

    @property
    def choices(self):
        """Get the valid choices of the config value."""
        return self._choices

    def validate(self, value, strict=False):
        """
        Validate the config `value`.

        Args:
            value: The value to be validated.
            strict (bool): If :obj:`True`, disable type conversion.
                If :obj:`False`, the validator will try its best to convert the
                input `value` into desired type.

        Returns:
            The validated value.
        """
        if value is None:
            if not self.nullable:
                raise ValueError('null value is not allowed')
            return None

        value = self._validator.validate(value, strict=strict)

        if value is not None and self._choices is not None and \
                value not in self._choices:
            raise ValueError('{!r} is not one of {}'.
                             format(value, list(self._choices)))

        return value


class Config(object):
    """
    Base class for defining config values.

    Derive sub-classes of :class:`Config`, and define config values as
    public class attributes.  For example::

        class YourConfig(Config):
            max_epoch = 100
            learning_rate = 0.01
            activation = ConfigField(str, default='leaky_relu',
                                     choices=['sigmoid', 'relu', 'leaky_relu'])
            l2_regularization = ConfigField(float, default=None)

        config = YourConfig()

    When an attribute is assigned, it will be validated by:

    1.  If the attribute is defined as a :class:`ConfigField`, then its
        :meth:`validate` will be used to validate the value.
    2.  If the attribute is not :obj:`None`, and a validator is registered
        for `type(value)`, then an instance of this type of validator will
        be used to validate the value.
    3.  Otherwise if the attribute is not defined, or is :obj:`None`,
        then no validation will be performed.

    For example::

        config.l2_regularization = 5e-4  # okay
        config.l2_regularization = 'xxx'  # raise an error
        config.activation = 'sigmoid'  # okay
        config.activation = 'tanh'  # raise an error
        config.new_attribute = 'yyy'  # okay

    The config object also implements dict-like interface::

        # to test whether a key exists
        print(key in config)

        # to iterate through all config values
        for key in config:
            print(key, config[key])

        # to set a config value
        config[key] = value

    You may get all the config values of a config object as dict::

        print(config_to_dict(config))

    Or you may get the default values of the config class as dict::

        print(config_defaults(YourConfig))
        print(config_defaults(config))  # same as the above line
    """

    def __init__(self):
        """Construct a new :class:`Config`."""
        for key in self:
            value = self[key]
            if isinstance(value, ConfigField):
                value = value.default_value
            self[key] = value

    def __setattr__(self, key, value):
        cls = self.__class__
        cls_value = getattr(cls, key, None)

        if isinstance(cls_value, ConfigField):
            value = cls_value.validate(value)
        elif cls_value is not None:
            if value is None:
                raise ValueError('null value is not allowed')
            try:
                validator = get_config_validator(type(cls_value))
            except TypeError:
                pass
            else:
                value = validator.validate(value)

        object.__setattr__(self, key, value)

    def __iter__(self):
        return (key for key in dir(self) if key in self)

    def __contains__(self, key):
        return (hasattr(self, key) and
                not key.startswith('_') and
                not hasattr(Config, key))

    def __getitem__(self, key):
        if key not in self:
            raise KeyError('`{}` is not a config key of `{}`.'.
                           format(key, self))
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key.startswith('_') or hasattr(Config, key):
            raise KeyError('`{}` is reserved and cannot be a config key.'.
                           format(key))
        setattr(self, key, value)

    def update(self, key_values):
        """
        Update the config values from `key_values`.

        Args:
            key_values: A dict, or a sequence of (key, value) pairs.
        """
        if not isinstance(key_values, (dict, OrderedDict)):
            key_values = dict(key_values)
        for k, v in six.iteritems(dict(key_values)):
            self[k] = v

    def to_dict(self):
        """
        Get the config values as a dict.

        Returns:
            dict[str, any]: The config values.
        """
        return {key: self[key] for key in self}


def get_config_defaults(config):
    """
    Get the default config values of `config`.

    Args:
        config: An instance of :class:`Config`, or a class which is a
            subclass of :class:`Config`.

    Returns:
        dict[str, any]: The default config values of `config`.
    """
    if isinstance(config, Config):
        config = config.__class__
    if not isinstance(config, six.class_types) or \
            not issubclass(config, Config):
        raise TypeError('`config` must be an instance of `Config`, or a '
                        'subclass of `Config`: got {!r}.'.format(config))
    ret = {}
    for key in dir(config):
        if not key.startswith('_') and not hasattr(Config, key):
            value = getattr(config, key)
            if isinstance(value, ConfigField):
                value = value.default_value
            ret[key] = value
    return ret


class _ConfigAction(Action):

    def __init__(self, config_obj, config_key, option_strings, dest, **kwargs):
        super(_ConfigAction, self).__init__(option_strings, dest, **kwargs)
        self._config_obj = config_obj
        self._config_key = config_key

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            value = yaml.load(values)
            self._config_obj[self._config_key] = value
        except Exception as ex:
            message = 'Invalid value for argument `{}`'.format(option_string)
            if str(ex):
                message += '; ' + str(ex)
            if not message.endswith('.'):
                message += '.'
            raise ValueError(message)
        else:
            setattr(namespace, self.dest, self._config_obj[self._config_key])


def register_config_arguments(config, parser, prefix=None, title=None,
                              description=None, sort_keys=False):
    """
    Register config to the specified argument parser.

    Usage::

        class YourConfig(Config):
            max_epoch = 1000
            learning_rate = 0.01
            activation = ConfigField(
                str, default='leaky_relu', choices=['relu', 'leaky_relu'])

        # First, you should obtain an instance of your config object
        config = YourConfig()

        # You can then parse config values from CLI arguments.
        # For example, if sys.argv[1:] == ['--max_epoch=2000']:
        from argparse import ArgumentParser
        parser = ArgumentParser()
        spt.register_config_arguments(config, parser)
        parser.parse_args(sys.argv[1:])

        # Now you can access the config value `config.max_epoch == 2000`
        print(config.max_epoch)

    Args:
        config (Config): The config object.
        parser (ArgumentParser): The argument parser.
        prefix (str): Optional prefix of the config keys.
            `new_config_key = prefix + '.' + old_config_key`
        title (str): If specified, will create an argument group to collect
            all the config arguments.
        description (str): The description of the argument group.
        sort_keys (bool): Whether or not to sort the config keys
            before registering to the parser? (default :obj:`False`)
    """
    # check the arguments
    if not isinstance(config, Config):
        raise TypeError('`config` is not an instance of `Config`: got {!r}.'.
                        format(config))

    prefix = '{}.'.format(prefix) if prefix else ''

    if description is not None and title is None:
        raise ValueError('`title` is required when `description` is specified.')

    # create the group if necessary
    if title is not None:
        target = parser.add_argument_group(title=title, description=description)
    else:
        target = parser

    # populate the arguments
    cls = config.__class__
    keys = list(config)
    if sort_keys:
        keys.sort()

    for key in keys:
        cls_value = getattr(cls, key, None)
        default_value = cls_value

        if isinstance(cls_value, ConfigField):
            config_help = cls_value.description or ''
            if config_help:
                config_help += ' '
            config_help += '(default {}'.format(cls_value.default_value)
            if cls_value.choices:
                config_help += '; choices {}'.format(sorted(cls_value.choices))
            config_help += ')'
            default_value = cls_value.default_value
        else:
            config_help = '(default {})'.format(cls_value)

        target.add_argument(
            '--{}{}'.format(prefix, key), help=config_help,
            action=_ConfigAction,  default=default_value,
            config_obj=config, config_key=key,
        )


@contextmanager
def scoped_set_config(config, **kwargs):
    """
    Set config values within a context scope.

    Args:
        config (Config): The config object to set.
        \\**kwargs: The key-value pairs.
    """
    keys_to_restore = {}
    keys_to_delete = set()

    try:
        for key, value in six.iteritems(kwargs):
            to_delete = False
            old_value = None

            if key in config:
                old_value = config[key]
            else:
                to_delete = True

            # the following stmt will also validate the `key`.
            config[key] = value

            if to_delete:
                keys_to_delete.add(key)
            else:
                keys_to_restore[key] = old_value

        yield

    finally:
        for key in keys_to_delete:
            delattr(config, key)
        for key, value in six.iteritems(keys_to_restore):
            config[key] = value

