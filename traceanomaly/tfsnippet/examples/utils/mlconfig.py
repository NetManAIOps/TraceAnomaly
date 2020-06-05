import codecs
import json
import os
import re
from pprint import pformat

import click
import six
import yaml
from click import pass_context
from lazy_object_proxy import Proxy
from yaml.parser import ParserError

__all__ = [
    'MLConfig', 'get_global_config', 'global_config', 'set_global_config',
    'pass_global_config', 'config_options'
]

CONFIG_STRING_PATTERN = re.compile(
    r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$')


def is_config_attr(cls_or_instance, key):
    return (not key.startswith('_') and hasattr(cls_or_instance, key) and
            not hasattr(MLConfig, key))


class MLConfig(object):
    """
    Base class of configuration values for machine learning experiments.

    Derive sub-classes of :class:`MLConfig`, and define configuration
    values as public class attributes.  These attributes can then be
    set via configuration parsers.  For example::

        class YourConfig(MLConfig):
            max_epoch = 100
            learning_rate = 0.01
            activation = 'leaky_relu'
            l2_regularization = 0.001

        config = YourConfig()
        config.parse_strings([
            'max_epoch=200',
            'learning_rate=0.1'
        ])
        config.l2_regularization = 0.0005


    Only the attributes already defined in classes can be assigned new
    values.  Assigning values to non-exist attributes will cause an error::

        config.non_exist_attr = 123  # will raise AttributeError
    """

    def __init__(self, **kwargs):
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)

    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError('Config key {!r} does not exist.'.format(key))
        object.__setattr__(self, key, value)

    def has_config_key(self, key):
        """
        Check whether or not the configuration key exists.

        Args:
            key (str): The key to check.

        Returns:
            bool: A boolean indicating whether the key exists.
        """
        return is_config_attr(self, key)

    @classmethod
    def defaults(cls):
        """
        Get the default configuration values of this class as a dict.

        Returns:
            dict[str, any]: The default configuration values of this class.
        """
        return {k: getattr(cls, k)
                for k in dir(cls)
                if is_config_attr(cls, k)}

    def overrides(self):
        """
        Get the overrided values (with respect to the default values) as a dict.

        Returns:
            dict[str, any]: The overrided values.
        """
        return {k: v
                for k, v in six.iteritems(self.__dict__)
                if is_config_attr(self, k)}

    def to_dict(self):
        """
        Get the configuration values of this config object as a dict.

        Returns:
            dict[str, any]: The configuration values of this config object.
        """
        return {k: getattr(self, k)
                for k in dir(self)
                if is_config_attr(self, k)}

    def parse_dict(self, d):
        """
        Parse configuration values from the given dict `d`.

        Args:
            d (dict[str, any]): The configuration values to be set.
        """
        for k, v in six.iteritems(dict(d)):
            setattr(self, k, v)

    def parse_file(self, path):
        """
        Parse configuration values form the given file.

        Args:
            path (str): Path of the file.  It should be a JSON file or
                a YAML file, with corresponding file extension.
        """
        _, ext = os.path.splitext(path)
        if ext == '.json':
            with codecs.open(path, 'rb', 'utf-8') as f:
                self.parse_dict(json.load(f))
        elif ext in ('.yml', '.yaml'):
            with codecs.open(path, 'rb', 'utf-8') as f:
                self.parse_dict(yaml.load(f))
        else:
            raise ValueError('Config file of this type is not supported: {}'.
                             format(path))

    def parse_strings(self, strings):
        """
        Parse configuration values from the given `strings`.

        Args:
            strings (Iterable[str]): Configuration strings in the format
                ``KEY=VALUE``, where ``VALUE`` should be valid YAML literal.
        """
        for s in strings:
            m = CONFIG_STRING_PATTERN.match(s)
            if not m:
                raise ValueError('Not a valid configuration string: {!r}'.
                                 format(s))
            key = m.group(1)
            value = yaml.load(m.group(2))
            setattr(self, key, value)

    def format_config(self):
        """
        Format the configuration values as string.

        Returns:
            str: The formatted configuration values.
        """
        return pformat(self.to_dict())


_global_config = None


def get_global_config():
    """
    Get the global config object.

    You may set the global config object by :func:`set_global_config`.

    Returns:
        The global config object.
    """
    return _global_config


global_config = Proxy(get_global_config)
"""
The proxy object to the global config.

Usage::

    from tfsnippet.examples.utils import MLConfig, global_config, set_global_config

    class YourConfig(MLConfig):
        max_epoch = 100

    set_global_config(YourConfig())  # you may also use click + config_options

    assert(isinstance(global_config, YourConfig))
    assert(global_config.max_epoch == 100)
"""


def set_global_config(config):
    """
    Set the global config object.

    Args:
        config: The config object.
    """
    global _global_config
    _global_config = config


def pass_global_config(method):
    """
    Decorate `method`, such that it will receive the global config object
    as its first argument.  For example::

        @pass_global_config
        def main(config):
            assert(config is get_global_config())

    A simpler way is to use the ``tfsnippet.examples.utils.global_config``
    proxy object, which acts as if it is a regular config object.

    Args:
        method: The method to be decorated.

    Returns:
        The decorated method.
    """

    @six.wraps(method)
    def wrapped(*args, **kwargs):
        return method(get_global_config(), *args, **kwargs)

    return wrapped


def config_options(cls):
    """
    Generate click options for parsing configuration values.

    This will generate the ``-c``, ``--config`` and ``--print-config`` options.
    The parsed configuration object will be used as the global config object.
    You may retrieve the config object by calling :func:`get_global_config`,
    by using the proxy ``tfsnippet.examples.utils.global_config``, or by
    :func:`pass_global_config` decorator.

    .. code-block:: python

        from tfsnippet.examples.utils import global_config as config

        class YourConfig(MLConfig):
            max_epoch = 100

        @click.command()
        @click.option('--your-own-option', ...)
        @click.argument('your-own-arg', ...)
        @config_options(YourConfig)
        def main(your_own_option, your_own_arg):
            print(config.max_epoch)


    Args:
        cls (class): The configuration class, a sub-class of :class:`MLConfig`.

    Returns:
        The click options decorator.
    """

    def ensure_ctx_config(ctx):
        ctx.ensure_object(dict)
        if 'config' not in ctx.obj:
            ctx.obj['config'] = cls()

    def parse_config(ctx, param, value):
        if not value or ctx.resilient_parsing:
            return
        ensure_ctx_config(ctx)
        try:
            ctx.obj['config'].parse_strings(value)
        except (AttributeError, ValueError, TypeError, ParserError) as e:
            raise click.BadParameter(str(e))

    def parse_config_file(ctx, param, value):
        if not value or ctx.resilient_parsing:
            return
        ensure_ctx_config(ctx)
        for v in value:
            try:
                ctx.obj['config'].parse_file(v)
            except Exception as e:
                raise click.BadParameter(str(e))

    def mlstorage_protocol(config):
        # The ``env["MLSTORAGE_EXPERIMENT_ID"]`` would be set if the program
        # is run via `mlrun` from MLStorage.  See
        # `MLStorage Server <https://github.com/haowen-xu/mlstorage-server>`_
        # and
        # `MLStorage Client <https://github.com/haowen-xu/mlstorage-client>`_
        # for details.
        if os.environ.get('MLSTORAGE_EXPERIMENT_ID'):
            # save default config values to "config.defaults.json"
            default_config_path = os.path.abspath(
                os.path.join(os.getcwd(), 'config.defaults.json'))
            default_config_json = json.dumps(config.defaults())
            with codecs.open(default_config_path, 'wb', 'utf-8') as f:
                f.write(default_config_json)

            # load user specified config from "config.json"
            config_path = os.path.abspath(
                os.path.join(os.getcwd(), 'config.json'))
            if os.path.isfile(config_path):
                with codecs.open(config_path, 'rb', 'utf-8') as f:
                    config_dict = json.load(f)
                if not isinstance(config_dict, dict):
                    raise ValueError('%s: expected a config dict, but got '
                                     '%r'.format(config_path, config_dict))
                config.parse_dict(config_dict)

            # save user specified config into "config.json"
            config_json = json.dumps(config.overrides())
            with codecs.open(config_path, 'wb', 'utf-8') as f:
                f.write(config_json)

    def wrapper(method):
        @click.option('-c', '--config',
                      help='Set a configuration value.',
                      metavar='KEY=VALUE', type=str, multiple=True,
                      expose_value=False, callback=parse_config)
        @click.option('-C', '--config-file',
                      help='Load a JSON or YAML configuration file.',
                      metavar='PATH', type=str, multiple=True,
                      expose_value=False, callback=parse_config_file)
        @click.option('--print-config',
                      help='Print configuration values and exit.',
                      is_flag=True, required=False, default=False)
        @pass_context
        @six.wraps(method)
        def wrapped(ctx, print_config, **kwargs):
            ensure_ctx_config(ctx)

            if print_config:
                click.echo(ctx.obj['config'].format_config())
                ctx.exit()
            else:
                old_config = get_global_config()
                try:
                    config = ctx.obj['config']
                    set_global_config(config)
                    mlstorage_protocol(config)
                    return method(**kwargs)
                finally:
                    set_global_config(old_config)

        return wrapped

    return wrapper
