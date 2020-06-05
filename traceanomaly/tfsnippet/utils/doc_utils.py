import re

import six

__all__ = [
    'DocInherit', 'append_to_doc', 'append_arg_to_doc',
    'add_name_arg_doc', 'add_name_and_scope_arg_doc',
]


class DocStringInheritor(type):
    """
    Meta-class for automatically inherit docstrings from base classes.
    """

    def __new__(kclass, name, bases, dct):
        def iter_mro():
            for base in bases:
                for mro in base.__mro__:
                    yield mro

        # inherit the class docstring
        if not dct.get('__doc__', None):
            for cls in iter_mro():
                cls_doc = getattr(cls, '__doc__', None)
                if cls_doc:
                    dct['__doc__'] = cls_doc
                    break

        # inherit the method docstrings
        for key in dct:
            attr = dct[key]
            if attr is not None and not getattr(attr, '__doc__', None):
                for cls in iter_mro():
                    cls_attr = getattr(cls, key, None)
                    if cls_attr:
                        cls_doc = getattr(cls_attr, '__doc__', None)
                        if cls_doc:
                            if isinstance(attr, property) and six.PY2:
                                # In Python 2.x, "__doc__" of a property
                                # is read-only.  We choose to wrap the
                                # original property in a new property.
                                dct[key] = property(
                                    fget=attr.fget,
                                    fset=attr.fset,
                                    fdel=attr.fdel,
                                    doc=cls_doc
                                )
                            else:
                                attr.__doc__ = cls_doc
                            break

        return super(DocStringInheritor, kclass). \
            __new__(kclass, name, bases, dct)


def DocInherit(kclass):
    """
    Class decorator to enable `kclass` and all its sub-classes to
    automatically inherit docstrings from base classes.

    Usage:

    .. code-block:: python

        import six


        @DocInherit
        class Parent(object):
            \"""Docstring of the parent class.\"""

            def some_method(self):
                \"""Docstring of the method.\"""
                ...

        class Child(Parent):
            # inherits the docstring of :meth:`Parent`

            def some_method(self):
                # inherits the docstring of :meth:`Parent.some_method`
                ...

    Args:
        kclass (Type): The class to decorate.

    Returns:
        The decorated class.
    """
    return six.add_metaclass(DocStringInheritor)(kclass)


def append_to_doc(doc, content):
    """
    Append content to the doc string.

    Args:
        doc (str): The original doc string.
        content (str): The new doc string, which should be a standalone section.

    Returns:
        str: The modified doc string.
    """
    content = '\n'.join(l.rstrip() for l in content.split('\n'))
    content = content.lstrip('\n')
    content = content.rstrip('\n')

    # fast path: doc is empty, just wrap the content
    if not doc:
        # the empty line before the content might be required for the
        # sphinx to correctly parse the section.
        if not content.startswith('\n'):
            content = '\n' + content
        if not content.endswith('\n'):
            content = content + '\n'
        return content

    # slow path: doc is not empty, parse it
    # find the indent of the doc string.
    indent = 0
    for line in doc.split('\n'):
        if line and line.strip() and line.startswith(' '):
            for c in line:
                if c != ' ':
                    break
                indent += 1
            break
    indent = ' ' * indent

    # compose the docstring
    contents = [doc, '\n']
    if not doc.endswith('\n'):
        contents.append('\n')

    for line in content.split('\n'):
        if not line.strip():
            contents.append('\n')
        else:
            contents.append(indent + line + '\n')

    return ''.join(contents)


def append_arg_to_doc(doc, arg_doc):
    """
    Add the doc for `name` and `scope` argument to the doc string.

    Args:
        doc: The original doc string.
        arg_doc: The argument documentations.

    Returns:
        str: The updated doc string.
    """
    doc = doc or ''
    section_start = re.search(r'^([ ]*)Args:[ ]*$', doc, re.M)

    # case #1: generate an args section
    if not section_start:
        new_doc = '\n'.join(('    ' + l) if l.strip() else ''
                            for l in arg_doc.strip().split('\n'))
        return append_to_doc(doc, 'Args:\n' + new_doc)

    # case #2: add to the args section
    arg_indent = ' ' * (len(section_start.group(1)) + 4)
    arg_start_pos = section_start.end(0)

    arg_len = 0
    for m in re.finditer(r'^.*?$', doc[arg_start_pos + 1:], re.M):
        line = m.group(0)
        if line.rstrip('\r\n') and (not line.startswith(arg_indent) or
                                    re.match(r'^\s*\\?\*', line)):
            break
        if line.strip():
            arg_len = m.end(0) + 1
    arg_end_pos = arg_start_pos + arg_len

    new_doc = '\n'.join((arg_indent + l) if l.strip() else ''
                        for l in arg_doc.split('\n'))
    new_doc = doc[:arg_end_pos].rstrip() + new_doc + doc[arg_end_pos:]
    if not new_doc.endswith('\n'):
        new_doc += '\n'

    return new_doc


def add_name_arg_doc(method):
    """
    Add `name` argument to the doc of `method`.
    """
    arg_doc = '''
name (str): Default name of the name scope.
    If not specified, generate one according to the method name.'''
    method.__doc__ = append_arg_to_doc(method.__doc__, arg_doc)
    return method


def add_name_and_scope_arg_doc(method):
    """
    Add `name` and `scope` argument to the doc of `method`.
    """
    arg_doc = '''
name (str): Default name of the variable scope.  Will be uniquified.
    If not specified, generate one according to the class name.
scope (str): The name of the variable scope.'''
    method.__doc__ = append_arg_to_doc(method.__doc__, arg_doc)
    return method
