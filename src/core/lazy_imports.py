"""
Lazy import system for improving startup performance
Author: Drmusab
Last Modified: 2025-08-10

This module provides a lazy import system that defers expensive imports
until they are actually needed, significantly improving startup time.
"""

import importlib
import threading
import weakref
from functools import wraps
from typing import Any, Dict, Optional, Set, Union


class LazyImporter:
    """
    Lazy import manager that loads modules only when they are first accessed.
    
    This significantly improves startup time by deferring expensive imports
    until they are actually needed.
    """
    
    def __init__(self):
        self._imported_modules: Dict[str, Any] = {}
        self._import_lock = threading.Lock()
        self._pending_imports: Set[str] = set()
        
    def lazy_import(self, module_name: str, attribute_name: str = None) -> Any:
        """
        Create a lazy import proxy for a module or module attribute.
        
        Args:
            module_name: Name of the module to import
            attribute_name: Optional attribute name within the module
            
        Returns:
            Lazy proxy object that will import on first access
        """
        cache_key = f"{module_name}.{attribute_name}" if attribute_name else module_name
        
        if cache_key in self._imported_modules:
            return self._imported_modules[cache_key]
            
        proxy = LazyProxy(self, module_name, attribute_name, cache_key)
        return proxy
    
    def _do_import(self, module_name: str, attribute_name: str = None, cache_key: str = None) -> Any:
        """Perform the actual import operation."""
        if cache_key and cache_key in self._imported_modules:
            return self._imported_modules[cache_key]
            
        with self._import_lock:
            # Double-check pattern
            if cache_key and cache_key in self._imported_modules:
                return self._imported_modules[cache_key]
                
            try:
                self._pending_imports.add(module_name)
                module = importlib.import_module(module_name)
                
                if attribute_name:
                    result = getattr(module, attribute_name)
                else:
                    result = module
                    
                if cache_key:
                    self._imported_modules[cache_key] = result
                    
                return result
                
            finally:
                self._pending_imports.discard(module_name)
    
    def is_imported(self, module_name: str) -> bool:
        """Check if a module has been imported."""
        return module_name in self._imported_modules
    
    def get_pending_imports(self) -> Set[str]:
        """Get list of imports currently being loaded."""
        return self._pending_imports.copy()
    
    def clear_cache(self) -> None:
        """Clear the import cache."""
        with self._import_lock:
            self._imported_modules.clear()


class LazyProxy:
    """
    Proxy object that performs lazy import on first access.
    """
    
    def __init__(self, importer: LazyImporter, module_name: str, 
                 attribute_name: str = None, cache_key: str = None):
        object.__setattr__(self, '_importer', importer)
        object.__setattr__(self, '_module_name', module_name)
        object.__setattr__(self, '_attribute_name', attribute_name)
        object.__setattr__(self, '_cache_key', cache_key)
        object.__setattr__(self, '_resolved', None)
        object.__setattr__(self, '_import_lock', threading.Lock())
    
    def _resolve(self):
        """Resolve the lazy import."""
        if object.__getattribute__(self, '_resolved') is not None:
            return object.__getattribute__(self, '_resolved')
            
        with object.__getattribute__(self, '_import_lock'):
            # Double-check pattern
            if object.__getattribute__(self, '_resolved') is not None:
                return object.__getattribute__(self, '_resolved')
                
            importer = object.__getattribute__(self, '_importer')
            module_name = object.__getattribute__(self, '_module_name')
            attribute_name = object.__getattribute__(self, '_attribute_name')
            cache_key = object.__getattribute__(self, '_cache_key')
            
            resolved = importer._do_import(module_name, attribute_name, cache_key)
            object.__setattr__(self, '_resolved', resolved)
            return resolved
    
    def __getattr__(self, name):
        resolved = self._resolve()
        return getattr(resolved, name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            resolved = self._resolve()
            setattr(resolved, name, value)
    
    def __call__(self, *args, **kwargs):
        resolved = self._resolve()
        return resolved(*args, **kwargs)
    
    def __str__(self):
        resolved = self._resolve()
        return str(resolved)
    
    def __repr__(self):
        resolved = self._resolve()
        return repr(resolved)
    
    def __bool__(self):
        resolved = self._resolve()
        return bool(resolved)


# Global lazy importer instance
_lazy_importer = LazyImporter()


def lazy_import(module_name: str, attribute_name: str = None) -> Any:
    """
    Create a lazy import for a module or module attribute.
    
    Usage:
        # Lazy import entire module
        numpy = lazy_import('numpy')
        
        # Lazy import specific attribute
        HTTPStatus = lazy_import('http', 'HTTPStatus')
        
    Args:
        module_name: Name of the module to import
        attribute_name: Optional attribute name within the module
        
    Returns:
        Lazy proxy that will import on first access
    """
    return _lazy_importer.lazy_import(module_name, attribute_name)


def lazy_import_decorator(dependencies: Dict[str, Union[str, tuple]]):
    """
    Decorator that lazy loads dependencies for a function or class.
    
    Usage:
        @lazy_import_decorator({
            'np': 'numpy',
            'HTTPStatus': ('http', 'HTTPStatus')
        })
        def my_function():
            return np.array([1, 2, 3])
    """
    def decorator(func_or_class):
        # Create lazy imports
        lazy_deps = {}
        for name, spec in dependencies.items():
            if isinstance(spec, tuple):
                module_name, attr_name = spec
                lazy_deps[name] = lazy_import(module_name, attr_name)
            else:
                lazy_deps[name] = lazy_import(spec)
        
        if isinstance(func_or_class, type):
            # Class decorator
            for name, lazy_obj in lazy_deps.items():
                setattr(func_or_class, name, lazy_obj)
            return func_or_class
        else:
            # Function decorator
            @wraps(func_or_class)
            def wrapper(*args, **kwargs):
                # Inject lazy imports into function globals
                func_globals = func_or_class.__globals__
                old_values = {}
                
                try:
                    for name, lazy_obj in lazy_deps.items():
                        if name in func_globals:
                            old_values[name] = func_globals[name]
                        func_globals[name] = lazy_obj
                    
                    return func_or_class(*args, **kwargs)
                finally:
                    # Restore original globals
                    for name, lazy_obj in lazy_deps.items():
                        if name in old_values:
                            func_globals[name] = old_values[name]
                        else:
                            func_globals.pop(name, None)
            
            return wrapper
    
    return decorator


def is_imported(module_name: str) -> bool:
    """Check if a module has been lazily imported."""
    return _lazy_importer.is_imported(module_name)


def get_pending_imports() -> Set[str]:
    """Get list of imports currently being loaded."""
    return _lazy_importer.get_pending_imports()


def clear_import_cache() -> None:
    """Clear the lazy import cache."""
    _lazy_importer.clear_cache()


# Commonly used lazy imports for the AI Assistant
def get_common_lazy_imports() -> Dict[str, Any]:
    """Get commonly used lazy imports for the AI Assistant."""
    return {
        # Core async/network libraries
        'aiohttp': lazy_import('aiohttp'),
        'websockets': lazy_import('websockets'),
        'redis': lazy_import('redis'),
        
        # AI/ML libraries
        'torch': lazy_import('torch'),
        'numpy': lazy_import('numpy'),
        
        # Data processing
        'pandas': lazy_import('pandas'),
        'json': lazy_import('json'),
        
        # Observability
        'prometheus_client': lazy_import('prometheus_client'),
        'opentelemetry': lazy_import('opentelemetry'),
        
        # Database
        'sqlalchemy': lazy_import('sqlalchemy'),
        'alembic': lazy_import('alembic'),
        
        # API frameworks
        'fastapi': lazy_import('fastapi'),
        'uvicorn': lazy_import('uvicorn'),
    }