"""Generator registry for Forge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from forge.generators.base import BaseFeatureGenerator

T = TypeVar("T", bound="BaseFeatureGenerator")

# Global registry instance
_REGISTRY: GeneratorRegistry | None = None


class GeneratorRegistry:
    """Registry for feature generators.

    Provides a central place to register and retrieve feature generators
    by name. Generators can be registered with metadata like category
    and description.

    Example:
        >>> registry = get_registry()
        >>> registry.register("log_transform", LogTransformer, category="numeric")
        >>> generator_cls = registry.get("log_transform")
        >>> generator = generator_cls()
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._generators: dict[str, type[BaseFeatureGenerator]] = {}
        self._metadata: dict[str, dict[str, str]] = {}

    def register(
        self,
        name: str,
        generator_class: type[T],
        category: str = "general",
        description: str = ""
    ) -> type[T]:
        """Register a generator class.

        Args:
            name: Unique name for the generator.,
            generator_class: The generator class to register.,
            category: Category for grouping (numeric, categorical, etc).
            description: Human-readable description.,

        Returns:
            The registered class (for decorator usage).

        Raises:
            ValueError: If name is already registered.
        """
        if name in self._generators:
            raise ValueError(f"Generator '{name}' is already registered")

        self._generators[name] = generator_class
        self._metadata[name] = {
            "category": category,
            "description": description,
        }

        return generator_class

    def register_decorator(
        self,
        name: str,
        category: str = "general",
        description: str = ""
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register a generator class.

        Args:
            name: Unique name for the generator.,
            category: Category for grouping.,
            description: Human-readable description.,

        Returns:
            Decorator function.

        Example:
            >>> @registry.register_decorator("my_generator", category="numeric")
            ... class MyGenerator(BaseFeatureGenerator):
            ...     pass
        """

        def decorator(cls: type[T]) -> type[T]:
            return self.register(name, cls, category, description)

        return decorator

    def get(self, name: str) -> type[BaseFeatureGenerator]:
        """Get a registered generator class.

        Args:
            name: Name of the generator.,

        Returns:
            The generator class.

        Raises:
            KeyError: If generator is not registered.
        """
        if name not in self._generators:
            available = list(self._generators.keys())
            raise KeyError(
                f"Generator '{name}' not found. Available: {available}"
            )
        return self._generators[name]

    def get_metadata(self, name: str) -> dict[str, str]:
        """Get metadata for a registered generator.

        Args:
            name: Name of the generator.,

        Returns:
            Metadata dictionary.
        """
        return self._metadata.get(name, {})

    def list_generators(self, category: str | None = None) -> list[str]:
        """List registered generator names.

        Args:
            category: Optional category to filter by.,

        Returns:
            List of generator names.
        """
        if category is None:
            return list(self._generators.keys())

        return [
            name
            for name, meta in self._metadata.items()
            if meta.get("category") == category
        ]

    def list_categories(self) -> list[str]:
        """List all registered categories.

        Returns:
            List of unique category names.
        """
        categories = set()
        for meta in self._metadata.values():
            if "category" in meta:
                categories.add(meta["category"])
        return sorted(categories)

    def create(self, name: str, **kwargs) -> BaseFeatureGenerator:
        """Create an instance of a registered generator.

        Args:
            name: Name of the generator.
            **kwargs: Arguments to pass to the generator constructor.

        Returns:
            Generator instance.
        """
        generator_cls = self.get(name)
        return generator_cls(**kwargs)

    def is_registered(self, name: str) -> bool:
        """Check if a generator is registered.

        Args:
            name: Name to check.,

        Returns:
            True if registered.
        """
        return name in self._generators

    def unregister(self, name: str) -> None:
        """Unregister a generator.

        Args:
            name: Name of the generator to remove.
        """
        self._generators.pop(name, None)
        self._metadata.pop(name, None)

    def clear(self) -> None:
        """Clear all registered generators."""
        self._generators.clear()
        self._metadata.clear()


def get_registry() -> GeneratorRegistry:
    """Get the global generator registry.

    Returns:
        The global registry instance.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = GeneratorRegistry()
    return _REGISTRY


def register_generator(
    name: str,
    category: str = "general",
    description: str = ""
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a generator with the global registry.

    Args:
        name: Unique name for the generator.,
        category: Category for grouping.,
        description: Human-readable description.,

    Returns:
        Decorator function.

    Example:
        >>> @register_generator("log_transform", category="numeric")
        ... class LogTransformer(BaseFeatureGenerator):
        ...     pass
    """
    return get_registry().register_decorator(name, category, description)
