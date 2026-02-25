"""
Fuzzy Menu Components for GAAP CLI
==================================

Provides fuzzy-search selection menus using questionary with graceful fallback.

Reference: docs/evolution_plan_2026/44_CLI_AUDIT_SPEC.md
"""

from dataclasses import dataclass
from typing import Any, Sequence

from rich.console import Console
from rich.prompt import Prompt

console = Console()

try:
    import questionary
    from questionary import Style

    HAS_QUESTIONARY = True
    CUSTOM_STYLE = Style(
        [
            ("qmark", "fg:cyan bold"),
            ("question", "bold"),
            ("answer", "fg:green bold"),
            ("pointer", "fg:cyan bold"),
            ("highlighted", "fg:cyan bold"),
            ("selected", "fg:green"),
            ("separator", "fg:gray"),
            ("instruction", "fg:gray"),
            ("text", ""),
        ]
    )
except ImportError:
    HAS_QUESTIONARY = False
    CUSTOM_STYLE = None


@dataclass
class MenuItem:
    label: str
    value: Any
    description: str | None = None


class FuzzyMenu:
    """
    Fuzzy-search menu for interactive selection.
    Falls back to basic input if questionary is not installed.
    """

    def __init__(self, use_fuzzy: bool = True):
        self.use_fuzzy = use_fuzzy and HAS_QUESTIONARY

    def select_provider(self, providers: Sequence[dict[str, Any]]) -> str | None:
        """
        Select a provider with fuzzy search.

        Args:
            providers: List of provider dicts with 'name', 'type', 'status' keys

        Returns:
            Selected provider name or None
        """
        items = []
        for p in providers:
            status_icon = "🟢" if p.get("enabled", False) else "🔴"
            label = f"{status_icon} {p['name']} ({p.get('provider_type', 'unknown')})"
            items.append(MenuItem(label=label, value=p["name"], description=p.get("status")))

        return self.select_from_list(items, title="Select Provider")

    def select_tool(self, tools: Sequence[dict[str, Any]]) -> str | None:
        """
        Select a tool with fuzzy search.

        Args:
            tools: List of tool dicts with 'name', 'description', 'category' keys

        Returns:
            Selected tool name or None
        """
        items = []
        for t in tools:
            category = t.get("category", "general")
            label = f"[{category}] {t['name']}"
            items.append(MenuItem(label=label, value=t["name"], description=t.get("description")))

        return self.select_from_list(items, title="Select Tool")

    def select_from_list(
        self, items: Sequence[MenuItem], title: str = "Select an option"
    ) -> Any | None:
        """
        Generic fuzzy selection from a list.

        Args:
            items: Sequence of MenuItem objects
            title: Title for the selection prompt

        Returns:
            Selected item value or None if cancelled
        """
        if not items:
            console.print("[yellow]No items to select from.[/yellow]")
            return None

        if self.use_fuzzy:
            return self._fuzzy_select(items, title)
        else:
            return self._fallback_select(items, title)

    def _fuzzy_select(self, items: Sequence[MenuItem], title: str) -> Any | None:
        """Use questionary for fuzzy selection."""
        choices = []
        for item in items:
            choice_display = item.label
            if item.description:
                choice_display = f"{item.label} - {item.description}"
            choices.append(choice_display)

        try:
            result = questionary.select(
                title,
                choices=choices,
                style=CUSTOM_STYLE,
                use_arrow_keys=True,
                use_jk_keys=True,
                use_search_filter=True,
            ).ask()

            if result is None:
                return None

            idx = choices.index(result)
            return items[idx].value
        except KeyboardInterrupt:
            return None
        except Exception as e:
            console.print(f"[yellow]Fuzzy selection failed: {e}. Using fallback.[/yellow]")
            return self._fallback_select(items, title)

    def _fallback_select(self, items: Sequence[MenuItem], title: str) -> Any | None:
        """Fallback to numbered list selection."""
        console.print(f"\n[bold cyan]{title}[/bold cyan]\n")

        for i, item in enumerate(items, 1):
            desc = f" - {item.description}" if item.description else ""
            console.print(f"  [cyan]{i:>2}[/cyan]. {item.label}{desc}")

        console.print()

        try:
            choice = Prompt.ask(
                "Enter number",
                default="1",
                show_default=True,
            )

            if choice.lower() in ("q", "quit", "exit", "cancel"):
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx].value
            else:
                console.print("[red]Invalid selection.[/red]")
                return None
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
            return None
        except KeyboardInterrupt:
            return None

    def multi_select(self, items: Sequence[MenuItem], title: str = "Select options") -> list[Any]:
        """
        Multi-selection with checkboxes.

        Args:
            items: Sequence of MenuItem objects
            title: Title for the selection prompt

        Returns:
            List of selected item values
        """
        if not items:
            console.print("[yellow]No items to select from.[/yellow]")
            return []

        if self.use_fuzzy:
            return self._fuzzy_multi_select(items, title)
        else:
            return self._fallback_multi_select(items, title)

    def _fuzzy_multi_select(self, items: Sequence[MenuItem], title: str) -> list[Any]:
        """Use questionary for multi-selection."""
        choices = []
        for item in items:
            choice_display = item.label
            if item.description:
                choice_display = f"{item.label} - {item.description}"
            choices.append({"name": choice_display, "value": item.value})

        try:
            results = questionary.checkbox(
                title,
                choices=choices,
                style=CUSTOM_STYLE,
                use_arrow_keys=True,
                use_jk_keys=True,
                use_search_filter=True,
            ).ask()

            if results is None:
                return []

            return list(results)
        except KeyboardInterrupt:
            return []
        except Exception as e:
            console.print(f"[yellow]Multi-selection failed: {e}. Using fallback.[/yellow]")
            return self._fallback_multi_select(items, title)

    def _fallback_multi_select(self, items: Sequence[MenuItem], title: str) -> list[Any]:
        """Fallback to comma-separated list selection."""
        console.print(f"\n[bold cyan]{title}[/bold cyan]\n")
        console.print("[dim]Enter numbers separated by commas (e.g., 1,3,5)[/dim]\n")

        for i, item in enumerate(items, 1):
            desc = f" - {item.description}" if item.description else ""
            console.print(f"  [cyan]{i:>2}[/cyan]. {item.label}{desc}")

        console.print()

        try:
            choice = Prompt.ask(
                "Enter numbers",
                default="",
                show_default=False,
            )

            if not choice or choice.lower() in ("q", "quit", "exit", "cancel", "none"):
                return []

            selected = []
            for part in choice.split(","):
                part = part.strip()
                if part:
                    idx = int(part) - 1
                    if 0 <= idx < len(items):
                        selected.append(items[idx].value)

            return selected
        except ValueError:
            console.print("[red]Please enter valid numbers separated by commas.[/red]")
            return []
        except KeyboardInterrupt:
            return []

    def confirm(self, message: str, default: bool = True) -> bool:
        """
        Confirmation prompt.

        Args:
            message: Confirmation message
            default: Default value

        Returns:
            True if confirmed, False otherwise
        """
        if self.use_fuzzy:
            try:
                return (
                    questionary.confirm(message, default=default, style=CUSTOM_STYLE).ask() or False
                )
            except Exception:
                pass

        try:
            result = Prompt.ask(
                f"{message} [y/N]",
                default="y" if default else "n",
            )
            return result.lower() in ("y", "yes", "true", "1")
        except KeyboardInterrupt:
            return False

    def text_input(self, message: str, default: str = "") -> str | None:
        """
        Text input prompt.

        Args:
            message: Input prompt message
            default: Default value

        Returns:
            Entered text or None if cancelled
        """
        if self.use_fuzzy:
            try:
                result = questionary.text(message, default=default, style=CUSTOM_STYLE).ask()
                return result
            except Exception:
                pass

        try:
            return Prompt.ask(message, default=default) or None
        except KeyboardInterrupt:
            return None


def select_provider(providers: Sequence[dict[str, Any]]) -> str | None:
    """Convenience function to select a provider."""
    menu = FuzzyMenu()
    return menu.select_provider(providers)


def select_tool(tools: Sequence[dict[str, Any]]) -> str | None:
    """Convenience function to select a tool."""
    menu = FuzzyMenu()
    return menu.select_tool(tools)


def select_from_list(items: Sequence[MenuItem], title: str = "Select an option") -> Any | None:
    """Convenience function for generic selection."""
    menu = FuzzyMenu()
    return menu.select_from_list(items, title)


def multi_select(items: Sequence[MenuItem], title: str = "Select options") -> list[Any]:
    """Convenience function for multi-selection."""
    menu = FuzzyMenu()
    return menu.multi_select(items, title)
