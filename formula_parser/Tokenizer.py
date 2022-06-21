from __future__ import annotations
from typing import List, Tuple


# TODO: Rethink
class Tokenizer:

    def __init__(self, recognised_symbols: List[Symbol] | None = None ):
        self.symbols: List[Symbol] = recognised_symbols if recognised_symbols is not None else []
        self.symbols.sort(key=lambda sym: sym.priority)

    def register(self, s: Symbol) -> Tokenizer:
        self.symbols.append(s)
        self.symbols.sort(key=lambda sym: sym.priority)
        return self

    def tokenize(self, expr: str) -> Tuple[bool, List[str]]:
        words = expr.split(' ')
        symbols = []

        if len(words) == 1 and len(words[0]) == 0:
            return True, []

        current = words.pop(0)
        popped = True
        while popped:
            popped = False
            for sym in self.symbols:
                if sym.match(current):
                    current, symbol = sym.pop(current)
                    symbols.append(symbol)
                    print(f"found symbol [{symbol}] (current={current})")
                    popped = True
                    if len(current) == 0:
                        if len(words) > 0:
                            current = words.pop(0)
                        else:
                            return True, symbols

        return False, []


class Symbol:

    def __init__(self, symbol_repr: str, distinct: bool = False, priority: float = 0):
        self.repr: str = symbol_repr
        self.distinct: bool = distinct
        self.priority: float = priority

    def match(self, word: str) -> bool:
        if self.distinct:
            return word.startswith(self.repr)
        else:
            return word == self.repr

    def pop(self, word: str) -> Tuple[str, str]:    # (word, symbol)
        if not self.match(word):
            return word, ""

        if self.distinct:
            return word[len(self.repr):], word[:len(self.repr)]
        else:
            return "", word
