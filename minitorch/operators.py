"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: first number.
        y: second number.

    Returns:
    -------
        The product of the two numbers.

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x: number

    Returns:
    -------
        The number itself.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x: first number.
        y: second number.

    Returns:
    -------
        The sum of the two numbers.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x: number.

    Returns:
    -------
        The negative of the number.

    """
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Less than between two numbers.

    Args:
    ----
        x: first number.
        y: second number

    Returns:
    -------
        True if x is less than y, False otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checking if two numbers are equal.

    Args:
    ----
        x: first number.
        y: second number.

    Returns:
    -------
        True if x is equal to y, False otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Maximum of the two numbers.

    Args:
    ----
        x: first number.
        y: second number.

    Returns:
    -------
        The maximum of the two numbers.

    """
    if x > y:
        return x
    return y


def is_close(x: float, y: float) -> float:
    """Checking if two numbers are close.

    Args:
    ----
        x: first number.
        y: second number.

    Returns:
    -------
        True if x is close to y, False otherwise.

    """
    return abs(x - y) < (10**-2)


def sigmoid(x: float) -> float:
    """Sigmoid function.

    Args:
    ----
        x: number.

    Returns:
    -------
        The sigmoid of the number.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """ReLU function.

    Args:
    ----
        x: number.

    Returns:
    -------
        The ReLU of the number.

    """
    return max(0.0, x)


def log(x: float) -> float:
    """Logarithm function.

    Args:
    ----
        x: number.

    Returns:
    -------
        The logarithm of the number.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function.

    Args:
    ----
        x: number.

    Returns:
    -------
        The exponential of the number.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Inverse function.

    Args:
    ----
        x: number.

    Returns:
    -------
        The inverse of the number.

    """
    return 1 / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
    ----
        x: number.
        y: number.

    Returns:
    -------
        The derivative of log times a second arg.

    """
    return -y / x**2


def log_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
    ----
        x: number.
        y: number.

    Returns:
    -------
        The derivative of reciprocal times a second arg.

    """
    return y / x


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
    ----
        x: number.
        y: number.

    Returns:
    -------
        The derivative of ReLU times a second arg.

    """
    dxdy: float = 1 if x > 0 else 0
    return dxdy * y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(
    func: Callable[[float], float],
) -> Callable[[Iterable[float]], Iterable[float]]:
    """Map a function over an iterable.

    Args:
    ----
        func: Function to apply.

    Returns:
    -------
        A function that maps fn over a list and returns a new list.

    """

    def m(ls: Iterable[float]) -> Iterable[float]:
        return [func(item) for item in ls]

    return m


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate an iterable.

    Args:
    ----
        ls: iterable of numbers.

    Returns:
    -------
        The negated iterable.

    """
    m: Callable[[Iterable[float]], Iterable[float]] = map(neg)
    return m(ls)


def zipWith(
    func: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Zip two iterables together.

    Args:
    ----
        func: function to apply.

    Returns:
    -------
        A function that zips two iterables together.

    """

    def z(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        result = []
        iterator1 = iter(ls1)
        iteraror2 = iter(ls2)

        while True:
            try:
                item1 = next(iterator1)
                item2 = next(iteraror2)
            except Exception:
                break
            result.append(func(item1, item2))
        return result

    return z


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists together.

    Args:
    ----
        ls1: first list.
        ls2: second list.

    Returns:
    -------
        The added list.

    """
    z: Callable[[Iterable[float], Iterable[float]], Iterable[float]] = zipWith(add)
    return z(ls1, ls2)


def reduce(
    func: Callable[[float, float], float],
    initial: float,
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable.

    Args:
    ----
        func: function to apply.
        initial: initial value.

    Returns:
    -------
        The reduced iterable.

    """

    def r(ls: Iterable[float]) -> float:
        result: float = initial
        for item in ls:
            result = func(result, item)
        return result

    return r


def sum(ls: Iterable[float]) -> float:
    """Sum a list.

    Args:
    ----
        ls: list of numbers.

    Returns:
    -------
        The sum of the list.

    """
    r: Callable[[Iterable[float]], float] = reduce(add, 0.0)
    return r(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list.

    Args:
    ----
        ls: list of numbers.

    Returns:
    -------
        The product of the list.

    """
    r: Callable[[Iterable[float]], float] = reduce(mul, 1.0)
    return r(ls)
