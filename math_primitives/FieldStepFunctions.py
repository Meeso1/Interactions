from abc import ABC, abstractmethod
from type_declarations.Types import *
from math_primitives.FieldValueRepr import ValueFunc


class FieldStepMethod(ABC):

    def __init__(self, req_steps: int = 2):
        self.required_steps: int = req_steps

    @staticmethod
    def _last_n(tvs: List[TimedVal], n: int, time: float | None = None) -> List[TimedVal]:
        if len(tvs) == 0:
            return []

        if time is None:
            time = tvs[-1].time

        last_index = len(tvs) - 1
        for tv in reversed(tvs):
            if tv.time > time:
                last_index -= 1
            else:
                break

        if last_index == -1:
            return []
        return tvs[last_index - n + 1:last_index + 1] if last_index - n + 1 >= 0 else tvs[:last_index + 1]

    @abstractmethod
    def __call__(self, derivatives: List[TimedVal[DerivativeFunc]], time: float | None = None) -> ValueFunc:
        pass


class EulerField(FieldStepMethod):

    def __init__(self):
        super(EulerField, self).__init__(req_steps=2)

    @staticmethod
    def euler_field(derivatives: List[TimedVal[DerivativeFunc]], time: float | None = None) -> ValueFunc:

        if time is None:
            time = derivatives[-1].time

        curr = derivatives[1]
        prev = derivatives[0]
        for i in range(len(derivatives) - 1):
            if derivatives[-i - 2].time < time <= derivatives[-i - 1].time:
                curr = derivatives[-i - 1]
                prev = derivatives[-i - 2]
                break

        return lambda x, y: curr.val(x, y) * (curr.time - prev.time)

    def __call__(self, derivatives: List[TimedVal[DerivativeFunc]], time: float | None = None) -> ValueFunc:
        return self.euler_field(derivatives, time)


class Method3212Field(FieldStepMethod):

    def __init__(self):
        super(Method3212Field, self).__init__(req_steps=2)

    @staticmethod
    def method_32_12(derivatives: List[TimedVal[DerivativeFunc]], time: float | None = None) -> ValueFunc:

        vals = Method3212Field._last_n(derivatives, 3, time)
        if len(vals) < 3:
            return EulerField.euler_field(derivatives, time)

        curr = vals[-1]
        prev = vals[-2]

        return (curr.time - prev.time) * (1.5 * curr.val - 0.5 * prev.val)

    def __call__(self, derivatives: List[TimedVal[DerivativeFunc]], time: float | None = None) -> ValueFunc:
        return self.method_32_12(derivatives, time)


class AdamsBashford5Field(FieldStepMethod):

    def __init__(self):
        super(AdamsBashford5Field, self).__init__(req_steps=5)

    # Assumes that derivatives are equally spaced
    @staticmethod
    def adams_bashford_5step(derivatives: List[TimedVal[DerivativeFunc]], time: float | None = None) -> ValueFunc:

        vals = AdamsBashford5Field._last_n(derivatives, 6, time)
        if len(vals) < 6:
            return EulerField.euler_field(derivatives, time)

        p4 = vals[-1]
        p3 = vals[-2]
        p2 = vals[-3]
        p1 = vals[-4]
        p0 = vals[-5]
        dt = (p4.time - p3.time)

        return dt * ((1901 * p4.val - 2774 * p3.val + 2616 * p2.val - 1274 * p1.val + 251 * p0.val) * (1/720))

    def __call__(self, derivatives: List[TimedVal[DerivativeFunc]], time: float | None = None) -> ValueFunc:
        return self.adams_bashford_5step(derivatives, time)


# Implicit. Doesn't work on its own
class AdamsMoulton5Field(FieldStepMethod):

    def __init__(self):
        super(AdamsMoulton5Field, self).__init__(req_steps=5)

    # Assumes that derivatives are equally spaced
    @staticmethod
    def adams_moulton_5step(derivatives: List[TimedVal[DerivativeFunc]], time: float | None = None) -> ValueFunc:
        vals = AdamsMoulton5Field._last_n(derivatives, 6, time)
        if len(vals) < 6:
            return EulerField.euler_field(derivatives, time)

        p4 = vals[-1]
        p3 = vals[-2]
        p2 = vals[-3]
        p1 = vals[-4]
        p0 = vals[-5]
        dt = (p4.time - p3.time)

        return dt * ((251 * p4.val + 646 * p3.val - 264 * p2.val + 106 * p1.val - 19 * p0.val) * (1/720))

    def __call__(self, derivatives: List[TimedVal[DerivativeFunc]], time: float | None = None) -> ValueFunc:
        return self.adams_moulton_5step(derivatives, time)
