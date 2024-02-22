import enum
from functools import total_ordering


@total_ordering
@enum.unique
class BaseUniqueSortedEnum(enum.Enum):
    """Base unique enum class with ordering."""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.index = len(cls.__members__) + 1
        return obj

    def __hash__(self) -> int:
        return hash(
          f"{self.__module__}_{self.__class__.__name__}_{self.name}_{self.value}"
        )

    def __eq__(self, other) -> bool:
        self._check_type(other)
        return super().__eq__(other)

    def __lt__(self, other) -> bool:
        self._check_type(other)
        return self.index < other.index

    def _check_type(self, other) -> None:
        if type(self) != type(other):
            raise TypeError(f"Different types of Enum: {self} != {other}")


class Dog(BaseUniqueSortedEnum):
    BLOODHOUND = "BLOODHOUND"
    WEIMARANER = "WEIMARANER"
    SAME = "SAME"


class Cat(BaseUniqueSortedEnum):
    BRITISH = "BRITISH"
    SCOTTISH = "SCOTTISH"
    SAME = "SAME"


assert Dog.BLOODHOUND < Dog.WEIMARANER
assert Dog.BLOODHOUND <= Dog.WEIMARANER
assert Dog.BLOODHOUND != Dog.WEIMARANER
assert Dog.BLOODHOUND == Dog.BLOODHOUND
assert Dog.WEIMARANER == Dog.WEIMARANER
assert Dog.WEIMARANER > Dog.BLOODHOUND
assert Dog.WEIMARANER >= Dog.BLOODHOUND

assert Cat.BRITISH < Cat.SCOTTISH
assert Cat.BRITISH <= Cat.SCOTTISH
assert Cat.BRITISH != Cat.SCOTTISH
assert Cat.BRITISH == Cat.BRITISH
assert Cat.SCOTTISH == Cat.SCOTTISH
assert Cat.SCOTTISH > Cat.BRITISH
assert Cat.SCOTTISH >= Cat.BRITISH

assert hash(Dog.BLOODHOUND) == hash(Dog.BLOODHOUND)
assert hash(Dog.WEIMARANER) == hash(Dog.WEIMARANER)
assert hash(Dog.BLOODHOUND) != hash(Dog.WEIMARANER)
assert hash(Dog.SAME) != hash(Cat.SAME)

# raise TypeError
Dog.SAME <= Cat.SAME
Dog.SAME < Cat.SAME
Dog.SAME > Cat.SAME
Dog.SAME >= Cat.SAME
Dog.SAME != Cat.SAME
