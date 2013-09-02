In this first step, we create a simple extension type that wraps a __float128
value.

Your task is to implement a basic number protocol so that the following works:

from _quad import qdouble

literal = "1233445.12398798723482398479287364"
a = qdouble(literal)

print repr(a + a)
print repr(float(a) + float(a))
