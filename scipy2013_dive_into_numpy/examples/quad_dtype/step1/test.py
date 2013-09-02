import _quad

literal = "123.456788910111213141516171819"
a = _quad.qdouble(literal)

print repr(a)
print repr(float(literal))

print a + a
