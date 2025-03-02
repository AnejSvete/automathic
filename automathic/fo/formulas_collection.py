# Strings that start with a and and with b
φ_0 = "exists x. exists y. ((forall z. (!(z < x))) and Qa(x) and (forall w. (!(y < w))) and Qb(y))"

# The language containing all strings that start with 'a'
φ_1 = "exists x. (forall y. (!(y < x)) and Qa(x))"

# The language containing all strings that end with 'b'
φ_2 = "exists x. (forall y. (!(x < y)) and Qb(x))"

# The language where every 'a' is immediately followed by 'b'
φ_3 = "forall x. (!Qa(x) or exists y. (y = x + 1 and Qb(y)))"

# The language where no 'a' is immediately followed by another 'a'
φ_4 = "forall x. forall y. (!(Qa(x) and y = x + 1 and Qa(y)))"

# The language containing strings with exactly one 'a'
φ_5 = "exists x. (Qa(x) and forall y. (!Qa(y) or x = y))"

# The language where every 'a' is preceded by a 'b'
φ_6 = "forall x. (!Qa(x) or exists y. (x = y + 1 and Qb(y)))"

# The language containing strings where 'a' and 'b' alternate, starting with 'a'
φ_7 = "(exists x. (forall y. (!(y < x)) and Qa(x))) and (forall x. forall y. (!(y = x + 1) or ((Qa(x) and Qb(y)) or (Qb(x) and Qa(y)))))"
