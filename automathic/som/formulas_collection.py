# Strings that start with a and and with b
φ_0 = "exists x. exists y. ((forall z. (!(z < x))) and Qa(x) and (forall w. (!(y < w))) and Qb(y))"

# The language containing all strings that start with 'a'
φ_1a = "exists x. (forall y. (!(y < x)) and Qa(x))"
φ_1b = "forall x. (exists y. ((y <= x)) and Qa(y))"

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

# The language of strings of even length
φ_8 = """
    exists X. (
        forall x1. (forall z1. (z1 >= x1) => X(x1)) and 
        forall x2. (forall z2. (z2 <= x2) => not (X(x2))) and 
        forall x3. (forall y3. ((x3 < y3 and forall z3. ((z3 > x3) => z3 >= y3)) => (X(x3) <=> not (X(y3)))))
    )
"""

# The even parity language
φ_8 = """
exists X. (
        forall x1. (Qa(x1) and (forall z1. (Qa(z1) => (z1 >= x1))) => X(x1)) and 
        forall x2. (Qa(x2) and (forall z2. (Qa(z2) => (x2 >= z2))) => not X(x2)) and 
        forall x3. forall y3. (
                (
                        Qa(x3) and Qa(y3) and y3 > x3 and (forall z3. (Qa(z3) and z3 > x3 => (z3 >= y3))) => (
                        (X(x3) <=> not (X(y3)))
                )
                )
        )
)
"""

# The odd parity language #1
φ_9a = """
not (exists X. (
                forall x1. (Qa(x1) and (forall z1. (Qa(z1) => (z1 >= x1))) => X(x1)) and 
                forall x2. (Qa(x2) and (forall z2. (Qa(z2) => (x2 >= z2))) => not X(x2)) and 
                forall x3. forall y3. (
                        (
                                Qa(x3) and Qa(y3) and y3 > x3 and (forall z3. (Qa(z3) and z3 > x3 => (z3 >= y3))) => (
                                (X(x3) <=> not (X(y3)))
                        )
                        )
                )
        )
)
"""

# The odd parity language #2
φ_9b = """
exists X. (
        exists x0. (Qa(x0)) and
        forall x1. (Qa(x1) and (forall z1. (Qa(z1) => (z1 >= x1))) => X(x1)) and 
        forall x2. (Qa(x2) and (forall z2. (Qa(z2) => (x2 >= z2))) => X(x2)) and 
        forall x3. forall y3. (
                (
                        Qa(x3) and Qa(y3) and y3 > x3 and (forall z3. (Qa(z3) and z3 > x3 => (z3 >= y3))) => (
                        (X(x3) <=> not (X(y3)))
                )
                )
        )
)
"""
