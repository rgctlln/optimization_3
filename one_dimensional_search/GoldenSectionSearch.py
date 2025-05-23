import math

def golden_section_search(f, l, r, eps=1e-8):
    phi = (math.sqrt(5) - 1) / 2  # ≈ 0.618...
    x1 = r - phi * (r - l)
    x2 = l + phi * (r - l)
    f1 = f(x1)
    f2 = f(x2)

    while (r - l) > eps * (abs(x1) + abs(x2)) / 2:
        if f1 < f2:
            r = x2
            x2 = x1
            f2 = f1
            x1 = r - phi * (r - l)
            f1 = f(x1)
        else:
            l = x1
            x1 = x2
            f1 = f2
            x2 = l + phi * (r - l)
            f2 = f(x2)

    return (l + r) / 2
