'''
Trapezoidal rule for integration
'''

def trapez_rule(f, a, b, n=100):
    delta_x = ((b - a) / n)
    approx = 0.0
    y_a = f(a)
    y_b = f(b)
    i = 1
    approx_end_start = delta_x * ((y_a + y_b) / 2)

    while i < n:
        approx += delta_x * f(a + (i * delta_x))
        i += 1

    result = approx_end_start + approx
    return result
