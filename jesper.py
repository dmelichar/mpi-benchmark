import math

p = 11

def f(a, d):
    print(f"Build H_{d}: {a*2**d}, ... {(a+1)*2**d-1}")
    print(f"Build H_({d}+1): " \
          f"{2*a*2**d}, ..., {(2*a+1)*2**d-1}" \
          " and " \
          f"{(2*a+1)*2**d}, ..., {(2*a+2)*2**d-1}")
    print()

for d in range(0, math.ceil(math.log(p) / math.log(2))):
    print(f"==> d = {d}")
    for a in range(math.ceil(p/(2**d))):
        f(a,d)


