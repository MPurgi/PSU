
try:
    ocjena = float(input("Unesite ocjenu (0.0 - 1.0): "))

    if ocjena < 0.0 or ocjena > 1.0:
        print("Greška: broj mora biti između 0.0 i 1.0")

    elif ocjena >= 0.9:
        print("A")
    elif ocjena >= 0.8:
        print("B")
    elif ocjena >= 0.7:
        print("C")
    elif ocjena >= 0.6:
        print("D")
    else:
        print("F")
except:
    print("greška: niste unijeli broj")