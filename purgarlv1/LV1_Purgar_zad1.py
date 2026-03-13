
def total_euro(sati, eura_po_satu):
    return sati * satnica

sati = float(input("Radni sati: "))
satnica = float(input ("eura/h: "))

ukupno = total_euro(sati, satnica)

print("Ukupno: ", ukupno, "eura.")