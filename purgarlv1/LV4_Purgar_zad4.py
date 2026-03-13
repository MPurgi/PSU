ime = input ("Ime datoteke: ")
try:
    dat = open(ime)
except FileNotFoundError:
    print("Datoteka ne postoji. Provjerite ime i lokaciju.")
    quit()

zbroj = 0
broj = 0

for linija in dat:
    if linija.startswith("X-DSPAM-Confidence:"):
        vrijednost = float(linija.split(":")[1].strip())
        zbroj+=vrijednost
        broj+=1
if broj>0:
    print("Average X-DSPAM-Confidence:", zbroj/broj)
else:
    print("Nema linija X-DSPAM-Confidence u datoteci.")