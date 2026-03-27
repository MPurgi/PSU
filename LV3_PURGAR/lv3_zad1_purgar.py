import pandas as pd

mtcars = pd.read_csv('mtcars.csv')

print("1. |Top 5 sa najvecom potrosnjom|")
mpg_top5 = mtcars.sort_values(by="mpg", ascending=True).head(5)
print(mpg_top5[["car", "mpg"]])
print()

print("2. |Top 3 8cyl s najmanjom potrosnjom|")
v8 = mtcars[mtcars.cyl == 8]
top3_v8 = v8.sort_values(by = "mpg", ascending=True).head(3)
print(top3_v8[["car","mpg"]])
print()

print("3. |Srednja potrosnja sa 6cyl|")
cyl6 = mtcars[mtcars.cyl == 6]
mean6 = cyl6.mpg.mean()
print(mean6)
print()

print("4. |Srednja potrosnja s 4cyl izmedu 2000 i 2200lbs|")
filtered=mtcars[(mtcars.cyl == 4) & (mtcars.wt >= 2.0) & (mtcars.wt <= 2.2)]
mean4 = filtered.mpg.mean()
print(mean4)
print()

print("5. |Automobili s rucnim i automatskim mjenjacem|")
manual = mtcars[mtcars.am == 1]
automatic = mtcars[mtcars.am == 0]
print(len(manual))
print(len(automatic))
print()

print("6. |Automobili s automatskim mjenjacem i preko 100 ks|")
auto_hp = mtcars[(mtcars.am == 0) & (mtcars.hp > 100)]
count_auto_hp = len(auto_hp)
print(count_auto_hp)
print()

print("7. |Masa svakog automobila u kilogramima|")
mtcars["wt_kg"] = mtcars.wt * 1000 * 0.453592
masa = mtcars[["car", "wt_kg"]]
print(masa)
print()

print(mtcars)

#print(len(mtcars))
#print(mtcars)
#print(mtcars.head(5))
#print(mtcars.tail(3))
#print(mtcars.info())
#print(mtcars.describe())