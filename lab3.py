import random
import numpy as np
import copy


x1min = 10
x1max = 50
x2min = -20
x2max = 60
x3min = -20
x3max = 20
xAvmax = x1max+x2max+x3max/3
xAvmin = x1min+x2min+x3min/3
ymax = int(200+xAvmax)
ymin = int(200+xAvmin)


print("{:^31}{:^41}".format('Кодованє значення X', 'Матриця для m=3'))
print("{:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}"
      .format("№", "X1", "X2", "X3", "#", "№", "X1", "X2", "X3", "Y1", "Y2", "Y3"))

Xi = [[1, 1, 1, 1], [-1, -1, +1, +1], [-1, +1, -1, +1], [-1, +1, +1, -1]]
X = [[x1min, x1min, x1max, x1max],
     [x2min, x2max, x2min, x2max],
     [x3min, x3max, x3max, x3min]]
Y = [[random.randrange(138, 247, 1) for _ in range(4)] for __ in range(3)]

for i in range(4):
    print("{:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}"
          .format(i+1, Xi[1][i], Xi[2][i], Xi[3][i], "#", i+1, X[0][i], X[1][i], X[2][i], Y[0][i], Y[1][i], Y[2][i]))


print("\n_________Критерій Кохрена________")
print("Середнє значення відгуку функції: ")
yav = [(Y[0][i]+Y[1][i]+Y[2][i])/3 for i in range(4)]
my = sum(yav)/4
mx = [sum(X[i]) for i in range(3)]

a = [(X[i][0]*yav[0] + X[i][1]*yav[1] + X[i][2]*yav[2] + X[i][3]*yav[3])/4 for i in range(3)]


a11 = (X[0][0]**2 + X[0][1]**2 + X[0][2]**2 + X[0][3]**2)/4
a22 = (X[1][0]**2 + X[1][1]**2 + X[1][2]**2 + X[1][3]**2)/4
a33 = (X[2][0]**2 + X[2][1]**2 + X[2][2]**2 + X[2][3]**2)/4

a12 = a21 = (X[0][0]*X[1][0] + X[0][1]*X[1][1] + X[0][2]*X[1][2] + X[0][3]*X[1][3])/4
a13 = a31 = (X[0][0]*X[2][0] + X[0][1]*X[2][1] + X[0][2]*X[2][2] + X[0][3]*X[2][3])/4
a23 = a32 = (X[1][0]*X[2][0] + X[1][1]*X[2][1] + X[1][2]*X[2][2] + X[1][3]*X[2][3])/4

b = []
b01 = np.array([[my, mx[0], mx[1], mx[2]], [a[0], a11, a12, a13], [a[1], a12, a22, a32], [a[2], a13, a23, a33]])
b02 = np.array([[1, mx[0], mx[1], mx[2]], [mx[0], a11, a12, a13], [mx[1], a12, a22, a32], [mx[2], a13, a23, a33]])
b.append(np.linalg.det(b01)/np.linalg.det(b02))

b11 = np.array([[1, my, mx[1], mx[2]], [mx[0], a[0], a12, a13], [mx[1], a[1], a22, a32], [mx[2], a[2], a23, a33]])
b12 = copy.deepcopy(b02)
b.append(np.linalg.det(b11)/np.linalg.det(b12))

b21 = np.array([[1, mx[0], my, mx[2]], [mx[0], a11, a[0], a13], [mx[1], a12, a[1], a32], [mx[2], a13, a[2], a33]])
b22 = copy.deepcopy(b02)
b.append(np.linalg.det(b21)/np.linalg.det(b22))

b31 = np.array([[1, mx[0], mx[1], my], [mx[0], a11, a12, a[0]], [mx[1], a12, a22, a[1]], [mx[2], a13, a23, a[2]]])
b32 = copy.deepcopy(b02)
b.append(np.linalg.det(b31)/np.linalg.det(b32))


for i in range(4):
    print("y{} середнє = {:.2f} = {:.2f}".format(i+1, b[0] + b[1]*X[0][i] + b[2]*X[1][i] + b[3]*X[2][i], yav[i]))


print("Рівняння регресії:  ŷ = {:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3".format(b[0], b[1], b[2], b[3]))

print("\nДисперсія по рядкам")
d = [((Y[0][i] - yav[0])**2 + (Y[1][i] - yav[1])**2 + (Y[2][i] - yav[2])**2)/3 for i in range(4)]

print("d1 = {:.2f} d2 = {:.2f} d3 = {:.2f} d4 = {:.2f}".format(*d))

m = 3
Gp = max(d)/sum(d)
f1 = m-1
f2 = N = 4
Gt = 0.7679
print(f"Gp = {Gp}\nGt = {Gt}")
if Gp < Gt:
    print("Gp < Gt\nОтже -Дисперсія однорідна-")
else:
    print("Дисперсія  неоднорідна(збільшемо кількість дослідів)")
    m += 1


print("\n____________Критерій Стьюдента__________")
sb = sum(d)/N
ssbs = sb / N * m
sbs = ssbs**0.5

beta = [(yav[0] * Xi[i][0] + yav[1] * Xi[i][1] + yav[2] * Xi[i][2] + yav[3]*Xi[i][3])/4 for i in range(4)]

t = [abs(beta[i])/sbs for i in range(4)]
print("t0 = {:.2f} t1 = {:.2f} t2 = {:.2f} t3 = {:.2f}".format(*t))

f3 = f1*f2
ttabl = 2.306


for i in range(4):
    if t[i] < ttabl:
        print(f"t{i} < ttabl, b{i} не значимий")
        b[i] = 0

yy = [b[0] + b[1]*X[0][i] + b[2]*X[1][i] + b[3]*X[2][i] for i in range(4)]


print("\n___________________________Критерій Фішера__________________________")
d_ = 2
sad = ((yy[0] - yav[0])**2 + (yy[1] - yav[1])**2 + (yy[2] - yav[2])**2 + (yy[3] - yav[3])**2)*(m/(N-d_))
Fp = sad / sb
print("d1 = {:.2f}  d2 = {:.2f}  d3 = {:.2f}  d4 = {:.2f}  d5 = {:.2f}".format(*d, sb))
print(f"Fpratk = {Fp:.2f}")
print('Ftabl = 4.5')
Ft = 4.5
if Fp > Ft:
    print("Fprakt > Ftabl", "\nРівняння неадекватно оригіналу")
else:
    print("Fprakt < Ftabl", "\nРівняння адекватно оригіналу")
