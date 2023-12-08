import matplotlib.pyplot as plt
import ast

d_A = []
d_B = []
g_AtoB = []
g_BtoA = []

with open('cpurun2.txt', 'r') as f:
    tmp = f.readlines()
    for t in tmp:
        if t.startswith('Iteration'):
            strs = t.split(' ')
            d_A.append(ast.literal_eval(strs[1][3:-1]))
            d_B.append(ast.literal_eval(strs[2][3:-1]))

            gs = strs[3][2:-2].split(',')
            g_AtoB.append(ast.literal_eval(gs[0]))
            g_BtoA.append(ast.literal_eval(gs[1]))

plt.figure()
plt.title('Discriminator Loss - Identifying Classical Music')
plt.plot([_[0] for _ in d_A], label='Real')
plt.plot([_[1] for _ in d_A], label='Fake')
plt.xticks(list(range(10)))
plt.xlabel('Epochs')
plt.ylabel('L2 Loss')
plt.legend()
plt.savefig('2d_A.png')

plt.figure()
plt.title('Discriminator Loss - Identifying Jazz Music')
plt.plot([_[0] for _ in d_B], label='Real')
plt.plot([_[1] for _ in d_B], label='Fake')
plt.xticks(list(range(10)))
plt.xlabel('Epochs')
plt.ylabel('L2 Loss')
plt.legend()
plt.savefig('2d_B.png')

plt.figure()
plt.title('Generator Loss')
plt.plot(g_AtoB, label='Jazz to Classical')
plt.plot(g_BtoA, label='Classical to Jazz')
plt.xticks(list(range(10)))
plt.xlabel('Epochs')
plt.ylabel('L2 Loss')
plt.legend()
plt.savefig('2g.png')
