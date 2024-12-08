import numpy as np

G = np.array([['Y','W','W','W'],
              ['W','W','G','G'],
              ['W','W','G','G'],
              ['W','W','G','G']])
A = {'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1),'S':(0,0)}
def c(f,t,a):
    if a=='S': return 0 if f=='Y' else 2
    return 10 if t=='G' else 1
def valid(r,c_): return 0<=r<4 and 0<=c_<4

V = np.zeros((4,4))
for i in range(6):
    if i>0:
        Vnew = np.zeros((4,4))
        P = np.empty((4,4),dtype=object)
        for r in range(4):
            for c_ in range(4):
                f = G[r,c_]
                vals = {}
                for a_ in A:
                    nr,nc = r+A[a_][0], c_+A[a_][1]
                    if not valid(nr,nc): nr,nc = r,c_
                    vals[a_] = c(f,G[nr,nc],a_)+V[nr,nc]
                best = min(vals,key=vals.get)
                Vnew[r,c_]=vals[best];P[r,c_]=best
        V=Vnew
        print(f"i={i}\nV:\n{V}\nPolicy:")
        arr={'U':'↑','D':'↓','L':'←','R':'→','S':'•'}
        for r in range(4):
            print(' '.join(arr[x] for x in P[r]))
        print()
    else:
        print("i=0\nV:\n",V,"\n")