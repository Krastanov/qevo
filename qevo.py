from itertools import product
from functools import reduce
from operator import mul
import itertools
import functools
import random
import sys

import qutip

import numpy as np
import scipy
from scipy import optimize
from scipy import interpolate
import sympy

try:
    from IPython import display
    import matplotlib
    import matplotlib.pyplot as plt
    plt.ioff()
    _gui = True
except ImportError:
    _gui = False

T = qutip.tensor
ψ2ρ = qutip.ket2dm
P = lambda *states: T(*states)*T(*states).dag()
withHC = lambda op: op + op.dag()
π = np.pi
l = qutip.basis(2,0)
h = qutip.basis(2,1)
ll = T(l,l)
lh = T(l,h)
hl = T(h,l)
hh = T(h,h)
A = φp = (ll+hh).unit()
B = ψm = (lh-hl).unit()
C = ψp = (lh+hl).unit()
D = φm = (ll-hh).unit()
toABCD_mat = ll*φp.dag() + lh*ψm.dag() + hl*ψp.dag() + hh*φm.dag()
id2 = qutip.identity(2)

q, epsilon_m, epsilon_g = sympy.symbols('q epsilon_m epsilon_g')
symF = sympy.Symbol('F_0')
eps_p2, eps_η = sympy.symbols('epsilon_p2 epsilon_η')
eps = sympy.Symbol('epsilon')
polyF     = sympy.poly(symF,      symF, q, epsilon_m, epsilon_g, domain='QQ')
inf_third = sympy.poly(q,         symF, q, epsilon_m, epsilon_g, domain='QQ')
inf_meas  = sympy.poly(epsilon_m, symF, q, epsilon_m, epsilon_g, domain='QQ')
inf_gate  = sympy.poly(epsilon_g, symF, q, epsilon_m, epsilon_g, domain='QQ')


def ψ_to_ABCD_basis(ψ):
    return toABCD_mat*ψ
def letter_code_amplitude(ρ, exceptions=True):
    letters = 'ABCD'
    for _l in itertools.product(letters, repeat=len(ρ.dims[0])//2):
        if abs((T(*({'A':A,'B':B,'C':C,'D':D}[_] for _ in _l)).dag()*ρ).tr()) > 0.999:
            return ''.join(_l)
    else:
        if exceptions:
            raise Exception('not diagonal in Bell basis')
        return None
def letter_map(op, exceptions=True):
    res = []
    for start in itertools.product('ABCD',repeat=len(op.dims[0])//2):
        s = ''.join(start)
        e = letter_code_amplitude(op*T(*(globals()[_] for _ in start)),
                                  exceptions=exceptions)
        if e is not None:
            res.append((s,e))
        else:
            return None
    return res
def letter_to_index_map_assuming_2idempotent(l):
    d = {'A':0,'B':1,'C':2,'D':3}
    pairs = []
    for (l1,l2), (r1,r2) in l:
        l,r = ((d[l1],d[l2]), (d[r1],d[r2]))
        if l != r and (r, l) not in pairs:
            pairs.append((l,r))
    return pairs

def cgate_to_permutation_map(cgate):
    if len(cgate.dims[0]) == 1:
        op = qutip.controlled_gate(cgate, N=4, control=0, target=2)*qutip.controlled_gate(cgate, N=4, control=1, target=3)
    if len(cgate.dims[0]) == 2:
        ...
        #op = qutip.tensor(cgate, cgate)
    elif len(cgate.dims[0]) == 4:
        op = cgate
    l = letter_map(op)
    i = letter_to_index_map_assuming_2idempotent(l)
    return l, i
# CPHASE, CNOT, CPNOT
cphase_gate = l*l.dag() - h*h.dag()
cphase_l, cphase_i = cgate_to_permutation_map(cphase_gate)
cnot_gate = h*l.dag() + l*h.dag()
cnot_l, cnot_i = cgate_to_permutation_map(cnot_gate)
cpnot_gate = h*l.dag() - l*h.dag()
cpnot_l, cpnot_i = cgate_to_permutation_map(cpnot_gate)

# Tools for working with 2-pair permutation lists
r = lambda a: sorted([(l[::-1], r[::-1]) for l,r in a], key=lambda _:_[0])
def _chain_permutations(l1, l2):
    d1 = dict(l1)
    d2 = dict(l2)
    d3 = {k1: d2[v1] for k1,v1 in d1.items()}
    return sorted(d1.items(), key=lambda _:_[0])
c = lambda *_: functools.reduce(_chain_permutations, _)
def print_table(*args):
    args = list(args)
    l = len(args)
    for _ in zip(*args):
        print(_[0][0], '->', ('%s  '*l)%tuple([r for l,r in _]))

# All 1-qubit gates
onequbitgates = list(itertools.permutations([0,1,2,3]))

# Utilities

def get_new_probs(N, F, qs=None):
    lF = np.log(F)
    if qs is None:
        q = (1-F)/3
        lq = np.log(q)
        lqs = [lq]*3
    else:
        lqs = [np.log(q) for q in qs]
    probs = np.empty((4,)*N)
    for i, p in zip(itertools.product(range(4),   repeat=N),
                    itertools.product((lF, *lqs), repeat=N)):
        probs[i] = np.exp(sum(p))
    return probs

def get_new_probs_sym(N, separate_F = False):
    q = inf_third
    F = polyF if separate_F else 1-3*q
    probs = np.empty((4,)*N, dtype=object)
    for i, p in zip(itertools.product(range(4),     repeat=N),
                    itertools.product((F, q, q, q), repeat=N)):
        probs[i] = reduce(mul,p)
    return probs

def fidelity(probs):
    return np.sum(probs[0,...])

def ABCD(probs):
    return tuple(np.sum(probs[_,...]) for _ in range(4))

def permute1(probs, target, permutation, N):
    slices = [slice(None)]*N
    slices[target] = permutation
    return probs[tuple(slices)]

def permute2_by_pairs(probs, targets, permutation_pairs, N):
    slices_a = [slice(None)]*N
    slices_b = [slice(None)]*N
    t1, t2 = targets
    for (l1, l2), (r1, r2) in permutation_pairs:
        slices_a[t1] = l1
        slices_a[t2] = l2
        slices_b[t1] = r1
        slices_b[t2] = r2
        tmp_a = np.copy(probs[tuple(slices_a)])
        tmp_b = np.copy(probs[tuple(slices_b)])
        probs[tuple(slices_a)] = tmp_b
        probs[tuple(slices_b)] = tmp_a
    return probs

def measure(probs, target, pair, N, F, Mη, qs=None):
    Mη = Mη**2 + (1-Mη)**2 # both have to be right or both have to be wrong!
    q = [(1-F)/3]*3 if qs is None else qs
    slices = [slice(None)]*N
    slices[target] = pair
    probs_meas = np.sum(probs[tuple(slices)], axis=target)
    slices[target] = [_ for _ in range(4) if _ not in pair]
    probs_meas_err = np.sum(probs[tuple(slices)], axis=target)
    probs_meas = Mη*probs_meas + (1-Mη)*probs_meas_err
    probs_meas /= np.sum(probs_meas)
    for i, f in zip(range(4),(F,*q)):
        slices[target] = i
        probs[tuple(slices)] = probs_meas*f
    return probs

def measure_sym_notnorm(probs, target, pair, N, separate_F=False):
    q = inf_third
    F = polyF if separate_F else 1-3*q
    slices = [slice(None)]*N
    slices[target] = pair
    probs_meas = np.sum(probs[tuple(slices)], axis=target)
    slices[target] = [_ for _ in range(4) if _ not in pair]
    probs_meas_err = np.sum(probs[tuple(slices)], axis=target)
    probs_meas = (1-inf_meas)*probs_meas + inf_meas*probs_meas_err
    #probs_meas /= np.sum(probs_meas)
    #probs_meas = np.vectorize(sympy.cancel)(probs_meas)
    for i, f in zip(range(4),(F,q,q,q)):
        slices[target] = i
        probs[tuple(slices)] = probs_meas*f
    return probs

def measure_succ_prob(probs, target, pair, N, F, Mη):
    Mη = Mη**2 + (1-Mη)**2 # both have to be right or both have to be wrong!
    q = (1-F)/3
    slices = [slice(None)]*N
    slices[target] = pair
    probs_meas = np.sum(probs[tuple(slices)], axis=target)
    slices[target] = [_ for _ in range(4) if _ not in pair]
    probs_meas_err = np.sum(probs[tuple(slices)], axis=target)
    probs_meas = Mη*probs_meas + (1-Mη)*probs_meas_err
    return np.sum(probs_meas)

def depolarize2(probs, targets, p):
    p = p**2 # depolarizing one of the pairs depolarizes the other pair!
    probs_ptrace = 1/4**2*np.sum(np.sum(probs, axis=targets[0], keepdims=True),
                                               axis=targets[1], keepdims=True)
    return p*probs + (1-p)*probs_ptrace

sixteenth = 1/sympy.sympify(16)
def depolarize2_sym(probs, targets, p):
    probs_ptrace = np.sum(np.sum(probs, axis=targets[0], keepdims=True),
                                        axis=targets[1], keepdims=True)*sixteenth
    probs = (1-inf_gate)*probs + inf_gate*probs_ptrace
    return probs

class Operation:
    def apply(self, probs):
        raise NotImplementedError
    def apply_perf(self, probs):
        raise NotImplementedError
    @classmethod
    def random(cls, N, F, P2, Mη, qs=None): # TODO XXX many of the arguments are not actually used - something like a **kwargs might be in order.
        raise NotImplementedError
    def mutate(self):
        raise NotImplementedError
    def strs(self):
        raise NotImplementedError
    def copy(self):
        raise NotImplementedError
    def succ_prob(self, probs):
        return 1.
    def rearrange(self, permutation_dict):
        raise NotImplementedError
    def __repr__(self):
        return str(self)

class Permutation(Operation):
    def __init__(self, target, permutation, N):
        self.permutation = permutation
        self.target = target
        self.N = N
    def __eq__(self, other):
        return type(other) is Permutation and self.target == other.target and self.permutation == other.permutaion
    def __hash__(self):
        return hash((Permutation, self.target, tuple(self.permutation), self.N))
    def strs(self):
        return ['-%s-'%(''.join(map(str,self.permutation)))
                if _ == self.target else '------'
                for _ in range(self.N)]
    def __str__(self):
        return 'Permutation(%s, %s, %s)'%(self.target, self.permutation, self.N)
    def apply(self, probs):
        return permute1(probs, self.target, self.permutation, self.N)
    @classmethod
    def random(cls, N, F, P2, Mη, qs=None):
        return cls(random.randint(0,N-1),
                   random.choice(onequbitgates))
    def mutate(self):
        return Permutation(self.target, random.choice(onequbitgates), self.N)
    def copy(self, F=None, P2=None, Mη=None):
        return Permutation(self.target, self.permutation, self.N)
    @property
    def targets(self):
        return (self.target,)
    def rearrange(self, permutation_dict):
        r = self.copy()
        r.target = permutation_dict[r.target]
        return r
    @property
    def args(self):
        return self.target, self.permutation, self.N

class CNOT(Operation):
    def __init__(self, targets, N, P2):
        self.targets = list(targets)
        self.N = N
        self.P2 = P2
    def __eq__(self, other):
        return type(other) is CNOT and self.targets == other.targets
    def __hash__(self):
        return hash((CNOT, tuple(self.targets), self.N))
    def strs(self):
        return ['-o-' if _ == self.targets[0]
                else '-X-' if _ == self.targets[1]
                else '-|-' if min(self.targets)<_<max(self.targets)
                else '---'
                for _ in range(self.N)]
    def __str__(self):
        return 'CNOT(%s, %s, %s)'%(self.targets, self.N, self.P2)
    def apply(self, probs):
        return depolarize2(permute2_by_pairs(probs, self.targets, cnot_i, self.N),
                                                    self.targets, self.P2)
    def apply_sym(self, probs):
        return depolarize2_sym(permute2_by_pairs(probs, self.targets, cnot_i, self.N),
                                                 self.targets, self.P2)
    @classmethod
    def random(cls, N, F, P2, Mη, qs=None):
        return cls(random.sample(range(N),2), N, P2)
    def mutate(self):
        return random.choice([_(self.targets, self.N, self.P2) for _ in [CPHASE, CPNOT]])
    def copy(self, F=None, P2=None, Mη=None):
        return CNOT(self.targets.copy(), self.N,
                    P2=P2 if P2 else self.P2)
    def rearrange(self, permutation_dict):
        r = self.copy()
        r.targets[0] = permutation_dict[r.targets[0]]
        r.targets[1] = permutation_dict[r.targets[1]]
        return r
    @property
    def args(self):
        return self.targets, self.N, self.P2

class CPHASE(Operation):
    def __init__(self, targets, N, P2):
        self.targets = sorted(targets)
        self.N = N
        self.P2 = P2
    def __eq__(self, other):
        return type(other) is CPHASE and sorted(self.targets) == sorted(other.targets)
    def __hash__(self):
        return hash((CPHASE, tuple(sorted(self.targets)), self.N))
    def strs(self):
        return ['-o-' if _ == self.targets[0]
                else '-Z-' if _ == self.targets[1]
                else '-|-' if min(self.targets)<_<max(self.targets)
                else '---'
                for _ in range(self.N)]
    def __str__(self):
        return 'CPHASE(%s, %s, %s)'%(self.targets, self.N, self.P2)
    def apply(self, probs):
        return depolarize2(permute2_by_pairs(probs, self.targets, cphase_i, N=self.N),
                                                    self.targets, self.P2)
    def apply_sym(self, probs):
        return depolarize2_sym(permute2_by_pairs(probs, self.targets, cphase_i, self.N),
                                                 self.targets, self.P2)
    @classmethod
    def random(cls, N, F, P2, Mη, qs=None):
        return cls(random.sample(range(N),2), N, P2)
    def mutate(self):
        return random.choice([_(self.targets, self.N, self.P2) for _ in [CNOT, CPNOT]])
    def copy(self, F=None, P2=None, Mη=None):
        return CPHASE(self.targets.copy(), self.N,
                      P2=P2 if P2 else self.P2)
    def rearrange(self, permutation_dict):
        r = self.copy()
        r.targets[0] = permutation_dict[r.targets[0]]
        r.targets[1] = permutation_dict[r.targets[1]]
        r.targets.sort()
        return r
    @property
    def args(self):
        return self.targets, self.N, self.P2

class CPNOT(Operation):
    def __init__(self, targets, N, P2):
        self.targets = list(targets)
        self.N = N
        self.P2 = P2
    def __eq__(self, other):
        return type(other) is CPNOT and self.targets == other.targets
    def __hash__(self):
        return hash((CPNOT, tuple(self.targets), self.N))
    def strs(self):
        return ['-o-' if _ == self.targets[0]
                else '-Y-' if _ == self.targets[1]
                else '-|-' if min(self.targets)<_<max(self.targets)
                else '---'
                for _ in range(self.N)]
    def __str__(self):
        return 'CPNOT(%s, %s, %s)'%(self.targets, self.N, self.P2)
    def apply(self, probs):
        return depolarize2(permute2_by_pairs(probs, self.targets, cpnot_i, self.N),
                                                    self.targets, self.P2)
    def apply_sym(self, probs):
        return depolarize2_sym(permute2_by_pairs(probs, self.targets, cpnot_i, self.N),
                                                 self.targets, self.P2)
    @classmethod
    def random(cls, N, F, P2, Mη, qs=None):
        return cls(random.sample(range(N),2), N, P2)
    def mutate(self):
        return random.choice([_(self.targets, self.N, self.P2) for _ in [CPHASE, CNOT]])
    def copy(self, F=None, P2=None, Mη=None):
        return CPNOT(self.targets.copy(), self.N,
                     P2=P2 if P2 else self.P2)
    def rearrange(self, permutation_dict):
        r = self.copy()
        r.targets[0] = permutation_dict[r.targets[0]]
        r.targets[1] = permutation_dict[r.targets[1]]
        return r
    @property
    def args(self):
        return self.targets, self.N, self.P2

class CNOTPerm(Operation):
    def __init__(self, targets, permcontrol, permtarget, N, P2):
        self.targets = sorted(targets)
        self.N = N
        self.P2 = P2
        self.permcontrol = tuple(permcontrol)
        self.permtarget = tuple(permtarget)
    def __eq__(self, other):
        return type(other) is CNOTPerm and self.targets == other.targets and self.permcontrol == other.permcontrol and self.permtarget == self.permtarget
    def __hash__(self):
        return hash((CNOTPerm, tuple(self.targets), self.permcontrol, self.permtarget, self.N))
    def strs(self):
        return ['*o-' if _ == self.targets[0]
                else '*X-' if _ == self.targets[1]
                else '-|-' if min(self.targets)<_<max(self.targets)
                else '---'
                for _ in range(self.N)]
    def __str__(self):
        return 'CNOTPerm(%s, %s, %s, %s, %s)'%(self.targets, self.permcontrol, self.permtarget, self.N, self.P2)
    def apply(self, probs):
        slices = [slice(None)]*self.N
        slices[self.targets[0]] = self.permcontrol
        probs[...] = probs[slices]
        slices = [slice(None)]*self.N
        slices[self.targets[1]] = self.permtarget
        probs[...] = probs[slices]
        return depolarize2(permute2_by_pairs(probs, self.targets, cnot_i, self.N),
                                                    self.targets, self.P2)
    def apply_sym(self, probs):
        slices = [slice(None)]*self.N
        slices[self.targets[0]] = self.permcontrol
        probs[...] = probs[slices]
        slices = [slice(None)]*self.N
        slices[self.targets[1]] = self.permtarget
        probs[...] = probs[slices]
        return depolarize2_sym(permute2_by_pairs(probs, self.targets, cnot_i, self.N),
                                                 self.targets, self.P2)
    @classmethod
    def random(cls, N, F, P2, Mη, qs=None):
        pc = [1,2,3]
        pt = [1,2,3]
        random.shuffle(pc)
        random.shuffle(pt)
        pc = [0]+pc
        pt = [0]+pt
        return cls(random.sample(range(N),2), tuple(pc), tuple(pt), N, P2)
    def mutate(self):
        pc = [1,2,3]
        pt = [1,2,3]
        random.shuffle(pc)
        random.shuffle(pt)
        pc = [0]+pc
        pt = [0]+pt
        c = self.copy()
        c.permcontrol = pc
        c.permtarget = pt
        return c
    def copy(self, F=None, P2=None, Mη=None):
        return CNOTPerm(self.targets.copy(), self.permcontrol, self.permtarget, self.N,
                        P2=P2 if P2 else self.P2)
    def rearrange(self, permutation_dict): # XXX breaks hash
        r = self.copy()
        r.targets[0] = permutation_dict[r.targets[0]]
        r.targets[1] = permutation_dict[r.targets[1]]
        return r
    @property
    def args(self):
        return self.targets, self.permcontrol, self.permtarget, self.N, self.P2

class Measurement(Operation):
    def __init__(self, target, pair, N, F, Mη, qs=None):
        self.target = target
        self.pair = sorted(pair)
        self.N = N
        self.F = F
        self.Mη = Mη
        self.qs = qs
    def __eq__(self, other):
        return type(other) is Measurement and self.target == other.target and self.pair == other.pair
    def __hash__(self):
        return hash((Measurement, self.target, tuple(self.pair), self.N))
    def strs(self):
        return ['-D%s >'%(''.join(map(str,self.pair)))
                if _ == self.target else '------'
                for _ in range(self.N)]
    def __str__(self):
        return 'Measurement(%s, %s, %s, %s, %s, %s)'%(self.target, self.pair,
                                                      self.N, self.F, self.Mη,
                                                      self.qs)
    def apply(self, probs):
        return measure(probs, self.target, self.pair, self.N, self.F, self.Mη, qs=self.qs)
    def apply_sym(self, probs, separate_F=False):
        return measure_sym_notnorm(probs, self.target, self.pair, self.N, separate_F=separate_F)
    def succ_prob(self, probs):
        return measure_succ_prob(probs, self.target, self.pair, self.N, self.F, self.Mη)
    @classmethod
    def random(cls, N, F, P2, Mη, qs=None):
        return cls(random.randint(0,N-1),
                   random.choice([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]),
                   N, F, Mη, qs)
    def mutate(self):
        return Measurement(self.target, random.choice([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]),
                           self.N, self.F, self.Mη, self.qs)
    def copy(self, F=None, P2=None, Mη=None, qs=None):
        return Measurement(self.target, self.pair.copy(), self.N,
                           F=F if F else self.F,
                           Mη=Mη if Mη else self.Mη,
                           qs=qs if qs else self.qs)
    @property
    def targets(self):
        return (self.target,)
    def rearrange(self, permutation_dict):
        r = self.copy()
        r.target = permutation_dict[r.target]
        return r
    @property
    def args(self):
        return self.target, self.pair, self.N, self.F, self.Mη

class AMeasurement(Measurement):
    def __init__(self, target, other, N, F, Mη, qs=None):
        self.target = target
        self.other = other
        self.N = N
        self.F = F
        self.Mη = Mη
        self.qs = qs
    @property
    def pair(self):
        return [0, self.other]
    def __eq__(self, other):
        return type(other) is AMeasurement and self.target == other.target and self.other == other.other
    def __hash__(self):
        return hash((AMeasurement, self.target, self.other, self.N))
    def __str__(self):
        return 'AMeasurement(%s, %s, %s, %s, %s, %s)'%(self.target, self.other,
                                                       self.N, self.F, self.Mη,
                                                       self.qs)
    @classmethod
    def random(cls, N, F, P2, Mη, qs=None):
        return cls(random.randint(0,N-1),
                   random.choice([1,2,3]),
                   N, F, Mη, qs)
    def mutate(self):
        return AMeasurement(self.target, random.choice([1,2,3]),
                           self.N, self.F, self.Mη, self.qs)
    def copy(self, F=None, P2=None, Mη=None, qs=None):
        return AMeasurement(self.target, self.other, self.N,
                           F=F if F else self.F,
                           Mη=Mη if Mη else self.Mη,
                           qs=qs if qs else self.qs)
    @property
    def args(self):
        return self.target, self.other, self.N, self.F, self.Mη

from enum import Enum
class History(Enum):
    manual = 0
    survivor = 1
    random = 2
    child = 3
    drop_m = 4
    gain_m = 5
    swap_m = 6
    ops_m = 7

class Individual:
    def __init__(self,ops,F,history=History.manual, weights=(1,0,0,0,0),qs=None): # qs is not particularly well tested
        self.ops = ops
        self._fitness = None
        self._fidelity_and_succ_prob = None
        self.F = F
        self.history = history
        self.weights = weights
        self.qs = qs
    def __str__(self):
        return '\n'.join(''.join(__) for __ in zip(*(_.strs() for _ in self.ops)))
    def __repr__(self):
        return 'Individual(F=%s, history=History.%s, weights=%s, ops=[\n           %s])'%(
            self.F, self.history.name, self.weights,
            ',\n           '.join(str(_) for _ in self.ops))
    def __eq__(self, other):
        return other.ops == self.ops
    def __hash__(self):
        return hash(tuple(self.ops))
    def montecarlo_chain(self):
        probs = get_new_probs(self.N, self.F, qs=self.qs)
        success_probabilities = []
        test_steps = []
        back_to_index = []
        reset_raw_pairs = []
        for i,o in enumerate(self.ops):
            p = o.succ_prob(probs)
            probs = o.apply(probs)
            if p != 1.:
                success_probabilities.append(p)
                test_steps.append(i)
                back_to_index.append(0) # XXX Fix this... This is worst case! It might be better in your case!
                reset_raw_pairs.append(self.N) # XXX Same issue...
        return success_probabilities, test_steps, back_to_index, reset_raw_pairs
    def montecarlo_resources(self, runs=1000, custom_chain=None):
        if custom_chain:
            success_probabilities, test_steps, back_to_index, reset_raw_pairs = custom_chain
        else:
            success_probabilities, test_steps, back_to_index, reset_raw_pairs = self.montecarlo_chain()
        N_tests = len(success_probabilities)
        delta_steps = [test_steps[0]] + [n-p for n,p in zip(test_steps[1:], test_steps[:-1])]
        results = np.zeros((runs,2),dtype=int)
        for run in range(runs):
            raw_pairs = self.N
            steps = 1
            current = 0
            while current < N_tests:
                if random.random() < success_probabilities[current]:
                    raw_pairs += 1 if current < N_tests-self.N+1 else 0
                    steps += delta_steps[current]
                    current += 1
                else:
                    raw_pairs += reset_raw_pairs[current]
                    steps += 1 + delta_steps[current]
                    current = back_to_index[current]
            results[run,0] = raw_pairs
            results[run,1] = steps
        return results
    def fidelity_and_succ_prob(self, reval=False):
        if reval or not self._fidelity_and_succ_prob:
            if not self.ops:
                self._probs = 1.
                self._fidelity_and_succ_prob = self.F, 1.
                self._ABCD = np.array([self.F]+[(1-self.F)/3]*3)
                return self.F, 1.
            p = 1.
            probs = get_new_probs(self.N, self.F, qs=self.qs)
            for o in self.ops:
                p *= o.succ_prob(probs)
                probs = o.apply(probs)
            self._probs = probs
            self._fidelity_and_succ_prob = fidelity(probs), p
            self._ABCD = ABCD(probs)
        return self._fidelity_and_succ_prob
    def ABCD_sym_notnorm(self, progress=True, separate_F=False):
        probs = get_new_probs_sym(self.N, separate_F)
        for i,o in enumerate(self.ops):
            if progress: print('\r %d/%d'%(i+1, len(self.ops)), end='', flush=True)
            probs = o.apply_sym(probs, separate_F=separate_F) if isinstance(o, Measurement) else o.apply_sym(probs)
        if progress: print('\rdone',flush=True)
        return ABCD(probs)
    def fidelity(self):
        return self.fidelity_and_succ_prob()[0]
    def succ_prob(self):
        return self.fidelity_and_succ_prob()[1]
    def ABCD(self):
        self.fidelity_and_succ_prob()
        return self._ABCD
    def probs(self):
        self.fidelity_and_succ_prob()
        return self._probs
    def fitness(self):
        if not self._fitness:
            if self.weights == 'yield':
                self._fitness = self.hashing_yield()
            else:
                f, p = self.fidelity_and_succ_prob()
                a,b,c,d = self.ABCD()
                try:
                    wf, wb, wc, wd, wp = self.weights
                    self._fitness = wf*f + wb*b + wc*c + wd*d + wp*p
                except ValueError:
                    wf, wp = self.weights
                    self._fitness = wf*f + wp*p
        return self._fitness
    def hashing_yield(self):
        f,p = self.fidelity_and_succ_prob()
        abcd = self.ABCD()
        n = self.raw_pairs()
        return p/n*(1+np.sum(np.log2(abcd)*abcd))
    @classmethod
    def random(cls, ops_len, permitted_ops, N, F, P2, Mη, weights, qs=None):
        return cls([random.choice(permitted_ops).random(N, F, P2, Mη, qs=qs) for _ in range(ops_len)], F,
                   history=History.random, weights=weights, qs=qs)
    def new_drop_op(self):
        ops = self.ops.copy()
        del ops[random.randint(0,len(ops)-1)]
        return Individual(ops, self.F, history=History.drop_m, weights=self.weights)
    def new_gain_op(self, permitted_ops, N, F, P2, Mη):
        ops = self.ops.copy()
        ops.insert(random.randint(0,len(ops)-1) if ops else 0, random.choice(permitted_ops).random(N, F, P2, Mη))
        return Individual(ops, self.F, history=History.gain_m,
                weights=self.weights, qs=self.qs)
    def new_swap_op(self):
        ops = self.ops.copy()
        i = random.randint(0,len(ops)-2)
        ops[i] = self.ops[i+1]
        ops[i+1] = self.ops[i]
        return Individual(ops, self.F, history=History.swap_m,
                weights=self.weights, qs=self.qs)
    def new_mutate(self, mutation_chance):
        return Individual([_.mutate() if random.random() < mutation_chance
                           else _
                           for _ in self.ops],
                          self.F,
                          history=History.ops_m,
                          weights=self.weights,
                          qs=self.qs)
    def new_child(self, p):
        ops1, ops2 = [self.ops, p.ops][::random.choice([1,-1])]
        i1, i2 = random.randint(0,len(ops1)), random.randint(0,len(ops2))
        return Individual(ops1[:i1]+ops2[i2:], self.F, history=History.child,
                weights=self.weights, qs=self.qs)
    def copy(self, F=None, P2=None, Mη=None, history=None, weights=None, qs=None):
        return Individual([_.copy(F=F, P2=P2, Mη=Mη) for _ in self.ops],
                          F=F if F else self.F,
                          history=history if history else self.history,
                          weights=weights if weights else self.weights,
                          qs=qs if qs else self.qs)
    @property
    def N(self):
        try:
            return self.ops[0].N
        except IndexError:
            return 0
    def rearrange(self, permutation_dict):
        new_ops = [_.rearrange(permutation_dict) for _ in self.ops]
        c = self.copy()
        c.ops = new_ops
        return c
    def canonical(self):
        assert assert_is_good(self.ops)
        cnotperm = False
        if any(isinstance(_, CNOTPerm) for _ in self.ops):
            cnotperm = True

        if not cnotperm:
            permutation_dict = {}
            done = set()
            for _ in (_ for _ in self.ops[::-1] if isinstance(_, Measurement)):
                if _.target in done:
                    continue
                done.add(_.target)
                permutation_dict[len(done)] = _.target
                if len(done) == self.N-1:
                    break
            permutation_dict[0] = 0
            assert len(set(permutation_dict.values())) == self.N
            reordered = self.rearrange(permutation_dict)
        else:
            reordered = self.copy()
            last_meas = sorted(reordered.ops[-self.N+1:], key=lambda _:_.target)
            reordered.ops = reordered.ops[:-self.N+1] + last_meas

        def measurement_before_gate(ops, start):
            for i, (measurement, next_gate) in enumerate(zip(ops[start:-1], ops[start+1:])):
                if (isinstance(measurement, Measurement)
                    and not isinstance(next_gate, Measurement)
                    and measurement.target not in next_gate.targets):
                    return i+start
            return 0
        i = measurement_before_gate(reordered.ops, 0)
        while i:
            reordered.ops[i], reordered.ops[i+1] = reordered.ops[i+1], reordered.ops[i]
            i = measurement_before_gate(reordered.ops, i)

        def higher_gate_before_lower_gate(ops):
            for i, (first_gate, next_gate) in enumerate(zip(ops[:-1], ops[1:])):
                if (not isinstance(first_gate, Measurement)
                    and not isinstance(next_gate, Measurement)
                    and not set(first_gate.targets).intersection(next_gate.targets)
                    and sorted(first_gate.targets) > sorted(next_gate.targets)):
                    return i
            return len(ops)
        i = higher_gate_before_lower_gate(reordered.ops)
        while i<reordered.N:
            reordered.ops[i], reordered.ops[i+1] = reordered.ops[i+1], reordered.ops[i]
            i = higher_gate_before_lower_gate(reordered.ops)

        reordered.fidelity_and_succ_prob(reval=True)
        assert np.sum(np.abs(reordered.probs() - self.probs())) < 1e-10

        return reordered
    def raw_pairs(self):
        return len([_ for _ in self.ops if isinstance(_, Measurement)]) + 1
    def len_parallel(self):
        backstep = 0
        last_ops = []
        for i,o in enumerate(self.ops):
            if last_ops and not set.intersection(set(o.targets), set.union(*(set(_.targets) for _ in last_ops))):
                backstep += 1
                last_ops.append(o)
            else:
                last_ops = [o]
        return len(self.ops) - backstep
    def len_nosingle(self):
        return len([_ for _ in self.ops if not isinstance(_,Permutation)])


def individual2plot(ops,title='',fig_axis=None,shading=False,parallelism=False):
    if isinstance(ops, Individual):
        ops = ops.ops
    N = ops[0].N
    L = len(ops)
    if fig_axis:
        f, s = fig_axis
    else:
        f = plt.figure(facecolor='white')
        s = f.add_subplot(111)
    f.set_size_inches((L+1),(N+1))
    s.clear()
    s.axis('off')

    z_shade = 0
    z_qubit = 10
    z_gate = 20

    if shading:
        shades_color = 'rgbcym'
        shades = list(range(N))
        for t, shade in enumerate(shades):
            s.add_artist(matplotlib.patches.Rectangle((0,t+0.5),0.5,1,
                                                  color=shades_color[shade],zorder=0,alpha=0.15))

    if not ops:
        return f

    backstep = 0
    last_ops = []
    for i,o in enumerate(ops):
        if parallelism and last_ops and not set.intersection(set(o.targets), set.union(*(set(_.targets) for _ in last_ops))):
            backstep += 1
            last_ops.append(o)
        else:
            last_ops = [o]
        i -= backstep
        s.add_artist(matplotlib.patches.Rectangle((i+0.5,0),1,N+1,
                                                  color='white',zorder=z_shade))
        if isinstance(o,(CNOT, CNOTPerm)):
            s.add_artist(matplotlib.patches.Circle((i+1,o.targets[0]+1),0.07,
                                                   color='black',lw=2,zorder=z_gate))
            s.add_artist(matplotlib.patches.Circle((i+1,o.targets[1]+1),0.3,
                                                   facecolor='white',edgecolor='black',lw=2,zorder=z_gate))
            s.add_artist(matplotlib.lines.Line2D([i+1-0.3,i+1+0.3],
                                                 [o.targets[1]+1,o.targets[1]+1],
                                                 color='black',lw=2,zorder=z_gate))
            s.add_artist(matplotlib.lines.Line2D([i+1,i+1],
                                                 [o.targets[1]+1-0.3,o.targets[1]+1+0.3],
                                                 color='black',lw=2,zorder=z_gate))
            sign = +1 if o.targets[0]>o.targets[1] else -1
            s.add_artist(matplotlib.lines.Line2D([i+1,i+1],[o.targets[0]+1,o.targets[1]+1+sign*0.3],
                                                 color='black',lw=2,zorder=z_gate))
            if isinstance(o, CNOTPerm):
                s.text(i+0.6,o.targets[0]+1,'\n'.join('ABCD'[_] for _ in o.permcontrol[1:]),fontsize=12,horizontalalignment='center',verticalalignment='center',zorder=z_gate+1)
                s.text(i+0.6,o.targets[1]+1,'\n'.join('ABCD'[_] for _ in o.permtarget[1:]),fontsize=12,horizontalalignment='center',verticalalignment='center',zorder=z_gate+1)
                s.add_artist(matplotlib.patches.Circle((i+1,o.targets[1]+1),0.25,
                                                       facecolor='white',edgecolor='black',lw=2,zorder=z_gate))
        elif isinstance(o,CPHASE):
            s.add_artist(matplotlib.patches.Circle((i+1,o.targets[0]+1),0.07,
                                                   color='black',lw=2,zorder=z_gate))
            s.add_artist(matplotlib.patches.Circle((i+1,o.targets[1]+1),0.07,
                                                   color='black',lw=2,zorder=z_gate))
            s.add_artist(matplotlib.lines.Line2D([i+1,i+1],[o.targets[0]+1,o.targets[1]+1],
                                                 color='black',lw=2,zorder=z_gate))
        elif isinstance(o,CPNOT):
            s.add_artist(matplotlib.patches.Circle((i+1,o.targets[0]+1),0.07,
                                                   color='black',lw=2,zorder=z_gate))
            s.add_artist(matplotlib.patches.Circle((i+1,o.targets[1]+1),0.3,
                                                   facecolor='white',edgecolor='black',lw=2))
            s.add_artist(matplotlib.patches.Circle((i+1,o.targets[1]+1),0.1,
                                                   color='black',lw=2,zorder=z_gate))
            s.add_artist(matplotlib.lines.Line2D([i+1-0.3,i+1+0.3],
                                                 [o.targets[1]+1,o.targets[1]+1],
                                                 color='black',lw=2,zorder=z_gate))
            s.add_artist(matplotlib.lines.Line2D([i+1,i+1],
                                                 [o.targets[1]+1-0.3,o.targets[1]+1+0.3],
                                                 color='black',lw=2,zorder=z_gate))
            sign = +1 if o.targets[0]>o.targets[1] else -1
            s.add_artist(matplotlib.lines.Line2D([i+1,i+1],[o.targets[0]+1,o.targets[1]+1+sign*0.3],
                                                 color='black',lw=2,zorder=z_gate))
        elif isinstance(o, Measurement):
            t = o.target+1
            s.add_artist(matplotlib.patches.Polygon([(i+0.5,t),(i+0.6,t+0.3),(i+1.3,t+0.3),(i+1.3,t-0.3),(i+0.6,t-0.3)],
                                                    facecolor='white',lw=2,edgecolor='black',zorder=z_gate))
            if i+backstep < L-N+1:
                s.add_artist(matplotlib.patches.Circle((i+1.45,t),0.09,facecolor='white',edgecolor='black',lw=2,zorder=z_gate))
            text = {(0,1):'antiY', (0,2):'coinX', (0,3):'coinZ'}.get(tuple(o.pair),
                    'ABCD'[o.pair[0]]+'ABCD'[o.pair[1]])
            s.text(i+0.94,t,text,fontsize=14,horizontalalignment='center',verticalalignment='center',zorder=z_gate+1)
        else:
            raise NotImplementedError
        if shading:
            if not isinstance(o,Measurement):
                shade = min(shades[_] for _ in o.targets)
                to_be_shaded = []
                for t in set(o.targets) - {shade}:
                    while True:
                        to_be_shaded.append(t)
                        if t == shades[t]:
                            break
                        t = shades[t]
                for t in to_be_shaded:
                    shades[t] = shade
            else:
                if shades[o.target] != o.target:
                    shades[o.target] = o.target
                else:
                    to_reset = [t for t, shade in enumerate(shades) if shade==shades[o.target] and t != o.target]
                    shade = min(to_reset)
                    for t in to_reset:
                        shades[t] = shade
            for t, shade in enumerate(shades):
                s.add_artist(matplotlib.patches.Rectangle((i+0.49,t+0.49),0.98,0.98,
                                                      facecolor=shades_color[shade],edgecolor='black',zorder=z_shade,alpha=0.15,lw=0.3))
    for i in range(N):
        s.add_artist(matplotlib.lines.Line2D([0,L+1-backstep],[i+1,i+1],
                         color='black',linewidth=2,zorder=1))
        s.add_artist(matplotlib.patches.Circle((0+0.045,i+1),0.09,
                         facecolor='white',edgecolor='black',lw=2,zorder=z_gate))


    s.text(0,0.5,title,fontsize=15)
    s.set_xlim(0,L+1)
    s.set_ylim(0,N+1)
    s.invert_yaxis()

    return f


def translate_individual(self, trans_dict_cgates, trans_dict_measurements):
    ind = self.copy()
    new_ops = []
    for o in ind.ops:
        if isinstance(o, (CNOT, CPHASE, CPNOT)):
            new_ops.append(trans_dict_cgates[type(o)](*o.args))
        elif isinstance(o, Measurement):
            new_o = Measurement(o.target, trans_dict_measurements[tuple(o.pair)], o.N, o.F, o.Mη)
            new_ops.append(new_o)
        else:
            raise NotImplementedError
    ind.ops = new_ops
    return ind


class Population:
    def __init__(self,
                 N,
                 F,
                 P2,
                 Mη,
                 WEIGHTS,
                 POPULATION_SIZE,
                 STARTING_POP_MULTIPLIER,
                 MAX_GEN,
                 MAX_OPS,
                 STARTING_OPS,
                 PERMITTED_OPS,
                 PAIRS,
                 CHILDREN_PER_PAIR,
                 MUTANTS_PER_INDIVIDUAL_PER_TYPE,
                 P_SINGLE_OPERATION_MUTATES,
                 P_LOSE_OPERATION,
                 P_ADD_OPERATION,
                 P_SWAP_OPERATIONS,
                 P_MUTATE_OPERATIONS):
        self.N = N
        self.F = F
        self.P2 = P2
        self.Mη = Mη
        self.WEIGHTS = WEIGHTS
        self.POPULATION_SIZE,                = POPULATION_SIZE,
        self.STARTING_POP_MULTIPLIER         = STARTING_POP_MULTIPLIER
        self.MAX_GEN                         = MAX_GEN
        self.MAX_OPS                         = MAX_OPS
        self.STARTING_OPS                    = STARTING_OPS
        self.PERMITTED_OPS                   = PERMITTED_OPS
        self.PAIRS                           = PAIRS
        self.CHILDREN_PER_PAIR               = CHILDREN_PER_PAIR
        self.MUTANTS_PER_INDIVIDUAL_PER_TYPE = MUTANTS_PER_INDIVIDUAL_PER_TYPE
        self.P_SINGLE_OPERATION_MUTATES      = P_SINGLE_OPERATION_MUTATES
        self.P_LOSE_OPERATION                = P_LOSE_OPERATION
        self.P_ADD_OPERATION                 = P_ADD_OPERATION
        self.P_SWAP_OPERATIONS               = P_SWAP_OPERATIONS
        self.P_MUTATE_OPERATIONS             = P_MUTATE_OPERATIONS
        print('Initializing %d individuals. Keeping only %d of them.'%(
              self.POPULATION_SIZE*self.STARTING_POP_MULTIPLIER, self.POPULATION_SIZE),
              flush=True)
        self.l = [Individual.random(self.STARTING_OPS, self.PERMITTED_OPS,
                                    self.N, self.F, self.P2, self.Mη,
                                    self.WEIGHTS)
                  for _ in range(self.POPULATION_SIZE*self.STARTING_POP_MULTIPLIER)]
        self._sort()
        self._cull()
        self.generations = 0
        self.besthistory = np.zeros(self.MAX_GEN)
        self.worsthistory = np.zeros(self.MAX_GEN)
        self.lenhistory = np.zeros(self.MAX_GEN, dtype=int)
        self.selhistory = np.zeros((self.MAX_GEN,len(History)), dtype=int)
        self._record_stats()
        if _gui:
            self._f = plt.figure(figsize=(12,4))
            self._f_current = self._f.add_subplot(2,2,1)
            self._f_history = self._f.add_subplot(2,2,2)
            self._f_lenh = self._f_history.twinx()
            self._f_selection = self._f.add_subplot(2,2,4)
            self._f_1st = plt.figure()
            self._f_1st_s = self._f_1st.add_subplot(1,1,1)
            self._f_2nd = plt.figure()
            self._f_2nd_s = self._f_2nd.add_subplot(1,1,1)
            self._f_lst = plt.figure()
            self._f_lst_s = self._f_lst.add_subplot(1,1,1)
    def _sort(self):
        self.l.sort(key=lambda _:_.fitness(), reverse=True)
    def _record_stats(self):
        self.besthistory[self.generations] = self.l[0].fitness()
        self.worsthistory[self.generations] = self.l[-1].fitness()
        self.lenhistory[self.generations] = len(self.l[0].ops)
        for i in History:
            self.selhistory[self.generations, i.value] = sum(1 for _ in self.l if _.history==i)
    def _cull(self):
        self.l = self.l[:self.POPULATION_SIZE]
    def generation_step(self):
        for s in self.l:
            s.history = History.survivor
        parents = int(0.5*((4*self.PAIRS+1)**0.5+1)) + 2
        print('\rGenerating %d children per parent pair (%d pairs).'%(
              self.CHILDREN_PER_PAIR, self.PAIRS), flush=True)
        for i, (p1, p2) in enumerate(
                           itertools.islice(
                           itertools.combinations(random.sample(self.l,parents), r=2),self.PAIRS)):
            if p1 == p2:
                continue
            for j in range(self.CHILDREN_PER_PAIR):
                child = p1.new_child(p2)
                while len(child.ops) > self.MAX_OPS:
                    child = p1.new_child(p2)
                self.l.append(child)
                #print('\r    pair %d    child %d'%(i,j), end='', flush=True)
        print('\rGenerating %d mutants per individual per mutation type (weighted by type).'%self.MUTANTS_PER_INDIVIDUAL_PER_TYPE,
              flush=True)
        for i in range(self.POPULATION_SIZE):
            individual = self.l[i]
            #print('\r    individual %d'%i, end='', flush=True)
            for _ in range(self.MUTANTS_PER_INDIVIDUAL_PER_TYPE):
                if random.random() < self.P_LOSE_OPERATION and len(individual.ops) > 0:
                    self.l.append(individual.new_drop_op())
                if random.random() < self.P_ADD_OPERATION and len(individual.ops) < self.MAX_OPS:
                    self.l.append(individual.new_gain_op(self.PERMITTED_OPS, self.N, self.F, self.P2, self.Mη))
                if random.random() < self.P_SWAP_OPERATIONS and len(individual.ops) > 1:
                    self.l.append(individual.new_swap_op())
                if random.random() < self.P_MUTATE_OPERATIONS and individual.ops:
                    self.l.append(individual.new_mutate(self.P_SINGLE_OPERATION_MUTATES))
        print('\rEvaluating fitness and culling.', flush=True)
        self._sort()
        self._cull()
        self.generations += 1
        self._record_stats()
    def report(self):
        if _gui:
            self._f_current.clear()
            self._f_history.clear()
            self._f_lenh.clear()
            self._f_selection.clear()
            self._f_current.plot([_.fitness() for _ in self.l],'k-',linewidth=1.5)
            if self.generations > 1:
                if self.WEIGHTS == 'yield':
                    self._f_history.semilogy(self.besthistory[:self.generations+1],'g-', label='best')
                    self._f_history.semilogy(self.worsthistory[:self.generations+1],'r-', label='worst')
                else:
                    self._f_history.semilogy(1-self.besthistory[:self.generations+1]/self.besthistory[self.generations],'g-', label='best')
                    self._f_history.semilogy(1-self.worsthistory[:self.generations+1]/self.besthistory[self.generations],'r-', label='worst')
            self._f_lenh.plot(self.lenhistory[:self.generations+1],'k-')
            self._f_selection.plot(self.selhistory[:self.generations+1,3:])
            self._f_selection.legend([_.name for _ in History][3:], bbox_to_anchor=(1.02, 0,0.25,1), loc=2,
                                     ncol=1, mode="expand", borderaxespad=0.)
            self._f_current.set_xlabel("Circuit's id number")
            self._f_current.set_ylabel("Fitness")
            self._f_history.set_xlabel("Generation")
            self._f_history.set_ylabel("Normalized Fitness")
            self._f_lenh.set_ylabel("Length of Best Circuit")
            self._f.tight_layout()
            title = lambda ind: 'L=%s, fitness=%.6f, F=%.6f, P=%.6f'%(len(ind.ops),ind.fitness(), ind.fidelity(), ind.succ_prob())
            try:
                individual2plot(self.l[0].ops,
                    title(self.l[0]),
                    (self._f_1st,self._f_1st_s), shading=True)
                individual2plot(self.l[1].ops,
                    title(self.l[1]),
                    (self._f_2nd,self._f_2nd_s), shading=True)
                individual2plot(self.l[-1].ops,
                    title(self.l[-1]),
                    (self._f_lst,self._f_lst_s))
            except Exception as e:
                print(e)
            display.clear_output(wait=True)
            display.display(self._f, self._f_1st)
            display.display(self._f_2nd)
            display.display(self._f_lst)
        print('Generation: %d    Population: %d    Δ: %e'%(
                self.generations, self.POPULATION_SIZE, self.l[0].fitness()-self.l[-1].fitness()), flush=True)
    def run(self, min_delta=1e-15):
        self.report()
        try:
            while self.generations < self.MAX_GEN-1:
                self.generation_step()
                self.report()
                if self.l[0].fitness()-self.l[-1].fitness()<=min_delta:
                    print('The difference between best and worst is less than %e. Quiting.'%min_delta)
                    break
        except KeyboardInterrupt:
            self._cull()
            self.report()
            print('Interrupted by user.', flush=True)


from collections import OrderedDict
default_config = OrderedDict([
('N', 3),
('F', 0.9),
('P2', 0.99),
('Mη', 0.99),

('WEIGHTS', (1,0)),

('STARTING_OPS', 10),
('MAX_OPS', 8),
('PERMITTED_OPS', [CNOTPerm, AMeasurement]),

('POPULATION_SIZE', 200),
('STARTING_POP_MULTIPLIER', 10),

# 4 types of mutations
('MUTANTS_PER_INDIVIDUAL_PER_TYPE', 1),
('P_ADD_OPERATION', 0.7),
('P_LOSE_OPERATION', 0.9),
('P_SWAP_OPERATIONS', 0.8),
('P_MUTATE_OPERATIONS', 0.8), # adjustment in the next variabl
('P_SINGLE_OPERATION_MUTATES', 0.1),

# parent pairs
('PAIRS', 20),
('CHILDREN_PER_PAIR', 10),

# for logging purposes we assume we are never taking more than MAX_GEN generations
('MAX_GEN', 500),
])


def double_select(f, p, η=None):
    if not η:
        η = p
    mx = lambda t: Measurement(t, [0,2], N=3, F=f, Mη=η)
    mz = lambda t: Measurement(t, [0,3], N=3, F=f, Mη=η)
    return Individual([CNOT([0,1], N=3, P2=p),
                       CNOT([2,1], N=3, P2=p),
                       mz(1), mx(2)],
                      f)

def double_select_twice(f, p, η=None):
    ds = double_select(f, p,η)
    return Individual(ds.ops*2, f)

def triple_select(f, p, η=None):
    if not η:
        η = p
    return Individual(F=f, ops=[
           CNOT([0, 1], 4, p),
           CNOT([1, 2], 4, p),
           CNOT([3, 1], 4, p),
           AMeasurement(3, 1, 4, f, η),
           AMeasurement(2, 1, 4, f, η),
           AMeasurement(1, 1, 4, f, η)])

def expedient(f, p, η=None):
    if not η:
        η = p
    m = lambda t: Measurement(t, [0,2], N=3, F=f, Mη=η)
    long = lambda gate: [
        gate([1,0], N=3, P2=p),
        CPHASE([2,1], N=3, P2=p),
        m(1),
        m(2)]
    return Individual(long(CPHASE)+long(CNOT), f)

def stringent(f, p, η=None):
    if not η:
        η = p
    m = lambda t: Measurement(t, [0,2], N=3, F=f, Mη=η)
    long = lambda gate: [
        gate([1,0], N=3, P2=p),
        CPHASE([2,1], N=3, P2=p),
        m(1),
        m(2)]
    long_dot = lambda gate: [
        CNOT([2,1], N=3, P2=p),
        m(2),
        CPHASE([2,1], N=3, P2=p),
        m(2),
        gate([1,0], N=3, P2=p),
        CPHASE([2,1], N=3, P2=p),
        m(2),
        m(1)]
    return Individual(long(CPHASE)+long(CNOT)+long_dot(CPHASE)+long_dot(CNOT), f)

def expedient_2Nsubcircuit(f, p, η=None):
    if not η:
        η = p
    m = Measurement(1, [0,2], N=2, F=f, Mη=η)
    cp = CPHASE([1,0], N=2, P2=p)
    cn = CNOT([1,0], N=2, P2=p)
    return Individual([cn,m,cp,m], f)


rx90 = lambda t: Permutation(t,[0,3,2,1],3)
def deutsch(f, p, η=None):
    if not η:
        η = p
    return Individual([
            rx90(0), rx90(1), CNOT([0,1],3,p), Measurement(1, [0,3], N=3, F=f, Mη=η)
        ], f)

def deutsch2(f, p, η=None):
    return Individual(deutsch(f,p,η).ops*2, f)

def deutsch3(f, p, η=None):
    return Individual(deutsch(f,p,η).ops*3, f)

def deutsch4(f, p, η=None):
    return Individual(deutsch(f,p,η).ops*4, f)

def deutsch5(f, p, η=None):
    return Individual(deutsch(f,p,η).ops*5, f)

def nested_deutsch(f, p, η=None):
    if not η:
        η = p
    deutsch_ops = lambda purify, sacrifice: [rx90(purify),
                                             rx90(sacrifice),
                                             CNOT([purify,sacrifice],3,p),
                                             Measurement(sacrifice, [0,3], N=3, F=f, Mη=η)]
    return Individual(deutsch_ops(0,2)+deutsch_ops(1,2)+deutsch_ops(0,1),
                      f)

def nested_deutsch2(f, p, η=None):
    if not η:
        η = p
    deutsch_ops = lambda purify, sacrifice: [rx90(purify),
                                             rx90(sacrifice),
                                             CNOT([purify,sacrifice],3,p),
                                             Measurement(sacrifice, [0,3], N=3, F=f, Mη=η)]
    return Individual(deutsch_ops(0,2)*2+deutsch_ops(1,2)*2+deutsch_ops(0,1),
                      f)

def contains_two_measurements(ops):
    return any(isinstance(l, Measurement) and isinstance(r, Measurement) and l.target == r.target
               for l, r in zip(ops[:-1], ops[1:]))

def contains_nonA_measurements(ops):
    return any(isinstance(_, Measurement) and _.pair[0] != 0 for _ in ops)

def contains_untargetted_qubits(ops):
    N = ops[0].N
    return len(set.union(*(set(_.targets) for _ in ops if len(_.targets)>1))) != N

def contains_nonmeasurement_last_steps(ops):
    N = ops[0].N
    measured = {0}
    for o in ops[::-1]:
        if isinstance(o, Measurement):
            measured.add(o.target)
        else:
            if not measured.issuperset(o.targets):
                return True
        if len(measured) == N:
            return False
    raise Exception('should not happen')

def contains_measurements_on_top_qubit(ops):
    return any(_.target==0 for _ in ops if isinstance(_, Measurement))

def assert_is_good(ops):
    return (not isinstance(ops[0], Measurement)
        and not contains_two_measurements(ops)
        and not contains_nonA_measurements(ops)
        and not contains_untargetted_qubits(ops)
        and not contains_measurements_on_top_qubit(ops)
        and not contains_nonmeasurement_last_steps(ops))

def filter_and_sort_by_length(individuals):
    return [{_ for _ in individuals
               if len(_.ops)==l and len(_.ops)
                  and assert_is_good(_.ops)
            }
            for l in range(41)]

def taylor_denom(s, order=3):
    s -= 1
    s = sum((-s)**i for i in range(order+1))
    return s

def poly_cut(p, orders):
    if isinstance(orders, tuple):
        p = sympy.Poly.from_dict({k:v for k,v in p.terms() if all(l<=r for l,r in zip(k, orders))},
                                 *p.gens,
                                 domain=p.domain)
    else:
        p = sympy.Poly.from_dict({k:v for k,v in p.terms() if sum(k)<=orders},
                                 *p.gens,
                                 domain=p.domain)
    return p

def poly_gm_to_pη(p):
    gate_to_p2 = {epsilon_g: 2*eps_p2 -   eps_p2**2}
    meas_to_η  = {epsilon_m: 2*eps_η  - 2*eps_η **2}
    p = p.as_expr().subs(gate_to_p2).subs(meas_to_η)
    p = sympy.poly(p, symF, q, eps_p2, eps_η, domain='QQ')
    return p

def poly_pη_to_ε(p):
    p = p.as_expr().subs(eps_p2, eps).subs(eps_η, eps)
    return sympy.poly(p, symF, q, eps, domain='QQ')

def expand_ABCDsymnotnorm(ABCD, beyond_leading_order=0):
    # out of the leading orders `n` for `q^n*eps^0` in B,C,D pick the highest number
    offset = 1 if symF in ABCD[0].gens else 0
    leading_order_q      = max(min(monom[0+offset]   for monom in component.monoms()
                                              if monom[0+offset] and monom[1+offset:]==(0,0))
                               for component in ABCD[1:])
    # for that order of q, pick the total q-eps-eps order that is the lowest
    leading_order_with_q = min(min(sum(monom) for monom in component.monoms()
                                              if monom[0+offset]==leading_order_q)
                               for component in ABCD[1:])
    order = leading_order_with_q + beyond_leading_order
    ABCD = [poly_gm_to_pη(poly_cut(component, order))
            for component in ABCD]
    s = sum(ABCD)
    assert s.TC() == 1, 'For some error the sum of A,B,C,D does not go to 1 in the limit.'
    s = taylor_denom(s, order=order)
    ABCD = [poly_cut(_*s, order) for _ in ABCD]
    return ABCD

def poly2latex(poly):
    def key2latex(*args):
        if len(args)==4:
            _,q,p2,eta = args
            return ((r'\color{darkred}{q^{%d}}'%q   if q  >1 else (r'\color{darkred}{q}' if q   else ''))+
                    (r'\varepsilon_{p2}^{%d}'  %p2  if p2 >1 else (r'\varepsilon_{p2}'   if p2  else ''))+
                    (r'\varepsilon_{\eta}^{%d}'%eta if eta>1 else (r'\varepsilon_{\eta}' if eta else '')))
        elif len(args)==3:
            _,q,eps = args
            return ((r'\color{darkred}{q^{%d}}'%q   if q  >1 else (r'\color{darkred}{q}' if q   else ''))+
                    (r'\varepsilon^{%d}'       %eps if eps>1 else (r'\varepsilon'        if eps else '')))
        else:
            _,q = args
            return r'\color{darkred}{q^{%d}}'%q   if q  >1 else (r'\color{darkred}{q}' if q   else '')
    return ''.join(
        ('+' if v>0 else '')+sympy.latex(v)+key2latex(*k)
        for k,v in reversed(poly.terms(order='grlex')))
