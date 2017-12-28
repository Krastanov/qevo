module Clifford

export PauliOperator, CliffordOperator, @p_str, ⊗, commutes, clifford_group, expand_2toN, mul!

t,f = true,false
ops_F2 = [[f,f],[f,t],[t,f],[t,t]]
ops_chars = ['I', 'Z', 'X', 'Y']
d_F2 = Dict(zip(ops_chars, ops_F2))
drev_F2 = Dict(zip(ops_F2, ops_chars))
d_ind = Dict(zip(ops_chars, 1:4))

################################################################################
# PauliOperator and constructors
################################################################################

mutable struct PauliOperator
    sign::Int
    singlequbitops::Array{Int,1}
end

function PauliOperator(operators::String)
    if startswith(operators, "+")
        sign = 0
    elseif startswith(operators, "i")
        sign = 1
    elseif startswith(operators, "-i")
        sign = 3
    elseif startswith(operators, "-")
        sign = 2
    else
        sign = 0
    end
    singlequbitops = [d_ind[op] for op in strip(operators,['+','-', 'i'])]
    PauliOperator(sign, singlequbitops)
end

macro p_str(p)
    PauliOperator(p)
end

################################################################################
# methods for PauliOperator
################################################################################

multiplication_table = Array{Tuple{Int,Int},2}(
#   I     Z     X     Y  <-- second argument
[[(0,1) (0,2) (0,3) (0,4)]; # I  <-- first argument
 [(0,2) (0,1) (1,4) (3,3)]; # Z
 [(0,3) (3,4) (0,1) (1,2)]; # X
 [(0,4) (1,3) (3,2) (0,1)]] # Y
)

commutation_table = Array{Int,2}(
#   I     Z     X     Y  <-- second argument
[[  0     0     0     0  ]; # I  <-- first argument
 [  0     0     1     1  ]; # Z
 [  0     1     0     1  ]; # X
 [  0     1     1     0  ]] # Y
)

function Base.:*(l::PauliOperator, r::PauliOperator)
    len = length(l.singlequbitops)
    s_ops = Array{Tuple{Int,Int}, 1}(len)
    for i in 1:len
        s_ops[i] = multiplication_table[l.singlequbitops[i],r.singlequbitops[i]]
    end
    PauliOperator(
        (l.sign+r.sign+sum(s_op[1] for s_op in s_ops))%4,
        [s_op[2] for s_op in s_ops])
end

⊗(l::PauliOperator, r::PauliOperator) = PauliOperator(
                                            (l.sign + r.sign)%4,
                                            [l.singlequbitops; r.singlequbitops])

function commutes(l::Array{Int,1}, r::Array{Int,1})::Bool
    com = sum(commutation_table[li,ri] for (li,ri) in zip(l, r))
    return com%2 == 0
end

commutes(l::PauliOperator, r::PauliOperator)::Bool = commutes(l.singlequbitops, r.singlequbitops)

Base.show(io::IO, p::PauliOperator) = print(io, ["+","i","-","-i"][p.sign+1], mapslices(x->ops_chars[x], p.singlequbitops, 1)...);

Base.length(p::PauliOperator) = length(p.singlequbitops)

Base.ones(p::PauliOperator) = PauliOperator(0, ones(p.singlequbitops))

SparseArrays.permute(p::PauliOperator, perm::Array{Int, 1}) = PauliOperator(p.sign, p.singlequbitops[perm])

Base.hash(p::PauliOperator, h::UInt) = hash(p.sign, hash(p.singlequbitops, h))

Base.:(==)(l::PauliOperator, r::PauliOperator) = l.sign==r.sign && l.singlequbitops==r.singlequbitops

################################################################################
# ClifforOperator and constructors
################################################################################

struct CliffordOperator
    zimages::Array{PauliOperator,1}
    ximages::Array{PauliOperator,1}
end

################################################################################
# methods for CliffordOperator
################################################################################

function Base.show(io::IO, c::CliffordOperator)
    len = length(c.zimages)
    for (i, im) in enumerate(c.zimages)
        print(io, PauliOperator(0, setindex!(ones(Int, len), 2, i)), " → ", im, '\n')
    end
    for (i, im) in enumerate(c.ximages)
        print(io, PauliOperator(0, setindex!(ones(Int, len), 3, i)), " → ", im, '\n')
    end
end

function ⊗(l::CliffordOperator, r::CliffordOperator)
    lones = ones(l.zimages[1])
    rones = ones(r.zimages[1])
    
    CliffordOperator(
        vcat([op⊗rones for op in l.zimages], [lones⊗op for op in r.zimages]),
        vcat([op⊗rones for op in l.ximages], [lones⊗op for op in r.ximages]))
end

################################################################################
# mixed methods
################################################################################

function Base.:*(p::PauliOperator, sign::Int)
    PauliOperator((p.sign+sign)%4, p.singlequbitops)
end

function Base.:*(c::CliffordOperator, p::PauliOperator)
    sign = p.sign
    unsigned = prod((i_op for i_op in enumerate(p.singlequbitops) if i_op[2]≠1)) do i_op
        i,op = i_op
        if op==2
            c.zimages[i]
        elseif op==3
            c.ximages[i]
        else
            sign += 3
            c.zimages[i]*c.ximages[i]
        end
    end
    unsigned*p.sign
end

################################################################################
# fast in-place methods
################################################################################

multiplication_table_s = Array{Int,2}(
# I Z X Y  <-- second argument
[[0 0 0 0]; # I  <-- first argument
 [0 0 1 3]; # Z
 [0 3 0 1]; # X
 [0 1 3 0]] # Y
)

multiplication_table_op = Array{Int,2}(
# I Z X Y  <-- second argument
[[1 2 3 4]; # I  <-- first argument
 [2 1 4 3]; # Z
 [3 4 1 2]; # X
 [4 3 2 1]] # Y
)

function mul!(l::PauliOperator, r::PauliOperator, pans::PauliOperator)
    pans.sign = l.sign+r.sign
    for i in 1:length(l.singlequbitops)
        #pans.sign += multiplication_table_s[l.singlequbitops[i],r.singlequbitops[i]]
        if l.singlequbitops[i] != 1 && r.singlequbitops[i] != 1 && l.singlequbitops[i] != r.singlequbitops[i]
            pans.sign += ((3+l.singlequbitops[i]-r.singlequbitops[i])%3-1)*2+3
        end
        #pans.singlequbitops[i] = multiplication_table_op[l.singlequbitops[i],r.singlequbitops[i]]
        if l.singlequbitops[i] % 2 == 1
            pans.singlequbitops[i] = (l.singlequbitops[i]+r.singlequbitops[i]+2)%4+1
        else
            pans.singlequbitops[i] = (l.singlequbitops[i]-r.singlequbitops[i]+4)%4+1
        end
    end
    pans.sign %= 4
    nothing
end             

function mul!(c::CliffordOperator, p::PauliOperator, pans::PauliOperator)
    # XXX Unsafe if p and pans are the same!
    pans.sign = p.sign
    pans.singlequbitops = ones(p.singlequbitops)
    for i in 1:length(p.singlequbitops)
        if p.singlequbitops[i] == 1
            continue
        elseif p.singlequbitops[i] == 2
            mul!(pans, c.zimages[i], pans)
        elseif p.singlequbitops[i] == 3
            mul!(pans, c.ximages[i], pans)
        else
            mul!(pans, c.zimages[i], pans)
            mul!(pans, c.ximages[i], pans)
            pans.sign = (pans.sign+3)%4
        end
    end
    nothing
end

################################################################################
# Enumerating the Clifford group
################################################################################

using IterTools
using Combinatorics

_images(n) = ([ops...]
              for ops
              in Iterators.drop(product((1:4 for i in 1:n)...),1))

"""
IIII...X -> σσSσ...σ where S≠I
generate arbitrary IIII...Z -> σσ_σ...σ and fill in the blank so that it anticommutes
"""
function _anticomm_constrained_images(n::Int, im::Array{Int,1})
    if n==1
        return [[[3],[4]],[[4],[2]],[[2],[3]]][im[1]-1]
    end
    nonIind = findfirst(x->x!=1, im)
    nonI = im[nonIind]
    im_except = deleteat!(copy(im), nonIind)
    function filler(new_im_except::Array{Int,1})::Tuple{Int,Int}
        com = commutes(im_except, new_im_except)
        if !com
            return (1, nonI)
        elseif nonI == 2
            return (3,4)
        elseif nonI == 3
            return (4,2)
        else
            return (2,3)
        end
    end
    iterator = ([ops...] for ops in product((1:4 for i in 1:n-1)...))
    return (insert!(copy(ops), nonIind, f)
            for ops in iterator
            for f in filler(ops))
end
            
_imagepairs(n::Int) = ((Xim, Zim)
                       for Xim
                       in _images(n)
                       for Zim
                       in _anticomm_constrained_images(n, Xim))
                        
function _unsigned_unpermuted_clifford_group(n::Int)
    if n==1
        return _imagepairs(1)
    end
    pool = collect(_imagepairs(n))
    comb = combinations(pool, n)
    # In the above combination, no two pairs will be the same,
    # but we can still have  (a) [(A, _), (A, _), ...] or (b) [(A, _), (_, A), ...]
    # which is not a Clifford operator (not injective).
    # However, both will fail the commutative test, because
    # anticommutation is already enforced inside the pair.
    # It would still be nice if we can completely skip generating
    # those though.
    pred(c) = all(zxpairs->all(zx->commutes(zx[1],zx[2]),
                               product(zxpairs[1],zxpairs[2])),
                  combinations(c,2))
    return Iterators.filter(pred, comb)
end

function _permute(n::Int, unpermuted)
    Iterators.flatten(permutations(unp) for unp in unpermuted)
end
                                    
function clifford_group(n::Int)
    unsigned_group = collect(_permute(n, _unsigned_unpermuted_clifford_group(n)))
    sign_perms = product(([0,2] for i in 1:2n)...)
    return (CliffordOperator(
                [PauliOperator(s, zx[1]) for (s,zx) in zip(signs[1:n],zximages)],
                [PauliOperator(s, zx[2]) for (s,zx) in zip(signs[n+1:end],zximages)])
            for signs in sign_perms
            for zximages in unsigned_group)
end
                                                
signs(n) = (2*n)^2
perms(n) = factorial(n)
signperms(n) = signs(n)*perms(n)

################################################################################
# Helpers
################################################################################
function expand_2toN(n, q1, q2, p::PauliOperator)
    Is = PauliOperator(0, ones(Int, n-2))
    perm = collect(1:n)
    if q1==2
        perm[[1,2,q2]] = perm[[q2,1,2]]        
    elseif q2==1
        perm[[1,2,q1]] = perm[[2,1,q1]]        
    else
        perm[[1,2,q1,q2]] = perm[[q1,q2,1,2]]
    end
    return permute(p⊗Is, perm)
end

end
