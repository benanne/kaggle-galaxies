import theano
import theano.tensor as T
from theano.tensor.opt import register_canonicalize

# TODO: implement w.r.t.?

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [g_out.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()

register_canonicalize(theano.gof.OpRemove(consider_constant), name='remove_consider_constant_')


if __name__=='__main__':
    import theano.tensor as T
    import numpy as np


    x = T.matrix('x')
    x_c = consider_constant(x)

    g = T.grad((x * T.exp(x)).sum(), x)

    f = theano.function([x], g) # should always return 1

    g_c = T.grad((x * T.exp(x_c)).sum(), x)

    f_c = theano.function([x], g_c) # should always return 0

    a = np.random.normal(0, 1, (3,3)).astype("float32")

    print f(a)
    print f_c(a)
    print np.exp(a) * (a + 1)
    print np.exp(a)


    theano.printing.debugprint(f_c)



#########

# WITHOUT CANONICALIZATION
# DeepCopyOp [@A] ''   1
#  |ConsiderConstant [@B] ''   0
#    |x [@C]

# Elemwise{exp} [@A] ''   1
#  |ConsiderConstant [@B] ''   0
#    |x [@C]


# WITH CANONICALIZATION
# DeepCopyOp [@A] 'x'   0
#  |x [@B]

# Elemwise{exp} [@A] ''   0
#  |x [@B]






# class ConsiderConstant(ViewOp):
#     def grad(self, args, g_outs):
#         return [tensor.zeros_like(g_out) for g_out in g_outs]
# consider_constant_ = ConsiderConstant()


# # Although the op just returns its input, it should be removed from
# # the graph to make sure all possible optimizations can be applied.
# register_canonicalize(gof.OpRemove(consider_constant_),
#     name='remove_consider_constant')


# #I create a function only to have the doc show well.
# def consider_constant(x):
#     """ Consider an expression constant when computing gradients.

#     The expression itself is unaffected, but when its gradient is
#     computed, or the gradient of another expression that this
#     expression is a subexpression of, it will not be backpropagated
#     through. In other words, the gradient of the expression is
#     truncated to 0.

#     :param x: A Theano expression whose gradient should be truncated.

#     :return: The expression is returned unmodified, but its gradient
#         is now truncated to 0.

#     Support rectangular matrix and tensor with more than 2 dimensions
#     if the later have all dimensions are equals.

#     .. versionadded:: 0.6.1
#     """
#     return consider_constant_(x)
#     