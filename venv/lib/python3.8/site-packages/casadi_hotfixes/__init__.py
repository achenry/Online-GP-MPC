import casadi
import types

for cl in [casadi.MX, casadi.SX, casadi.DM]:

  # Workaround for https://github.com/casadi/casadi/issues/2625

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    conversion = {"multiply": "mul", "divide": "div", "true_divide": "div", "subtract":"sub","power":"pow","greater_equal":"ge","less_equal": "le", "less": "lt", "greater": "gt"}
    name = ufunc.__name__
    inputs = list(inputs)
    if len(inputs)==3:
      raise Exception("Error with %s. Looks like you are using an assignment operator, such as 'a+=b' where 'a' is a numpy type. This is not supported, and cannot be supported without changing numpy." % name)
    if "vectorized" in name:
        name = name[:-len(" (vectorized)")]
    if name in conversion:
      name = conversion[name]
    if len(inputs)==2 and inputs[1] is self and not(inputs[0] is self):
      name = 'r' + name
      inputs.reverse()
    if not(hasattr(self,name)) or ('mul' in name):
      name = '__' + name + '__'
    fun=getattr(self, name)
    return fun(*inputs[1:])

  cl.__array_ufunc__ = __array_ufunc__
