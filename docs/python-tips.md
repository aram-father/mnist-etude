This document summarizes useful tips regarding python(or its internal/external libraries)

### package structure

- In python, a module is a file ending with `.py`
- Package helps developers manage a lot of modules hierarchically
- Pacakge is consist of **modules** and **directories**
- `__init__.py` let the interpreter know this directory is a part of a pacakge
  - Users could import a specific directory of a pacakge like `from example_package.example_directory import *`
  - Developer should specify modules that want to be exported
    - `__all__ = ['example_module']` or `from . import example_module`


### np.max vs. np.maximum(np.min vs. np.minimum)

- `np.max(ex_arr, axis=0)` returns an array consist of the maximum element along the specified axis
- `np.maximum(ex_arr0, ex_arr1, axis=0)` returns an array consist of the bigger element of the two input arrays

### boolean operation on numpy array

- A boolean operation on a numpy array retruns an array consist of boolean value

### astype

- `ex_arr.astype(np.int)` returns a type converted array


### reshape

- `ex_arr.reshape(-1, (1, 2, 2))` returns a reshaped array
- `-1` indicates that the actual size is determined by the following dimension size

### np.prod

- `np.prod(ex_arr, axis=0)` returns the product of array elements over a given axis

### np.random.choice

- `np.random.choice(60000, 3)` returns an array like `[55, 321, 2]`

### np.nditer

- `np.nditer(ex_arr, flags=['multi_index'], op_flags=['readwrite'])` helps iterating an array in a specific order
- The first option make multi-dimensional index retrievning be possible
- The second option means we could modify the array we are iterating(if not designated we could not modify it)

```python
it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished
  idx = it.multi_index
  it = it.iternext()
```