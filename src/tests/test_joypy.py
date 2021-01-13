import joypy
import numpy as np
import pandas
import matplotlib.pyplot as plt

a = np.array([[2, 2, 2, 2, 2, 3], [3, 3, 3, 3, 3, 4]]).transpose()
df = pandas.DataFrame(a, columns=['a', 'b'])


fig, axes = joypy.joyplot(df)
plt.show()
