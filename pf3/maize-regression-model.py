import pandas as pd, numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

reg_df = pd.read_stata('SourceData/RegData.dta')

gm_df = reg_df[reg_df['AGE'] == 1].copy()

yearly = gm_df.groupby('release')['Yield'].agg(['mean', 'std', 'count']).reset_index()

x = yearly['release'].values - 2000 #RLYR offset
y = yearly['mean'].values

w = yearly['count'].values

"""def model_lin(th, x):
    return th[0] + th[1]*x

def model_quad(th, x):
    return th[0] + th[1]*x + th[2]*(x**2)

def model_cube(th, x):
    return th[0] + th[1]*x + th[2]*(x**2) + th[3]*(x**3)"""

def model(th, x):
    return sum(th[i] * x**i for i in range(len(th)))

def mse_loss(th, x, y):
    return np.mean((y - model(th, x))**2)

def huber_loss(th, x, y, delta=0.5):
    r = y - model(th, x)
    abs_r = np.abs(r)
    per_point_loss = np.where(
        abs_r <= delta,
        0.5 * r**2,
        delta * (abs_r - 0.5 * delta)
    )
    return np.mean(per_point_loss)

def wmse_loss(th, x, y, w):
    return np.sum(w * (y - model(th, x))**2) / np.sum(w)

result_lin = minimize(mse_loss, x0=[0, 0], args=(x, y), method='BFGS')
result_quad = minimize(huber_loss, x0=[0, 0, 0], args=(x, y), method='BFGS')
result_cube = minimize(wmse_loss, x0=[0, 0, 0, 0], args=(x, y, w), method='BFGS')

print(result_lin)
print(result_quad)
print(result_cube)

x_grid = np.linspace(x.min(), x.max(), 200)
y_fit_lin = model(result_lin.x, x_grid)
y_fit_quad = model(result_quad.x, x_grid)
y_fit_cube = model(result_cube.x, x_grid)
release_grid = x_grid + 2000 #re-add year offset for graph

plt.figure(figsize=(8, 5))
plt.scatter(yearly['release'], yearly['mean'], s=60, color='tab:blue', edgecolor='black')
plt.plot(release_grid, y_fit_lin, color='tab:red', linewidth=2,
         label=f'Linear model (MSE) (b0={result_lin.x[0]:.2f}, b1={result_lin.x[1]:.3f})')
plt.plot(release_grid, y_fit_quad, color='tab:purple', linewidth=2,
         label=f'Quadratic model (Huber (d=0.5)) (b0={result_quad.x[0]:.2f}, b1={result_quad.x[1]:.3f}, b2={result_quad.x[2]:.3f})')
plt.plot(release_grid, y_fit_cube, color='tab:green', linewidth=2,
         label=f'Cubic model (WMSE) (b0={result_cube.x[0]:.2f}, b1={result_cube.x[1]:.3f}, b2={result_cube.x[2]:.3f}, b3={result_cube.x[3]:.3f})')
plt.xlabel('Release year')
plt.ylabel('Mean yield (MT/ha)')
plt.title('GM cultivar mean yield by release year')
plt.legend()

plt.tight_layout()
plt.savefig('yearly_scatter.png', dpi=150)