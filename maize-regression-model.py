import pandas as pd, numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

reg_df = pd.read_stata('SourceData/RegData.dta')

gm_df = reg_df[reg_df['AGE'] == 1][['release', 'Yield']].dropna()

x = gm_df['release'].values - 2000 #RLYR offset
y = gm_df['Yield'].values

def model_quad(th, x):
    return th[0] + th[1]*x + th[2]*(x**2)

def model_non_linear(th, x):
    return th[0] - th[1]*np.exp(-th[2]*x) # asymptotic growth

def mse_loss(th, x, y, model):
    return np.mean((y - model(th, x))**2)

x0_quad = [y.mean(), 0, 0]
x0_nonlinear = [y.mean(), 2, 0.5]

result_quad = minimize(mse_loss, x0=x0_quad, args=(x, y, model_quad), method='BFGS')
result_nonlinear = minimize(mse_loss, x0=x0_nonlinear, args=(x, y, model_non_linear), method='BFGS')

print(f'result quad: {result_quad}')
print(f'result non linear: {result_nonlinear}')

#bootstrapping
rng = np.random.default_rng(42)
n = len(x)

def bootstrap_replicate():
    i = rng.integers(0, n, size=n) # resample rows ...
    quad = minimize(mse_loss, x0=result_quad.x, args=(x[i], y[i], model_quad), method='BFGS').x
    nonlinear = minimize(mse_loss, x0=result_nonlinear.x, args=(x[i], y[i], model_non_linear), method='BFGS').x
    return quad, nonlinear # ... and refit both models

reps = [bootstrap_replicate() for _ in range(1000)]
bs_quad = np.array([r[0] for r in reps])
bs_nonlinear = np.array([r[1] for r in reps])

# percentile intervals for the parameters
print('quad confidence interval (b0, b1, b2):\n', np.quantile(bs_quad, [0.025, 0.975], axis=0).T)
print('nonlinear confidence interval (a, b, k):\n', np.quantile(bs_nonlinear, [0.025, 0.975], axis=0).T)

# yield gain betweeen 2000 and 2019
def gain(model, th):
    return model(th, 19) - model(th, 0)

gain_quad = np.array([gain(model_quad, th) for th in bs_quad])
gain_nonlinear = np.array([gain(model_non_linear, th) for th in bs_nonlinear])
print('gain quad confidence interval:', np.quantile(gain_quad, [0.025, 0.975]))
print('gain nonlinear confidence interval:', np.quantile(gain_nonlinear, [0.025, 0.975]))
print('gain difference confidence interval (quad - nonlinear):', np.quantile(gain_quad - gain_nonlinear, [0.025, 0.975]))

x_grid = np.linspace(x.min(), x.max(), 200)
y_fit_quad = model_quad(result_quad.x, x_grid)
y_fit_nonlinear = model_non_linear(result_nonlinear.x, x_grid)
release_grid = x_grid + 2000 #re-add year offset for graph

yearly_mean = gm_df.groupby('release')['Yield'].mean()

plt.figure(figsize=(8, 5))
plt.scatter(gm_df['release'], gm_df['Yield'], s=10, alpha=0.15,
            color='tab:blue', edgecolor='none', label='Individual observations')
plt.scatter(yearly_mean.index, yearly_mean.values, s=30, color='black',
            zorder=3, label='Yearly mean')

plt.plot(release_grid, y_fit_quad, color='tab:purple', linewidth=2,
         label=f'Quadratic model (MSE) (b0={result_quad.x[0]:.2f}, b1={result_quad.x[1]:.2f}, b2={result_quad.x[2]:.2f})')

plt.plot(release_grid, y_fit_nonlinear, color='tab:orange', linewidth=2,
         label=f'Nonlinear asymptotic growth model (MSE) (a={result_nonlinear.x[0]:.2f}, b={result_nonlinear.x[1]:.2f}, k={result_nonlinear.x[2]:.3f})')

plt.xlabel('Release year')
plt.ylabel('Mean yield (MT/ha)')
plt.title('GM cultivar mean yield by release year')
plt.legend()

plt.tight_layout()
plt.savefig('yearly_scatter.png', dpi=150)