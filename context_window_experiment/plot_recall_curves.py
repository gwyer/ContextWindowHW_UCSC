import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

# ===== Wilson 95% CI for proportions =====
def wilson_ci(k, n, z=1.96):
    if n == 0: return (0,0)
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    margin = z*sqrt((p*(1-p)+z*z/(4*n))/n)/denom
    return center, margin

def bin_and_ci(df, xcol, bins=25):
    # quantile bins produce ~even counts → smoother curve
    df = df.copy()
    df['xb'] = pd.qcut(df[xcol].rank(method='first'), q=bins, duplicates='drop')
    g = df.groupby('xb', observed=True)
    out = g.agg(
        x_mid=(xcol, 'median'),
        n=('correct','size'),
        k=('correct','sum')
    ).reset_index(drop=True)
    out['p'] = out['k']/out['n']
    ci = out.apply(lambda r: wilson_ci(r['k'], r['n']), axis=1)
    out['p_hat']  = [c[0] for c in ci]
    out['p_err']  = [c[1] for c in ci]
    return out

def rolling_mean(df, xcol, w=200):
    d = df.sort_values(xcol).copy()
    d['roll'] = d['correct'].rolling(w, min_periods=max(10, w//5)).mean()
    d['xsm']  = d[xcol].rolling(w, min_periods=max(10, w//5)).median()
    return d[['xsm','roll']].dropna()

def two_segment_breakpoint(x, y):
    # simple grid search over candidate breakpoints (fast & robust)
    x = np.asarray(x); y = np.asarray(y)
    order = np.argsort(x); x = x[order]; y = y[order]
    best = (None, 1e18)
    for i in range(10, len(x)-10):
        x1,y1 = x[:i], y[:i]
        x2,y2 = x[i:], y[i:]
        # fit lines
        a1,b1 = np.polyfit(x1, y1, 1)
        a2,b2 = np.polyfit(x2, y2, 1)
        yhat = np.concatenate([a1*x1+b1, a2*x2+b2])
        sse = np.sum((np.concatenate([y1,y2]) - yhat)**2)
        if sse < best[1]:
            best = (i, sse)
    bp_idx = best[0]
    return (x[bp_idx], y[bp_idx]) if bp_idx is not None else (None, None)

# ===== Main plotting helper =====
def plot_recall_curves(csv_path, model=None):
    df = pd.read_csv(csv_path)

    if model:
        df = df[df['model']==model].copy()

    for xcol, title in [
        ('total_tokens', 'Recall vs Total Tokens (log-x)'),
        ('target_distance_tokens', 'Recall vs Token Distance (log-x)')
    ]:
        plt.figure()

        for mode in sorted(df['similarity_mode'].unique()):
            d = df[df['similarity_mode']==mode].copy()
            # Binned CI curve
            b = bin_and_ci(d, xcol, bins=25)
            plt.errorbar(b['x_mid'], b['p_hat'], yerr=b['p_err'], fmt='o', alpha=0.6, label=f'{mode} (binned CI)')
            # Rolling smoothing overlay
            r = rolling_mean(d[[xcol,'correct']], xcol, w=min(200, max(50, len(d)//20)))
            if len(r):
                plt.plot(r['xsm'], r['roll'], linewidth=2, label=f'{mode} (rolling)')

            # Breakpoint estimate on rolling line (optional)
            if len(r) > 40:
                bx, by = two_segment_breakpoint(r['xsm'].values, r['roll'].values)
                if bx is not None:
                    plt.axvline(bx, linestyle='--')
                    plt.text(bx, 0.05, f'break ~ {int(bx)}', rotation=90, va='bottom')

        plt.xscale('log')
        plt.ylim(-0.05, 1.05)
        plt.xlabel(xcol)
        plt.ylabel('Recall (exact)')
        plt.title(title + (f' — {model}' if model else ''))
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example:
# plot_recall_curves('all_results.csv')                 # all models
# plot_recall_curves('all_results.csv', model='llama3') # one model
