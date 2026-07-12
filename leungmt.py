import yfinance as yf
import datetime as dt
from itertools import pairwise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from scipy.integrate import quad
from scipy.optimize import brentq
import warnings
from scipy.integrate import quad, IntegrationWarning

warnings.filterwarnings("ignore", category=IntegrationWarning)


from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

def estimator(A,B):
    beta = (A @ B)/(B @ B)
    X = A - beta * B
    X = np.array(X)
    n = len(A)
    Xx = np.sum(X[:-1])
    Xy = np.sum(X[1:])
    Xxx = X[:-1] @ X[:-1]
    Xxy = X[:-1] @ X[1:]
    Xyy = X[1:] @ X[1:]

    mu = (Xy*Xxx - Xx*Xxy)/(n*(Xxx - Xxy)-(Xx**2- Xx*Xy))
    theta = -np.log((Xxy - mu * Xx - mu* Xy + n * mu**2)/(Xxx - 2 * mu * Xx + n * mu**2))

    sigma2 = ((2 * theta)/(n * (1-np.exp(-2*theta))))*(Xyy - 2*np.exp(-theta)*Xxy + 
                                             np.exp(-2*theta)*Xxx 
                                             - 2*mu*(1-np.exp(-theta))
                                            *(Xy-np.exp(-theta)*Xx) 
                                            + n*mu**2 * (1-np.exp(-theta))**2)
    

    sigma = np.sqrt(sigma2)

    
    return X, mu, theta, sigma
def plot( X,mu, theta,sigma):
    X0 = X[0]         # Initial value
    T = len(X)         # Total time
    dt = 1        # Time step
    N = int(T / dt)  # Number of time steps

    Xsim = np.zeros(N)
    Xsim[0] = X0

    # Generate the OU process
    for t in range(1, N):
        dW = np.sqrt(dt) * np.random.normal(0, 1)
        Xsim[t] = Xsim[t-1] + theta * (mu - Xsim[t-1]) * dt + sigma * dW

    # Plot the result
    dates = A.index
    fig, ax = plt.subplots()

    ax.plot(dates, X, label="Real Spread")

    ax.set_title("Ornstein-Uhlenbeck Process Simulation")
    ax.set_ylabel("X(t)")

    # Major ticks: months, minor: weeks
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    # Financial-style grid
    ax.grid(True, which="major", linestyle="-", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    fig.autofmt_xdate()
    ax.legend()
    plt.show()
    res = adfuller(X,regression="c",autolag="AIC")
    print("ADF stat: ", res[0],"P verdi: ",res[1])
    


def OU_generator( u,x, mu, theta, sigma, dx=1):
    return theta * (mu - x) * (u(x + dx) - u(x))/dx + sigma**2/2 * (u(x + 2*dx) - 2*u(x + dx) + u(x))/(dx**2)



def F(x, r, theta, mu, sigma):
    def integrand(u):
        return u**((r/theta)-1) * np.exp(np.sqrt(2*theta)/sigma * (x-mu)*u - u**2 /2)
    integral, _ = quad(integrand, 1e-8, 100)
    return integral

def G(x, r, theta, mu, sigma):
    def integrand(u):
        return u**((r/theta)-1) * np.exp(np.sqrt(2*theta)/sigma * (mu - x)*u - u**2 /2)
    integral, _ = quad(integrand, 1e-8, 100)
    return integral


def F_diff_x(x, r, theta, mu, sigma):
    def integrand(u):
        return np.sqrt(2*theta)/sigma * u**(r/theta) * np.exp(np.sqrt(2*theta)/sigma * (x-mu)*u - u**2 /2)
    integral, _ = quad(integrand, 1e-8, 100)
    return integral

def G_diff_x(x, r, theta, mu, sigma):
    def integrand(u):
        return -np.sqrt(2*theta)/sigma * u**(r/theta) * np.exp(np.sqrt(2*theta)/sigma * (mu - x)*u - u**2 /2)
    integral, _ = quad(integrand, 1e-8, 100)
    return integral


def Phi(x, r, theta, mu, sigma):
    return F(x, r, theta, mu, sigma)/G(x, r, theta, mu, sigma)


def exit_long_optimal(c, r, theta, mu, sigma):
    equation = lambda b:  F(b, r, theta, mu, sigma) - (b - c) * F_diff_x(b, r, theta, mu, sigma)
    
    sigma_x = sigma / np.sqrt(2 * theta)                    # Forventer å finne rot på intervallet her
    grid = np.linspace(mu - 3*sigma_x, mu + 3*sigma_x, 60)  ###########################################

    vals = [equation(x) for x in grid]

    for i in range(len(grid) - 1):
        if vals[i] * vals[i+1] < 0:
            return brentq(equation, grid[i], grid[i+1])

    print("No local sign change found in grid.", grid[0], grid[-1])
    return -np.inf


def entry_long_optimal(b,c, r, theta, mu, sigma):
    def V(x, b, c, r, theta, mu, sigma):
        if x < b:
            return (b - c) * F(x, r, theta, mu, sigma) / F(b, r, theta, mu, sigma)
        return x - c


    def V_diff_x(x, b, c, r, theta, mu, sigma):
        if x < b:
            return (b - c) * F_diff_x(x, r, theta, mu, sigma) / F(b, r, theta, mu, sigma)
        return 1.0
    
    
    equation = lambda d: G(d, r, theta, mu, sigma) * (V_diff_x(d, b, c, r, theta, mu, sigma) - 1) \
                        - G_diff_x(d, r, theta, mu, sigma) * (V(d, b, c, r, theta, mu, sigma) - d - c)
    sigma_x = sigma / np.sqrt(2 * theta)
    grid = np.linspace(mu - 3*sigma_x, mu + 3*sigma_x, 60)
    vals = [equation(x) for x in grid]

    for i in range(len(grid) - 1):
        if vals[i] * vals[i+1] < 0:
            return brentq(equation, grid[i], grid[i+1])

    print("No local sign change found in grid.", grid[0], grid[-1])
    return -np.inf
    


def compute_levels_for_date(date, data, Aticker, Bticker, lookback, c, r, rhat):
    cur_data = data.loc[:date].tail(lookback)
    A = cur_data[Aticker]
    B = cur_data[Bticker]
    X, mu, theta, sigma = estimator(A, B)

    # Safe defaults
    x_last = X[-1]
    enter_long = np.nan
    exit_long = np.nan
    enter_short = np.inf
    exit_short = np.inf
    skip_entry = True

    if np.isfinite(theta) and np.isfinite(sigma) and theta > (4*np.log(2) / lookback)and sigma > 0:
        #t05 = np.log(2) / theta
        #skip_entry = (t05 > lookback / 4)
        #if not skip_entry:
        skip_entry = False
        b_star = exit_long_optimal(c, r, theta, mu, sigma)
        d_star = entry_long_optimal(b_star, c, rhat, theta, mu, sigma)

        enter_long = d_star
        exit_long = b_star

        
        enter_short = np.inf
        exit_short = np.inf

    return {
        "date": date,
        "x_last": x_last,
        "mu": mu,
        "theta": theta,
        "sigma": sigma,
        "skip_entry": skip_entry,
        "enter_long": enter_long,
        "exit_long": exit_long,
        "enter_short": enter_short,
        "exit_short": exit_short,
    }
    


def main():

    FIGSIZE = (6.6, 4.2)      # Single plot size
    # FIGSIZE = (3.3, 2.4)    # side by side plot size

    TITLE_SIZE = 10
    LABEL_SIZE = 9
    TICK_SIZE = 8
    LEGEND_SIZE = 8

    LINE_WIDTH = 1.1
    THIN_LINE_WIDTH = 0.8
    MARKER_SIZE = 5

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,

        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,

        "axes.linewidth": 0.8,

        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })

    
    preday0 = dt.date(2022, 6, 1)
    day0 = dt.date(2023, 6, 1)
    endday = dt.date(2025, 6, 1)

    tickers0 = [
        "GLD",
        "SLV",
        "USO",
        "UNG",
        "CPER",
        "DBA",
        "GDX",
        "SIL",
        "COPX",
        "URA",
        "PICK",
        "XLE",
        "XOP",
        "NUE",
        "NTR",
        "ALB",
        "XOM"
           
    ]
    tickersc = ["NOKUSD=X",
        "SEKUSD=X"]
    tickers = ["GDX","GLD"]
    tickerscrypto = ["BTC-USD","ETH-USD"]

    data = yf.download(tickers, start=preday0, end=endday)
    data = data["Close"].dropna()
    
    Bticker =  "GDX"#"EURRUB=X"#"GBPUSD=X"#"SEKUSD=X"#"ETH-USD"#
    Aticker =  "GLD"#"USDRUB=X"#"EURUSD=X""NOKUSD=X"#"BTC-USD"#

    Portfolio0 = 100000.0
    Portfolio = Portfolio0

    

    A_pre = data[Aticker].loc[:day0]
    B_pre = data[Bticker].loc[:day0]

    X_pre, mu, theta, sigma = estimator(A_pre, B_pre)

    half_life = np.log(2) / theta
    lookback = int(half_life * 2)

    predates = data.loc[preday0:day0].index

    print("Mellom datoer fra:", predates[0].date(), "til", predates[-1].date())

    res1 = adfuller(X_pre, regression="c", autolag="AIC")
    print("ADF stat:", res1[0], "P verdi:", res1[1])

  

    bench_data = data.loc[day0:endday]
    dates = bench_data.index

    A_full = bench_data[Aticker]
    B_full = bench_data[Bticker]

    A0 = A_full.iloc[0]
    B0 = B_full.iloc[0]


    sharesA = 0.5 * Portfolio / A0
    sharesB = 0.5 * Portfolio / B0
    holding = "50/50"

    passive_portfolio_A = Portfolio0 * A_full.iloc[-1] / A_full.iloc[0]
    passive_portfolio_B = Portfolio0 * B_full.iloc[-1] / B_full.iloc[0]

    passive_portfolio_50_50 = (
        0.5 * Portfolio0 * A_full.iloc[-1] / A_full.iloc[0]
        + 0.5 * Portfolio0 * B_full.iloc[-1] / B_full.iloc[0]
    )

    X_trade, mu, theta, sigma = estimator(A_full, B_full)

    print("Half-life:", np.log(2) / theta)

    c = 0.0#0.0
    r = 0.01#0.05
    rhat = 0.01#0.05#

   

    jobs = list(dates)

    worker = partial(
        compute_levels_for_date,
        data=data,
        Aticker=Aticker,
        Bticker=Bticker,
        lookback=lookback,
        c=c,
        r=r,
        rhat=rhat,
    )

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(ex.map(worker, jobs, chunksize=1))

    signals = pd.DataFrame(results).set_index("date")

    

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    number_of_switches = 0

    X_plot = []
    plot_dates = []

    enter_long = []
    exit_long = []
    skip_flags = []

    for date in signals.index:
        plot_dates.append(date)
        X_plot.append(signals.loc[date, "x_last"])
        enter_long.append(signals.loc[date, "enter_long"])
        exit_long.append(signals.loc[date, "exit_long"])
        skip_flags.append(signals.loc[date, "skip_entry"])

    
    for i in range(len(X_plot)):
        if skip_flags[i] and i > 0:
            enter_long[i] = enter_long[i - 1]
            exit_long[i] = exit_long[i - 1]

  

    for i, date in enumerate(plot_dates):

        A_now = data.loc[date, Aticker]
        B_now = data.loc[date, Bticker]
        x_now = X_plot[i]

        Portfolio = sharesA * A_now + sharesB * B_now

        if not skip_flags[i]:

            # Enter long spread: all in A
            if x_now <= enter_long[i] and holding != "A":
                sharesA = Portfolio / A_now
                sharesB = 0.0
                holding = "A"
                number_of_switches += 1

                ax.plot(date, x_now, marker="x", color="green", markersize=10)

            # Exit long spread: switch all into B
            elif x_now >= exit_long[i] and holding != "B":
                sharesA = 0.0
                sharesB = Portfolio / B_now
                holding = "B"
                number_of_switches += 1

                ax.plot(date, x_now, marker="x", color="red", markersize=6)

    # Final portfolio value
    Portfolio = sharesA * A_full.iloc[-1] + sharesB * B_full.iloc[-1]

    # ---------------- Plot ----------------

    ax.plot(plot_dates, X_plot, label="Spredning", linewidth=LINE_WIDTH)

    ax.plot(
        plot_dates,
        enter_long,
        color="green",
        linewidth=THIN_LINE_WIDTH,
        label=f"Kjøp {Aticker}, d*"
    )

    ax.plot(
        plot_dates,
        exit_long,
        color="red",
        linewidth=THIN_LINE_WIDTH,
        label=f"Kjøp {Bticker}, b*"
    )

    ax.plot(
        [],
        [],
        marker="x",
        color="green",
        linestyle="None",
        markersize=MARKER_SIZE,
        label=f"All inn på {Aticker}"
    )

    ax.plot(
        [],
        [],
        marker="x",
        color="red",
        linestyle="None",
        markersize=MARKER_SIZE,
        label=f"All inn på {Bticker}"
    )

    ax.set_title("Spredning, Optimal Stopping signaler, og handler")
    ax.set_ylabel("Spredning Verdi")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.grid(True, which="major", linestyle="-", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    fig.autofmt_xdate()
    ax.legend(prop={"size": LEGEND_SIZE})

    ax.set_ylim(np.nanmin(X_plot) , np.nanmax(X_plot))
    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.18,
        top=0.88
    )
    #fig.savefig(f"figures/optimal_stopping_{Aticker}_{Bticker}_{day0}-{endday}_plot.pdf")
    plt.show()

    # ---------------- Resultat ----------------

    print("Antall handelsdager:", len(dates))
    print("Halveringstid:", round(half_life, 2), "og lengde på vinduet:", lookback)
    print("Antall bytter:", number_of_switches)
    print("Siste posisjon:", holding)

    print(f"Passiv porteføljeverdi alt i {Aticker}:", round(passive_portfolio_A, 2))
    print(f"Passiv porteføljeverdi alt i {Bticker}:", round(passive_portfolio_B, 2))
    print("Passiv porteføljeverdi 50/50:", round(passive_portfolio_50_50, 2))

    print("Aktiv OU optimal stopping porteføljeverdi:", round(Portfolio, 2))
    print("Mellom datoer fra:", dates[0].date(), "til", dates[-1].date())

    res2 = adfuller(X_plot, regression="c", autolag="AIC")
    print("ADF stat:", res2[0], "P verdi:", res2[1])




if __name__ == "__main__":
    main()